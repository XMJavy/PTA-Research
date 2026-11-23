# -*- coding: utf-8 -*-
"""
通用「两步」影响因素分析引擎（符合临床流行病学研究规范）
========================================================
对任意连续终点 y（ΔQ = 术后−术前血流量；ΔRI = 术前−术后阻力指数）通用，流程：

1. 描述性统计 + 正态性检验（Shapiro-Wilk）
2. 单因素分析（按变量类型选择检验）：
   - 二分类：Mann-Whitney U（非正态主检验），并附报 Welch t 检验 P 值
   - 多分类：Kruskal-Wallis（非正态主检验），并附报单因素 ANOVA P 值
   - 连续：Spearman 秩相关（非正态主检验），并附报 Pearson P 值
   - 筛选 P<0.2 的候选变量进入多因素分析
3. 多因素线性回归：分类变量哑变量化、中位数填补、VIF>10 迭代剔除，
   OLS + HC3 稳健标准误；并给出 Huber 稳健回归作为非正态敏感性分析
4. 临床意义明确的交互作用检验 + 模型拟合优度（R², F）+ 残差分析
5. 6 模型综合权重：岭回归 / LASSO / 随机森林 / XGBoost / LightGBM / CatBoost
   各自重要性 min-max 归一化后等权平均，换算为百分比并从大到小排名
6. 生成森林图（标准化效应量 + 95%CI）与综合权重条形图（PNG）

调用方式：run_two_stage(target_col, cfg, make_figs=True) → 返回结果 dict。
dri_analysis.py（ΔRI）与 dq_analysis.py（ΔQ）均复用本引擎。
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research import config as C
from research import data_pipeline as DP


# ---------------- 变量类型判定 ----------------
# 心脏超声测值：连续生理测量，但取值恰为整数且水平数≤10（如 LVPW 为 7~16mm），
# 必须强制按连续变量处理，否则会被 classify_var 误判为分类变量并拆成无序哑变量。
_FORCE_CONTINUOUS = {C.C_AAO, C.C_LA, C.C_LVD, C.C_LVPW, C.C_IVS, C.C_RA, C.C_RV, C.C_EF}


def classify_var(series):
    """判定变量类型: 返回 'binary'(2水平) / 'multiclass'(3-10水平整数) / 'continuous'"""
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) == 0:
        return 'continuous'
    uniq = s.unique()
    nunique = len(uniq)
    is_int = np.allclose(uniq, np.round(uniq))
    if nunique <= 10 and is_int:
        if nunique == 2:
            return 'binary'
        return 'multiclass'
    return 'continuous'


def _calc_vif(Xd):
    """逐列 VIF（用各列对其余列回归的 R² 反算）"""
    vif = {}
    cols = Xd.columns.tolist()
    for col in cols:
        others = [c for c in cols if c != col]
        r2 = LinearRegression().fit(Xd[others].values, Xd[col].values).score(
            Xd[others].values, Xd[col].values)
        vif[col] = 1e6 if r2 >= 0.99999 else 1.0 / (1.0 - r2)
    return pd.Series(vif).sort_values(ascending=False)


def _norm01(vals, index):
    """min-max 归一化（除以最大值），同主报告 interpretation._norm01"""
    s = pd.Series(np.asarray(vals, dtype=float), index=index).fillna(0.0)
    mx = s.max()
    return s / mx if mx > 0 else s * 0.0


def run_two_stage(target_col, cfg, make_figs=True, verbose=True):
    """执行「单因素筛选 + 多因素回归」两步分析，返回结果 dict。

    参数:
        target_col: 终点列名（C.T_DFLOW 或 C.T_DRI）
        cfg:        配置 dict，键：
            target_short   终点简称（用于标题/文件名）
            desc_title     描述性统计章节标题
            formula_desc   公式文字说明
            interaction_pairs  交互对列表 [(变量A, 变量B), ...]
            fig_forest     森林图文件名（不含扩展名时的路径）
            fig_weight     权重图文件名
            forest_title   森林图标题
            weight_title   权重图标题
            pos_note       森林图正向颜色说明
            clinical_interp 显著因素临床解读 dict
        make_figs: 是否重新生成图表 PNG
        verbose:   是否打印过程
    返回: dict（desc_stats / uni_df / reg_df / inter_df / 模型与诊断指标 / 图表路径 ...）
    """
    def log(*a):
        if verbose:
            print(*a)

    cfg = cfg or {}
    target_short = cfg.get('target_short', C.short(target_col))
    fig_forest = cfg.get('fig_forest', os.path.join(C.FIG_DIR, f'{target_short}_森林图.png'))
    fig_weight = cfg.get('fig_weight', os.path.join(C.FIG_DIR, f'{target_short}_权重图.png'))
    forest_title = cfg.get('forest_title', f'{target_short} 影响因素森林图')
    weight_title = cfg.get('weight_title', f'{target_short} 影响因素综合权重排名 Top15（从大到小）')
    pos_note = cfg.get('pos_note', '红=正向影响  蓝=负向影响')
    clinical_interp = cfg.get('clinical_interp', {})
    interaction_pairs = cfg.get('interaction_pairs', [])

    # ---------------- 数据加载 ----------------
    raw = DP.load_raw()
    df = DP.clean_frame(raw)
    log(f'[{target_short}] 清洗后: {len(df)} 行, {df.shape[1]} 列')

    y = pd.to_numeric(df[target_col], errors='coerce')
    log(f'[{target_short}] 有效样本: {y.notna().sum()}, 均值={y.mean():.4f}, SD={y.std():.4f}')

    # ---------------- 特征集（仅术前基线+干预因素，与 ΔQ 一致，防数据泄露） ----------------
    fs = DP.feature_sets(df)
    features = fs[C.T_DFLOW]
    features = [c for c in features if c in df.columns and c != target_col]
    log(f'[{target_short}] 候选特征数: {len(features)}')

    var_types = {c: classify_var(df[c]) for c in features}
    # 心脏超声测值为连续生理测量（如 LVPW 恰为 7~16mm 的整数、水平数≤10），
    # 不能被 classify_var 误判为分类变量而拆成哑变量
    for c in list(var_types):
        if c in _FORCE_CONTINUOUS:
            var_types[c] = 'continuous'
    n_bin = sum(1 for v in var_types.values() if v == 'binary')
    n_multi = sum(1 for v in var_types.values() if v == 'multiclass')
    n_cont = sum(1 for v in var_types.values() if v == 'continuous')
    log(f'[{target_short}] 变量类型: 二分类={n_bin}, 多分类={n_multi}, 连续={n_cont}')

    # 正态性
    W_stat, W_p = stats.shapiro(y.dropna())
    y_normal = W_p > 0.05
    log(f'[{target_short}] 正态性: Shapiro W={W_stat:.4f}, p={W_p:.4e}, 正态={y_normal}')

    # ---------------- 1. 描述性统计 ----------------
    desc_stats = {
        'n': int(y.notna().sum()),
        'mean': float(y.mean()),
        'sd': float(y.std(ddof=1)),
        'min': float(y.min()),
        'max': float(y.max()),
        'p25': float(y.quantile(0.25)),
        'median': float(y.median()),
        'p75': float(y.quantile(0.75)),
        'shapiro_W': float(W_stat),
        'shapiro_p': float(W_p),
        'normal': y_normal,
    }
    log(f"[{target_short}] 描述: n={desc_stats['n']}, "
        f"{desc_stats['mean']:.3f}±{desc_stats['sd']:.3f}, "
        f"范围[{desc_stats['min']:.3f},{desc_stats['max']:.3f}]")

    # ---------------- 2. 单因素分析 ----------------
    log(f'\n=== [{target_short}] 单因素分析 ===')
    uni_rows = []
    df_work = df[features + [target_col]].copy()
    df_work[target_col] = pd.to_numeric(df_work[target_col], errors='coerce')
    df_work = df_work.dropna(subset=[target_col])

    for col in features:
        vtype = var_types[col]
        s = pd.to_numeric(df_work[col], errors='coerce')
        mask = s.notna()
        if mask.sum() < 10:
            continue
        x = s[mask]
        yv = df_work[target_col][mask]

        if vtype == 'binary':
            uvals = x.unique()
            if len(uvals) < 2:
                continue
            g0 = yv[x == uvals[0]]
            g1 = yv[x == uvals[1]]
            if len(g0) < 3 or len(g1) < 3:
                continue
            try:
                u_stat, u_p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
            except Exception:
                u_stat, u_p = np.nan, np.nan
            try:
                _, t_p = stats.ttest_ind(g0, g1, equal_var=False)
            except Exception:
                t_p = np.nan
            uni_rows.append({
                '变量': C.short(col), '类型': '二分类',
                '检验方法': 'Mann-Whitney U',
                '统计量': round(float(u_stat), 2) if not np.isnan(u_stat) else None,
                'P值': float(u_p),
                't检验P值': float(t_p) if not np.isnan(t_p) else None,
                '筛选(P<0.2)': '是' if (not np.isnan(u_p) and u_p < 0.2) else '否',
            })
        elif vtype == 'multiclass':
            groups = [yv[x == v].values for v in sorted(x.unique()) if (x == v).sum() >= 3]
            if len(groups) < 2:
                continue
            try:
                h_stat, kw_p = stats.kruskal(*groups)
            except Exception:
                h_stat, kw_p = np.nan, np.nan
            try:
                _, a_p = stats.f_oneway(*groups)
            except Exception:
                a_p = np.nan
            uni_rows.append({
                '变量': C.short(col), '类型': f'多分类({len(groups)}组)',
                '检验方法': 'Kruskal-Wallis',
                '统计量': round(float(h_stat), 2) if not np.isnan(h_stat) else None,
                'P值': float(kw_p),
                't检验P值': float(a_p) if not np.isnan(a_p) else None,
                '筛选(P<0.2)': '是' if (not np.isnan(kw_p) and kw_p < 0.2) else '否',
            })
        else:  # continuous
            try:
                rho, sp_p = stats.spearmanr(x, yv)
            except Exception:
                rho, sp_p = np.nan, np.nan
            try:
                r, pe_p = stats.pearsonr(x, yv)
            except Exception:
                r, pe_p = np.nan, np.nan
            uni_rows.append({
                '变量': C.short(col), '类型': '连续',
                '检验方法': 'Spearman相关',
                '统计量': round(float(rho), 4) if not np.isnan(rho) else None,
                'P值': float(sp_p),
                't检验P值': float(pe_p) if not np.isnan(pe_p) else None,
                '筛选(P<0.2)': '是' if (not np.isnan(sp_p) and sp_p < 0.2) else '否',
            })

    uni_df = pd.DataFrame(uni_rows).sort_values('P值', na_position='last').reset_index(drop=True)
    n_pass = int((uni_df['筛选(P<0.2)'] == '是').sum())
    log(f'[{target_short}] 单因素 {len(uni_df)} 个变量, P<0.2 通过 {n_pass} 个')

    # ---------------- 3. 多因素线性回归 ----------------
    log(f'\n=== [{target_short}] 多因素线性回归 ===')
    candidate_names = uni_df[uni_df['筛选(P<0.2)'] == '是']['变量'].tolist()
    name_to_col = {C.short(c): c for c in features}
    candidate_cols = [name_to_col[n] for n in candidate_names if n in name_to_col]
    log(f'[{target_short}] 多因素候选变量数: {len(candidate_cols)}')

    df_reg = df_work[candidate_cols + [target_col]].copy()
    for c in candidate_cols:
        df_reg[c] = pd.to_numeric(df_reg[c], errors='coerce')

    cat_cols = [c for c in candidate_cols if var_types[c] in ('binary', 'multiclass')]
    cont_cols = [c for c in candidate_cols if var_types[c] == 'continuous']
    log(f'[{target_short}] 分类变量 {len(cat_cols)}, 连续变量 {len(cont_cols)}')

    X_parts = []
    for c in cat_cols:
        s = df_reg[c]
        if var_types[c] == 'multiclass':
            X_parts.append(pd.get_dummies(s, prefix=C.short(c), drop_first=True).astype(float))
        else:
            X_parts.append(s.astype(float).rename(C.short(c)))
    for c in cont_cols:
        X_parts.append(df_reg[c].astype(float).rename(C.short(c)))

    X = pd.concat(X_parts, axis=1)
    y_reg = df_reg[target_col].astype(float)

    imp_model = SimpleImputer(strategy='median')
    X = pd.DataFrame(imp_model.fit_transform(X), columns=X.columns, index=X.index)
    const = [c for c in X.columns if X[c].std(ddof=0) < 1e-10]
    X = X.drop(columns=const)
    log(f'[{target_short}] 进入回归特征 {X.shape[1]} 个 (n={len(y_reg)})')

    # VIF>10 迭代剔除
    dropped_vif = []
    while True:
        vif = _calc_vif(X)
        if len(vif) == 0 or vif.iloc[0] <= 10:
            break
        col = vif.index[0]
        X = X.drop(columns=[col])
        dropped_vif.append(col)
    log(f'[{target_short}] VIF 剔除: {dropped_vif}; 最终特征 {X.shape[1]}')

    # OLS + HC3
    X_const = sm.add_constant(X)
    model_ols = sm.OLS(y_reg, X_const).fit(cov_type='HC3')
    log(f'[{target_short}] OLS: R²={model_ols.rsquared:.4f}, '
        f'Adj.R²={model_ols.rsquared_adj:.4f}, F_p={model_ols.f_pvalue:.4e}')

    # Huber 稳健回归（非正态敏感性分析）
    model_huber = sm.RLM(y_reg, X_const, M=sm.robust.norms.HuberT()).fit()

    reg_rows = []
    for name in X.columns:
        ci_low, ci_high = model_ols.conf_int().loc[name]
        p = model_ols.pvalues[name]
        reg_rows.append({
            '变量': name,
            '回归系数β': round(float(model_ols.params[name]), 4),
            '稳健SE': round(float(model_ols.bse[name]), 4),
            '95%CI下限': round(float(ci_low), 4),
            '95%CI上限': round(float(ci_high), 4),
            'P值': float(p),
            '显著性': 'P<0.05' if p < 0.05 else ('P<0.1' if p < 0.1 else 'NS'),
            'Huberβ': round(float(model_huber.params[name]), 4),
        })
    reg_df = pd.DataFrame(reg_rows).sort_values('P值').reset_index(drop=True)
    sig_vars = reg_df[reg_df['P值'] < 0.05]['变量'].tolist()
    log(f'[{target_short}] 显著影响因素 (P<0.05): {sig_vars}')

    # ---------------- 3.5 6 模型综合权重 ----------------
    log(f'\n=== [{target_short}] 6 模型综合权重 ===')
    from catboost import CatBoostRegressor
    SEED = C.RANDOM_STATE
    Xs = StandardScaler().fit_transform(X.values)
    ridge_m = RidgeCV(alphas=np.logspace(-3, 3, 100)).fit(Xs, y_reg)
    lasso_m = LassoCV(cv=5, max_iter=50000).fit(Xs, y_reg)
    rf_m = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1).fit(X.values, y_reg)
    xgb_m = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=SEED,
                         n_jobs=-1, objective='reg:squarederror', verbosity=0,
                         importance_type='gain').fit(X.values, y_reg)
    lgb_m = LGBMRegressor(n_estimators=500, learning_rate=0.05, importance_type='gain',
                          random_state=SEED, n_jobs=-1, verbose=-1).fit(X.values, y_reg)
    cat_m = CatBoostRegressor(iterations=500, depth=4, learning_rate=0.05,
                              random_seed=SEED, verbose=0,
                              allow_writing_files=False).fit(X.values, y_reg)

    imp = pd.DataFrame({
        '岭回归': _norm01(np.abs(ridge_m.coef_), X.columns),
        'LASSO': _norm01(np.abs(lasso_m.coef_), X.columns),
        '随机森林': _norm01(rf_m.feature_importances_, X.columns),
        'XGBoost': _norm01(xgb_m.feature_importances_, X.columns),
        'LightGBM': _norm01(lgb_m.feature_importances_, X.columns),
        'CatBoost': _norm01(cat_m.feature_importances_, X.columns),
    })
    comp = imp.mean(axis=1)
    weight_pct = (comp / comp.sum() * 100).round(4)
    weight_detail = imp.copy()
    weight_detail['综合权重(%)'] = weight_pct
    weight_detail = weight_detail.sort_values('综合权重(%)', ascending=False)

    reg_df = reg_df.merge(weight_pct.rename('综合权重(%)'),
                          left_on='变量', right_index=True, how='left')
    reg_df['排名'] = reg_df['综合权重(%)'].rank(ascending=False, method='min').astype(int)
    reg_df = reg_df[['排名', '变量', '回归系数β', '稳健SE', '95%CI下限',
                     '95%CI上限', 'P值', '显著性', 'Huberβ', '综合权重(%)']]

    # ---------------- 3.6 综合权重 × 标准化回归系数明细（Top15） ----------------
    # 将 X 与 y 均做 Z 标准化（ddof=0）后重新拟合 OLS+HC3：
    #   标准化β 无量纲，可横向比较不同量纲因素的相对效应大小（按标准差变化计的效应）；
    #   X、y 的尺度线性变换不改变 t 统计量，故 P 值与原始模型完全一致，
    #   此处直接沿用 model_ols.pvalues，避免重复推断产生数值噪声。
    _sx = X.std(ddof=0).replace(0.0, np.nan)
    _sy = float(y_reg.std(ddof=0))
    Xz = ((X - X.mean()) / _sx).fillna(0.0)
    yz = (y_reg - y_reg.mean()) / _sy
    model_z = sm.OLS(yz, sm.add_constant(Xz)).fit(cov_type='HC3')
    z_ci = model_z.conf_int()
    z_rows = []
    for name in X.columns:
        zb = float(model_z.params[name])
        p = float(model_ols.pvalues[name])
        z_rows.append({
            '特征因素': name,
            '综合权重(%)': float(weight_pct.get(name, np.nan)),
            '标准化β': round(zb, 4),
            '95%CI下限': round(float(z_ci.loc[name, 0]), 4),
            '95%CI上限': round(float(z_ci.loc[name, 1]), 4),
            'OLS P值': float(p),
            '显著性': 'P<0.05' if p < 0.05 else ('P<0.1' if p < 0.1 else 'NS'),
            '方向': '正向' if zb >= 0 else '负向',
        })
    weight_reg_df = pd.DataFrame(z_rows).sort_values(
        '综合权重(%)', ascending=False).reset_index(drop=True)
    weight_reg_df.insert(0, '排名', weight_reg_df['综合权重(%)']
                         .rank(ascending=False, method='min').astype(int))

    # ---------------- 4. 交互作用检验 ----------------
    log(f'\n=== [{target_short}] 交互作用检验 ===')
    inter_rows = []
    for a, b in interaction_pairs:
        if a not in X.columns or b not in X.columns:
            continue
        X_int = X_const.copy()
        X_int[f'{a}×{b}'] = X[a] * X[b]
        try:
            m_int = sm.OLS(y_reg, X_int).fit(cov_type='HC3')
            p = m_int.pvalues[f'{a}×{b}']
            inter_rows.append({
                '交互项': f'{a} × {b}',
                '回归系数': round(float(m_int.params[f'{a}×{b}']), 4),
                'P值': float(p),
                '显著性': 'P<0.05' if p < 0.05 else 'NS',
            })
        except Exception as e:
            log(f'  {a}×{b} 失败: {e}')
    inter_df = pd.DataFrame(inter_rows)

    # ---------------- 5. 模型诊断 ----------------
    log(f'\n=== [{target_short}] 模型诊断 ===')
    resid = model_ols.resid
    resid_W, resid_p = stats.shapiro(resid)
    bp_stat, bp_p = np.nan, np.nan
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(resid, X_const)
    except Exception as e:
        log(f'BP检验失败: {e}')
    outliers = int((resid.abs() > 2.5 * resid.std()).sum())
    final_vif = _calc_vif(X)
    max_vif = float(final_vif.max()) if len(final_vif) else np.nan
    log(f'[{target_short}] 残差正态 p={resid_p:.4e}; BP p={bp_p:.4e}; '
        f'异常值={outliers}; 最大VIF={max_vif:.2f}')

    # ---------------- 6. 图表 ----------------
    if make_figs:
        log(f'\n=== [{target_short}] 生成图表 ===')
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import font_manager
        import matplotlib.pyplot as plt
        FONT_PATH = r'C:\Windows\Fonts\msyh.ttc'
        font_manager.fontManager.addfont(FONT_PATH)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10
        BLUE, RED = '#2E6F9E', '#C0504D'

        # 森林图（Top15 by |β|，从大到小）
        d_plot = reg_df.copy()
        d_plot['abs_beta'] = d_plot['回归系数β'].abs()
        d_plot = d_plot.sort_values('abs_beta').tail(15)
        yy = np.arange(len(d_plot))
        beta = d_plot['回归系数β'].values
        lo = beta - d_plot['95%CI下限'].values
        hi = d_plot['95%CI上限'].values - beta
        colors = [RED if b > 0 else BLUE for b in beta]
        sig_mask = d_plot['P值'].values < 0.05
        fig, ax = plt.subplots(figsize=(10.5, 7.2))
        ax.errorbar(beta, yy, xerr=[lo, hi], fmt='o', ms=7, ecolor='#555',
                    elinewidth=1.4, capsize=3, color='none')
        for yi, b, c, sig in zip(yy, beta, colors, sig_mask):
            ax.plot(b, yi, 'o', ms=9 if sig else 6, color=c, zorder=3,
                    markeredgecolor='white', markeredgewidth=0.8)
        ax.axvline(0, color='#333', lw=1)
        ax.set_yticks(yy)
        ax.set_yticklabels(d_plot['变量'].values, fontsize=10)
        ax.set_xlabel('多因素回归系数 β (95% CI, OLS+HC3稳健SE)', fontsize=11)
        ax.set_title(f'{forest_title}\n{pos_note}  实心大圆点=P<0.05',
                     fontsize=12.5, fontweight='bold')
        plt.tight_layout()
        fig.savefig(fig_forest, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log(f'森林图已保存: {fig_forest}')

        # 权重图（6模型，从大到小）
        d_plot2 = reg_df.sort_values('综合权重(%)').tail(15)
        w_max = d_plot2['综合权重(%)'].max()
        fig, ax = plt.subplots(figsize=(10, 7.2))
        bar_colors = [RED if b > 0 else BLUE for b in d_plot2['回归系数β'].values]
        ax.barh(d_plot2['变量'], d_plot2['综合权重(%)'], color=bar_colors, alpha=0.88)
        for i, v in enumerate(d_plot2['综合权重(%)'].values):
            ax.text(v + w_max * 0.012, i, f'{v:.2f}%', va='center', fontsize=9)
        ax.set_xlabel('综合权重 (%, 6模型归一化等权平均)', fontsize=11)
        ax.set_title(f'{weight_title}\n权重=岭回归/LASSO/随机森林/XGBoost/LightGBM/CatBoost 6模型归一化等权平均  '
                     f'红=正向  蓝=负向', fontsize=12, fontweight='bold')
        ax.set_xlim(0, w_max * 1.25)
        plt.tight_layout()
        fig.savefig(fig_weight, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log(f'权重图已保存: {fig_weight}')

    # ---------------- 汇总返回 ----------------
    return {
        'target_col': target_col,
        'target_short': target_short,
        'n_rows': int(len(df)),
        'n_patients': int(df[C.C_PATIENT_ID].nunique())
        if C.C_PATIENT_ID in df.columns else 0,
        'n_features': int(len(features)),
        'n_bin': n_bin, 'n_multi': n_multi, 'n_cont': n_cont,
        'desc_stats': desc_stats,
        'uni_df': uni_df,
        'n_pass': n_pass,
        'reg_df': reg_df,
        'sig_vars': sig_vars,
        'weight_detail': weight_detail,
        'weight_reg_df': weight_reg_df,
        'n_reg_features': int(X.shape[1]),
        'n_reg': int(len(y_reg)),
        'dropped_vif': dropped_vif,
        'r2': float(model_ols.rsquared),
        'adj_r2': float(model_ols.rsquared_adj),
        'fvalue': float(model_ols.fvalue),
        'f_pvalue': float(model_ols.f_pvalue),
        'aic': float(model_ols.aic),
        'bic': float(model_ols.bic),
        'inter_df': inter_df,
        'resid_W': float(resid_W), 'resid_p': float(resid_p),
        'bp_stat': float(bp_stat), 'bp_p': float(bp_p),
        'resid_mean': float(resid.mean()), 'resid_sd': float(resid.std()),
        'resid_min': float(resid.min()), 'resid_max': float(resid.max()),
        'outliers': outliers,
        'max_vif': max_vif,
        'clinical_interp': clinical_interp,
        'forest_path': fig_forest,
        'weight_path': fig_weight,
    }
