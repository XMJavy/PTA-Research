# -*- coding: utf-8 -*-
"""
模型解释:
  1) 标准化 OLS 系数 + 稳健P值(HC3) + 95%CI
  2) Ridge 标准化系数 + Bootstrap(1000次)95%CI
  3) 6 模型归一化权重 → 综合权重
  4) Kendall's W 排序一致性
  5) SHAP(最佳模型)
  6) Top10 特征 PDP 剂量-反应曲线 + Bootstrap 95% 带
"""
import os, warnings
import numpy as np
import pandas as pd
import joblib
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.linear_model import Ridge, Lasso

from research import config as C

warnings.filterwarnings('ignore')
SEED = C.RANDOM_STATE
MODEL_ORDER = ['岭回归', 'LASSO', '随机森林', 'XGBoost', 'LightGBM', 'CatBoost']


def _norm01(s):
    s = pd.Series(s).fillna(0.0).astype(float)
    mx = s.max()
    return s / mx if mx > 0 else s * 0.0


def _scaled_xy(r):
    sc = StandardScaler().fit(r['X'].values)
    Xs = sc.transform(r['X'].values)
    ys = (r['y'].values - r['y'].values.mean()) / r['y'].values.std(ddof=0)
    return Xs, ys, sc


def interpret_target(tgt, r, B_boot=1000):
    feats = r['features']
    X, y = r['X'].values, r['y'].values
    Xs, ys, sc = _scaled_xy(r)
    n, p = Xs.shape
    out = {}

    # ---------- 1) OLS 推断(标准化) ----------
    Xc = sm.add_constant(Xs, has_constant='add')
    ols = sm.OLS(ys, Xc).fit(cov_type='HC3')
    ols_beta = ols.params[1:]
    ols_ci = ols.conf_int()[1:]
    ols_p = ols.pvalues[1:]

    # ---------- 2) 完全标准化空间下的 Ridge / LASSO (惩罚参数需在标准化y上重选) ----------
    from sklearn.linear_model import RidgeCV, LassoCV
    # y 已标准化, 原模型 alpha(按原始y尺度选出)不可直接复用
    rcv = RidgeCV(alphas=np.logspace(-3, 3, 100)).fit(Xs, ys)
    alpha = float(rcv.alpha_)
    ridge = Ridge(alpha=alpha).fit(Xs, ys)
    r_beta = ridge.coef_
    lcv = LassoCV(cv=10, max_iter=50000).fit(Xs, ys)
    lasso_beta = lcv.coef_

    rng = np.random.default_rng(SEED)
    boot = np.zeros((B_boot, p))
    for b in range(B_boot):
        idx = rng.integers(0, n, n)
        boot[b] = Ridge(alpha=alpha).fit(Xs[idx], ys[idx]).coef_
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5], axis=0)
    sig_boot = ~((ci_lo <= 0) & (ci_hi >= 0))

    # ---------- 3) 各模型重要性 ----------
    imp = pd.DataFrame(index=feats)
    imp['岭回归'] = np.abs(r_beta)
    imp['LASSO'] = np.abs(lasso_beta)
    for name in ['随机森林', 'XGBoost', 'LightGBM', 'CatBoost']:
        m = r['fitted'][name]
        imp[name] = getattr(m, 'feature_importances_', np.zeros(p))

    # 最佳模型排列重要性(测试集)
    best = r['best_name']
    best_m = r['fitted'][best]
    scaled_in = r['inputs'][best] == 'scaled'
    pin = permutation_importance(
        best_m,
        sc.transform(r['X_test']) if scaled_in else r['X_test'],
        r['y_test'], n_repeats=20, random_state=SEED, n_jobs=-1,
        scoring='neg_root_mean_squared_error')
    imp['最佳模型排列重要性'] = np.abs(pin.importances_mean)

    # 综合权重: 6 模型等权(归一化后平均), 再换算百分比
    norm_imp = pd.DataFrame({m: _norm01(imp[m]) for m in MODEL_ORDER}, index=feats)
    comp = norm_imp.mean(axis=1)
    comp_pct = comp / comp.sum() * 100

    detail = pd.DataFrame({
        '特征短名': [C.short(f) for f in feats],
        '列名': feats,
        'OLS标准化β': np.round(ols_beta, 4),
        'OLS_P值(HC3)': ols_p,
        'OLS_95CI下限': np.round(ols_ci[:, 0], 4),
        'OLS_95CI上限': np.round(ols_ci[:, 1], 4),
        'Ridge标准化β': np.round(r_beta, 4),
        'Ridge_95CI下限': np.round(ci_lo, 4),
        'Ridge_95CI上限': np.round(ci_hi, 4),
        'Bootstrap显著(0不在CI)': sig_boot.astype(int),
        'LASSO标准化β': np.round(lasso_beta, 4),
        '随机森林重要性': np.round(imp['随机森林'].values, 4),
        'XGBoost重要性': np.round(imp['XGBoost'].values, 4),
        'LightGBM重要性': np.round(imp['LightGBM'].values, 4),
        'CatBoost重要性': np.round(imp['CatBoost'].values, 4),
        '排列重要性_RMSE增加': np.round(pin.importances_mean, 4),
        '综合权重(%)': np.round(comp_pct.values, 4),
    }).sort_values('综合权重(%)', ascending=False)

    # ---------- 4) Kendall's W (6个模型=评分者, n个特征=被评对象) ----------
    # 直接按定义计算(含结校正); 不能用 friedmanchisquare(各模型秩和恒等会得χ²=0)
    from scipy.stats import rankdata
    m_k = len(MODEL_ORDER)
    n_it = p
    # 每位评分者(模型)内部用平均秩(1=最重要), 结获得平均秩
    ranks = np.column_stack([
        rankdata(-norm_imp[m].values, method='average') for m in MODEL_ORDER])
    R_i = ranks.sum(axis=1)                       # 每个特征的秩和
    S = np.sum((R_i - m_k * (n_it + 1) / 2) ** 2)
    # 结校正量 T_j = Σ(t³−t)/12
    T_total = 0.0
    for j in range(m_k):
        _, counts = np.unique(ranks[:, j], return_counts=True)
        tie_counts = counts[counts > 1]
        T_total += np.sum(tie_counts ** 3 - tie_counts) / 12.0
    denom = (m_k ** 2 * (n_it ** 3 - n_it)) / 12.0 - m_k * T_total
    kendall_w = float(S / denom) if denom > 0 else 0.0
    chi2 = m_k * (n_it - 1) * kendall_w
    kw_p = float(stats.chi2.sf(chi2, df=n_it - 1))

    # 两两 Kendall τ-b (对结稳健)
    taus = {}
    for i in range(m_k):
        for j in range(i + 1, m_k):
            tau, _ = stats.kendalltau(ranks[:, i], ranks[:, j])
            taus[f'{MODEL_ORDER[i]} vs {MODEL_ORDER[j]}'] = round(float(tau), 4)

    out['detail'] = detail
    out['kendall_w'] = kendall_w
    out['kendall_chi2'] = float(chi2)
    out['kendall_p'] = kw_p
    out['kendall_tau'] = taus

    # ---------- 5) SHAP ----------
    shap_info = {}
    try:
        import shap
        if best in ('岭回归', 'LASSO'):
            explainer = shap.LinearExplainer(best_m, sc.transform(r['X_test']))
            sv = explainer.shap_values(sc.transform(r['X'].values))
            bg = sc.transform(r['X'].values)
        else:
            explainer = shap.TreeExplainer(best_m)
            sv = explainer.shap_values(r['X'].values)
            bg = r['X'].values
        sv = np.asarray(sv)
        shap_info['values'] = sv
        shap_info['background'] = bg
        shap_info['explainer'] = explainer
        shap_info['mean_abs'] = pd.Series(np.abs(sv).mean(axis=0), index=feats)
    except Exception as e:
        shap_info['error'] = str(e)
    out['shap'] = shap_info

    # ---------- 6) PDP Top10 (最佳模型, bootstrap 95% 带, 向量化重抽样) ----------
    top10 = detail['列名'].head(10).tolist()
    pdp_store = {}
    rng2 = np.random.default_rng(SEED)
    Xref = r['X'].values.copy()
    use_scaled = r['inputs'][best] == 'scaled'
    B = 300
    bidx = rng2.integers(0, n, size=(B, n))  # 预生成重抽样索引
    for f in top10:
        j = feats.index(f)
        grid = np.quantile(Xref[:, j], np.linspace(0.02, 0.98, 24))
        means = np.zeros(len(grid))
        lo = np.zeros(len(grid)); hi = np.zeros(len(grid))
        # 每个网格点: 全体患者该特征替换为网格值 → 预测 → 对预测均值做Bootstrap
        pred_grid = np.zeros((len(grid), n))
        for gi, gv in enumerate(grid):
            Xp = Xref.copy()
            Xp[:, j] = gv
            Xin = sc.transform(Xp) if use_scaled else Xp
            pred_grid[gi] = best_m.predict(Xin)
            means[gi] = pred_grid[gi].mean()
        boot_means = pred_grid[:, bidx].mean(axis=2)  # (n_grid, B)
        lo, hi = np.percentile(boot_means, [2.5, 97.5], axis=1)
        pdp_store[f] = dict(grid=grid, mean=means, lo=lo, hi=hi)
    out['pdp'] = pdp_store
    return out


def run_interpretation(bundle):
    res = {}
    rows = []
    for tgt, r in bundle['targets'].items():
        print(f'  解释分析: {C.short(tgt)} ...', flush=True)
        ir = interpret_target(tgt, r)
        res[tgt] = ir
        rows.append(dict(目标=C.short(tgt),
                         Kendall_W=round(ir['kendall_w'], 4),
                         卡方=round(ir['kendall_chi2'], 2),
                         P值=('%.2e' % ir['kendall_p']),
                         结论='排序一致性强' if ir['kendall_w'] > 0.5
                         else ('中等一致' if ir['kendall_w'] > 0.3 else '一致性弱')))
        # 明细落盘
        ir['detail'].to_excel(
            os.path.join(C.TAB_DIR, f'权重明细_{C.short(tgt)}.xlsx'), index=False)
        # PDP 数据
        np.savez_compressed(
            os.path.join(C.TAB_DIR, f'pdp_{C.short(tgt)}.npz'),
            **{f'{f}__{k}': v for f, d in ir['pdp'].items()
               for k, v in d.items()})

    kw_tab = pd.DataFrame(rows)
    # 汇总 Kendall τ
    first_tgt = bundle['targets'] and list(bundle['targets'].keys())[0]
    tau_tab = pd.DataFrame([
        dict(模型对=k, Kendall_tau=v)
        for k, v in res[first_tgt]['kendall_tau'].items()])

    path = os.path.join(C.TAB_DIR, '排序一致性与权重汇总.xlsx')
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        kw_tab.to_excel(w, sheet_name="Kendall_W", index=False)
        tau_tab.to_excel(w, sheet_name='两两Kendall_tau', index=False)
        for tgt, ir in res.items():
            ir['detail'].head(25).to_excel(
                w, sheet_name=f'Top25_{C.short(tgt)}'[:28], index=False)

    joblib.dump({'interpretation': res},
                os.path.join(C.TAB_DIR, 'interpretation.joblib'))
    print('  解释结果已保存')
    return res


if __name__ == '__main__':
    bundle = joblib.load(os.path.join(C.TAB_DIR, 'model_bundle_raw.joblib'))
    run_interpretation(bundle)
    print('DONE')
