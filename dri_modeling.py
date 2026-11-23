# -*- coding: utf-8 -*-
"""
ΔRI（阻力指数变化量）机器学习预测建模
======================================
与 ΔQ 完全相同的建模框架（research.modeling.run_one_target）：
  1. 仅术前基线 + 干预因素（data_pipeline.feature_sets 的 ΔQ 特征集，不含术后信息，防泄露）
  2. 中位数填补、VIF>10 迭代剔除（PT 与 INR 数学同源，强制剔除）
  3. 7:3 患者分组划分（GroupShuffleSplit，同一患者的手术记录不跨集）
  4. 六模型对比：岭回归 / LASSO / 随机森林 / XGBoost / LightGBM / CatBoost
  5. 全数据 GroupKFold 10 折交叉验证 + 30% 测试集评估（R²/MAE/RMSE）
  6. 最优模型（按 10 折 CV RMSE 最小选择）SHAP 特征贡献蜂群图
  7. 六模型 CV R²/RMSE 对比柱状图（带误差线）
  8. Top6 因素剂量-反应曲线（PDP, Bootstrap 95% 带）
  9. 最优模型测试集预测诊断（预测 vs 观测 + 残差分布）

用法：
    from research.dri_modeling import run_dri_model
    ml = run_dri_model(make_fig=True)
"""
import os, warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from research import config as C
from research import data_pipeline as DP
from research.modeling import run_one_target

warnings.filterwarnings('ignore')
FONT_PATH = r'C:\Windows\Fonts\msyh.ttc'
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

MCOLORS = {'岭回归': '#4E79A7', 'LASSO': '#B07AA1', '随机森林': '#59A14F',
           'XGBoost': '#ED7D31', 'LightGBM': '#F28E2B', 'CatBoost': '#C0504D'}
ORANGE = '#ED7D31'; BLUE = '#2E6F9E'; RED = '#C0504D'; GREEN = '#59A14F'
GREY = '#888'; PURPLE = '#B07AA1'

SHAP_FIG = os.path.join(C.FIG_DIR, 'ΔRI_SHAP蜂群图.png')
METRICS_R2_FIG = os.path.join(C.FIG_DIR, '05_模型对比_R2_ΔRI.png')
METRICS_RMSE_FIG = os.path.join(C.FIG_DIR, '06_模型对比_RMSE_ΔRI.png')
PDP_FIG = os.path.join(C.FIG_DIR, '11_PDP_ΔRI.png')
PRED_FIG = os.path.join(C.FIG_DIR, '12_预测诊断_ΔRI.png')


def _plot_metrics_bars(metrics_df, fig_r2, fig_rmse, fig_title_tgt='ΔRI'):
    """六模型 CV R² / RMSE 对比柱状图（ΔRI 单柱版，与 figures.fig_metrics 同风格）"""
    models = ['岭回归', 'LASSO', '随机森林', 'XGBoost', 'LightGBM', 'CatBoost']
    x = np.arange(len(models)); w = 0.6

    # --- R² ---
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    vals = [metrics_df.loc[metrics_df['模型'] == m, 'CV_R2均值'].iloc[0] for m in models]
    errs = [metrics_df.loc[metrics_df['模型'] == m, 'CV_R2标准差'].iloc[0] for m in models]
    bars = ax.bar(x, vals, w, yerr=errs, capsize=3,
                  color=[MCOLORS[m] for m in models], alpha=0.92,
                  error_kw=dict(ecolor='#444', lw=0.9, capsize=2.5))
    ax.axhline(0, color='#666', ls='-', lw=0.8)
    ax.axhline(0.6, color=RED, ls=':', lw=1.2)
    ax.text(len(models) - 0.5, 0.61, 'R²=0.6 目标线', color=RED, fontsize=8.5, ha='right')
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('R²'); ax.set_title(f'ΔRI：六模型 10 折交叉验证 R² 对比（误差线=标准差）',
                                       fontsize=12.5, fontweight='bold')
    ax.set_ylim(min(-0.5, min(vals) - max(errs) - 0.05), max(0.7, max(vals) + max(errs) + 0.05))
    plt.tight_layout()
    fig.savefig(fig_r2, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # --- RMSE ---
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    vals2 = [metrics_df.loc[metrics_df['模型'] == m, 'CV_RMSE均值'].iloc[0] for m in models]
    errs2 = [metrics_df.loc[metrics_df['模型'] == m, 'CV_RMSE标准差'].iloc[0] for m in models]
    ax.bar(x, vals2, w, yerr=errs2, capsize=3,
           color=[MCOLORS[m] for m in models], alpha=0.92,
           error_kw=dict(ecolor='#444', lw=0.9, capsize=2.5))
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('RMSE（ΔRI 单位）'); ax.set_title(f'ΔRI：六模型 10 折交叉验证 RMSE 对比（越低越好）',
                                                     fontsize=12.5, fontweight='bold')
    plt.tight_layout()
    fig.savefig(fig_rmse, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return fig_r2, fig_rmse


def _plot_pdp(r, fig_path, top_n=6, B=300):
    """最佳模型 TopN 因素 PDP 剂量-反应曲线（与 figures.fig_pdp 同风格）"""
    from research import config as C_
    best = r['best_name']
    feats = r['features']
    names = [C_.short(f) for f in feats]
    best_m = r['fitted'][best]
    use_scaled = r['inputs'][best] == 'scaled'
    Xref = r['X'].values.copy()
    n = len(Xref)
    rng = np.random.default_rng(42)
    bidx = rng.integers(0, n, size=(B, n))

    # 选 top_n 特征：按六模型综合权重（用各模型 importance min-max 归一化后平均）
    imp = pd.DataFrame(index=feats)
    for name in ['岭回归', 'LASSO', '随机森林', 'XGBoost', 'LightGBM', 'CatBoost']:
        m = r['fitted'][name]
        raw = getattr(m, 'feature_importances_', None)
        if raw is None:
            imp[name] = 0.0
        else:
            s = pd.Series(raw.astype(float)).fillna(0.0)
            mx = s.max()
            imp[name] = (s / mx if mx > 0 else s * 0.0).values
    comp = imp.mean(axis=1)
    top_feats = comp.sort_values(ascending=False).head(top_n).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7))
    for ax, f in zip(axes.ravel(), top_feats):
        j = feats.index(f)
        grid = np.quantile(Xref[:, j], np.linspace(0.02, 0.98, 24))
        pred_grid = np.zeros((len(grid), n))
        for gi, gv in enumerate(grid):
            Xp = Xref.copy()
            Xp[:, j] = gv
            Xin = r['scaler'].transform(Xp) if use_scaled else Xp
            pred_grid[gi] = best_m.predict(Xin)
        means = pred_grid.mean(axis=1)
        boot_means = pred_grid[:, bidx].mean(axis=2)
        lo, hi = np.percentile(boot_means, [2.5, 97.5], axis=1)

        ax.plot(grid, means, color=ORANGE, lw=2.2)
        ax.fill_between(grid, lo, hi, color=ORANGE, alpha=0.2,
                        label='95% Bootstrap CI')
        rug = np.quantile(grid, [0.1, 0.5, 0.9])
        for q in rug:
            ax.axvline(q, color=GREY, ls=':', lw=0.8)
        ax.set_title(C_.short(f), fontsize=10.5, fontweight='bold')
        ax.tick_params(labelsize=8.5)
        ax.legend(frameon=False, fontsize=7.5)
    for ax in axes.ravel()[len(top_feats):]:
        ax.axis('off')
    fig.suptitle('ΔRI：Top6 因素剂量-反应曲线（PDP, 阴影=95%CI）',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return fig_path


def _plot_prediction(r, fig_path):
    """最佳模型测试集预测诊断（预测 vs 观测 + 残差分布，与 figures.fig_prediction 同风格）"""
    from sklearn.metrics import r2_score
    best = r['best_name']
    p = r['pred_test'][best]
    yt = r['y_test']
    resid = yt - p
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    ax.scatter(yt, p, s=28, alpha=0.6, color=BLUE, edgecolor='white', linewidths=0.4)
    lim = [min(yt.min(), p.min()), max(yt.max(), p.max())]
    ax.plot(lim, lim, '--', color=RED, lw=1.5, label='y=x 理想线')
    z = np.polyfit(yt, p, 1)
    xx = np.linspace(lim[0], lim[1], 50)
    ax.plot(xx, np.polyval(z, xx), color=GREEN, lw=1.8, label='拟合线')
    r2 = r2_score(yt, p)
    rmse = np.sqrt(np.mean(resid**2))
    ax.set_xlabel('观测值'); ax.set_ylabel('预测值')
    ax.set_title(f'预测 vs 观测 (测试集)\nR²={r2:.3f}  RMSE={rmse:.3f}',
                 fontsize=11, fontweight='bold')
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    sns.histplot(resid, bins=22, kde=True, ax=ax, color=PURPLE, edgecolor='white')
    ax.axvline(0, color=RED, ls='--', lw=1.4)
    ax.set_xlabel('残差(观测 − 预测)'); ax.set_ylabel('频数')
    ax.set_title(f'残差分布 (均值={resid.mean():.3f}, 偏度={stats.skew(resid):.2f})',
                 fontsize=11, fontweight='bold')

    fig.suptitle(f'ΔRI：最佳模型({best}) 测试集预测诊断',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return fig_path


def run_dri_model(make_fig=True, verbose=True):
    """ΔRI 六模型对比建模 + SHAP + CV 对比柱状图 + PDP + 预测诊断，返回结果 dict。"""
    def log(msg):
        if verbose:
            print(msg, flush=True)

    # ---- 数据矩阵（与 ΔQ 同一套术前基线+干预特征） ----
    raw = DP.load_raw()
    df = DP.clean_frame(raw)
    feats = DP.feature_sets(df)[C.T_DFLOW]
    X, y, dropped, _vif = DP.prepare_matrix(
        df, feats, C.T_DRI, drop_for_vif=[C.C_PT])
    groups = df.loc[X.index, C.C_GROUP_KEY].values

    log(f'[ΔRI 机器学习] 建模矩阵: n={len(y)}, p={X.shape[1]}, '
        f'groups={len(set(groups))}（按患者序号分组）')

    # ---- 六模型对比（7:3 分组划分 + GroupKFold 10 折） ----
    r = run_one_target(C.T_DRI, X, y, groups=groups)
    metrics = r['metrics']
    best = r['best_name']
    best_row = metrics.set_index('模型').loc[best]
    log(f'[ΔRI 机器学习] 最优模型: {best}  '
        f'CV-R2={best_row["CV_R2均值"]:.3f}  '
        f'测试R2={best_row["测试集_R2"]:.3f}  '
        f'测试RMSE={best_row["测试集_RMSE"]:.3f}')

    # ---- ① 六模型 CV R² / RMSE 对比柱状图 ----
    metrics_r2_path = metrics_rmse_path = None
    if make_fig:
        metrics_r2_path, metrics_rmse_path = _plot_metrics_bars(
            metrics, METRICS_R2_FIG, METRICS_RMSE_FIG)
        log(f'[ΔRI 机器学习] CV 对比柱状图已保存')

    # ---- ② 最优模型 SHAP 蜂群图（与主报告 figures.fig_shap 完全同风格） ----
    shap_path = None
    shap_top = None
    shap_err = None
    try:
        import shap
        best_m = r['fitted'][best]
        names = [C.short(f) for f in r['features']]
        if best in ('岭回归', 'LASSO'):
            explainer = shap.LinearExplainer(
                best_m, r['scaler'].transform(r['X_test']))
            sv = np.asarray(explainer.shap_values(
                r['scaler'].transform(r['X'].values)))
            bg = r['scaler'].transform(r['X'].values)
        else:
            explainer = shap.TreeExplainer(best_m)
            sv = np.asarray(explainer.shap_values(r['X'].values))
            bg = r['X'].values

        shap_top = pd.Series(np.abs(sv).mean(axis=0), index=names)\
                     .sort_values(ascending=False)

        if make_fig:
            fig = plt.figure(figsize=(9.6, 6.6))
            shap.summary_plot(sv, bg, feature_names=names,
                              show=False, max_display=15, plot_size=None)
            plt.title(f'图  ΔRI：SHAP 特征贡献蜂群图（最佳模型：{best}）',
                      fontsize=12, fontweight='bold', pad=12)
            plt.tight_layout()
            fig.savefig(SHAP_FIG, dpi=160, bbox_inches='tight',
                        facecolor='white')
            plt.close(fig)
            shap_path = SHAP_FIG
            log(f'[ΔRI 机器学习] SHAP 蜂群图已保存: {SHAP_FIG}')
    except Exception as e:
        shap_err = str(e)
        log(f'[ΔRI 机器学习] SHAP 计算失败（不影响其他结果）: {shap_err}')

    # ---- ③ PDP Top6 剂量-反应曲线 ----
    pdp_path = None
    if make_fig:
        try:
            pdp_path = _plot_pdp(r, PDP_FIG)
            log(f'[ΔRI 机器学习] PDP 曲线已保存: {pdp_path}')
        except Exception as e:
            log(f'[ΔRI 机器学习] PDP 计算失败: {e}')

    # ---- ④ 预测诊断图 ----
    pred_path = None
    if make_fig:
        try:
            pred_path = _plot_prediction(r, PRED_FIG)
            log(f'[ΔRI 机器学习] 预测诊断图已保存: {pred_path}')
        except Exception as e:
            log(f'[ΔRI 机器学习] 预测诊断图失败: {e}')

    return {
        'metrics': metrics,
        'best_name': best,
        'best_row': best_row,
        'shap_path': shap_path,
        'shap_top': shap_top,
        'shap_error': shap_err,
        'metrics_r2_path': metrics_r2_path,
        'metrics_rmse_path': metrics_rmse_path,
        'pdp_path': pdp_path,
        'pred_path': pred_path,
        'n': int(len(y)),
        'n_features': int(X.shape[1]),
        'n_train': int(len(r['y_train'])),
        'n_test': int(len(r['y_test'])),
        'n_patients': int(len(set(groups))),
        'ks_stat': r['ks_stat'],
        'ks_p': r['ks_p'],
        'dropped_vif': dropped,
    }


if __name__ == '__main__':
    res = run_dri_model(make_fig=True)
    print('\n六模型性能:')
    print(res['metrics'].to_string(index=False))
    print('\nSHAP Top10:')
    if res['shap_top'] is not None:
        print(res['shap_top'].head(10).to_string())
    print('\n图表路径:')
    for k in ['shap_path', 'metrics_r2_path', 'metrics_rmse_path', 'pdp_path', 'pred_path']:
        print(f'  {k}: {res.get(k)}')
