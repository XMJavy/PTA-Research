# -*- coding: utf-8 -*-
"""
出版级图表生成 (PNG 150dpi + PDF), 全部来自真实数据与模型输出
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats

from research import config as C

warnings.filterwarnings('ignore')
FONT_PATH = r'C:\Windows\Fonts\msyh.ttc'
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid', {'font.sans-serif': ['Microsoft YaHei']})

BLUE, ORANGE, GREEN, RED, PURPLE, GREY = '#2E6F9E', '#E8833A', '#3A9D6E', '#C0504D', '#8E6EAB', '#9AA5B1'
MCOLORS = {'岭回归': BLUE, 'LASSO': PURPLE, '随机森林': GREEN,
           'XGBoost': ORANGE, 'LightGBM': '#D4A017', 'CatBoost': RED}


def _save(fig, name):
    for ext, dpi in [('png', 160), ('pdf', None)]:
        p = os.path.join(C.FIG_DIR, f'{name}.{ext}')
        fig.savefig(p, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ==================== 1. 目标分布 ====================
def fig_target_dist(df):
    items = [(C.T_PRE, '术前肱动脉血流量 (ml/min)', BLUE),
             (C.T_POST, '术后肱动脉血流量 (ml/min)', ORANGE),
             (C.T_DFLOW, '血流量变化量 ΔQ (ml/min)', GREEN),
             (C.T_DRI, '阻力指数变化量 ΔRI', PURPLE)]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    for ax, (col, lab, color) in zip(axes.ravel(), items):
        s = df[col].dropna()
        sns.histplot(s, bins=26, kde=True, ax=ax, color=color, stat='density',
                     edgecolor='white', linewidth=0.4)
        ax.axvline(s.median(), color=RED, ls='--', lw=1.5, label=f'中位数 {s.median():.1f}')
        ax.set_title(lab, fontsize=11.5, fontweight='bold')
        ax.set_xlabel(''); ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle('图1  四个目标变量的分布', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, '01_目标变量分布')


# ==================== 2. 术前术后配对 ====================
def fig_paired(df):
    pair = df[[C.T_PRE, C.T_POST]].dropna()
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    bp = ax.boxplot([pair[C.T_PRE], pair[C.T_POST]], positions=[1, 2], widths=0.42,
                    patch_artist=True, showmeans=True,
                    medianprops=dict(color='#222', lw=1.6),
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='#222', markersize=6),
                    flierprops=dict(marker='o', markersize=3, alpha=0.35))
    for patch, c in zip(bp['boxes'], [BLUE, ORANGE]):
        patch.set_facecolor(c); patch.set_alpha(0.78)
    for (_, row) in pair.iloc[::3].iterrows():
        ax.plot([1, 2], [row[C.T_PRE], row[C.T_POST]], '-', color=GREY, alpha=0.25, lw=0.7)
    _, p = stats.wilcoxon(pair[C.T_POST], pair[C.T_PRE])
    ax.set_xticks([1, 2]); ax.set_xticklabels(['术前', '术后'], fontsize=11)
    ax.set_ylabel('肱动脉血流量 (ml/min)')
    ax.set_title(f'图2  PTA术前-术后血流量配对比较 (n={len(pair)}, Wilcoxon p={p:.2e})',
                 fontsize=12, fontweight='bold')
    _save(fig, '02_术前术后配对')
    return float(p)


# ==================== 3. 数据质量 ====================
def fig_quality(q_tables):
    miss = q_tables['01_缺失值分析'].copy()
    miss = miss[miss['缺失数'] > 0].sort_values('缺失比例').tail(12)
    out = q_tables['02_异常值检测'].copy()
    out['统计异常合计'] = out['Zscore异常数_abs_z_gt3'] + out['IQR异常数_1p5倍']
    out = out.sort_values('临床越界数', ascending=False)
    clin = out[out['临床越界数'] > 0].head(12).sort_values('临床越界数')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    if len(miss):
        axes[0].barh(miss['短名'], miss['缺失比例'] * 100, color=BLUE, alpha=0.85)
        for i, v in enumerate(miss['缺失比例'] * 100):
            axes[0].text(v + 0.03, i, f'{v:.1f}%', va='center', fontsize=8)
    axes[0].set_xlabel('缺失比例 (%)')
    axes[0].set_title('清洗后各列缺失比例 Top12', fontsize=12, fontweight='bold')
    axes[1].barh(clin['短名'], clin['临床越界数'], color=RED, alpha=0.8)
    for i, v in enumerate(clin['临床越界数']):
        axes[1].text(v + 0.05, i, str(int(v)), va='center', fontsize=8)
    axes[1].set_xlabel('按临床合理区间判为录入错误的值个数')
    axes[1].set_title('临床越界(录入错误)列 Top12', fontsize=12, fontweight='bold')
    fig.suptitle('图3  数据质量审计：缺失值与临床越界值', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, '03_数据质量审计')


# ==================== 4. VIF ====================
def fig_vif(bundle):
    tgts = list(bundle['targets'].keys())
    n = len(tgts)
    if n == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.5))
        axes = [ax]
    else:
        ncols = 2
        nrows = (n + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 4.2 * nrows))
        axes = axes.ravel()
    for ax, t in zip(axes, tgts):
        r = bundle['targets'][t]
        # 重算当前矩阵VIF
        from sklearn.linear_model import LinearRegression
        Xv = r['X']
        vifs = {}
        for col in Xv.columns:
            oth = [c for c in Xv.columns if c != col]
            r2 = LinearRegression().fit(Xv[oth], Xv[col]).score(Xv[oth], Xv[col])
            vifs[col] = 1 / (1 - r2) if r2 < 0.99999 else 1e6
        top = pd.Series(vifs).sort_values(ascending=False).head(10)[::-1]
        cols = [BLUE if v <= 5 else (ORANGE if v <= 10 else RED) for v in top.values]
        ax.barh([C.short(i) for i in top.index], top.values, color=cols, alpha=0.85)
        ax.axvline(10, color=RED, ls='--', lw=1.2)
        for i, v in enumerate(top.values):
            ax.text(v + 0.08, i, f'{v:.1f}', va='center', fontsize=7.8)
        ax.set_title(C.short(t), fontsize=11.5, fontweight='bold')
        ax.set_xlim(0, max(top.values) * 1.18)
    # 隐藏多余空子图
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle('图4  VIF 多重共线性诊断(处理后保留特征, 红线=10)', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, '04_VIF诊断')


# ==================== 5. 模型性能对比 ====================
def fig_metrics(bundle):
    tgts = list(bundle['targets'].keys())
    tnames = [C.short(t) for t in tgts]
    models = ['岭回归', 'LASSO', '随机森林', 'XGBoost', 'LightGBM', 'CatBoost']
    x = np.arange(len(tgts)); w = 0.13
    for metric, fname, title in [
        ('CV_R2均值', '05_模型对比_R2', '图5  六模型 10折交叉验证 R² 对比(误差线为标准差)'),
        ('CV_RMSE均值', '06_模型对比_RMSE', '图6  六模型 10折交叉验证 RMSE 对比(越低越好)')]:
        fig, ax = plt.subplots(figsize=(10.5, 5.0))
        for i, m in enumerate(models):
            vals, errs = [], []
            for t in tgts:
                mt = bundle['targets'][t]['metrics'].set_index('模型')
                vals.append(mt.loc[m, metric])
                sd_col = 'CV_R2标准差' if metric == 'CV_R2均值' else 'CV_RMSE标准差'
                errs.append(mt.loc[m, sd_col])
            ax.bar(x + (i - 2.5) * w, vals, w, yerr=errs, capsize=2.5,
                   label=m, color=MCOLORS[m], alpha=0.92,
                   error_kw=dict(ecolor='#444', lw=0.8, capsize=2))
        ax.set_xticks(x); ax.set_xticklabels(tnames, fontsize=10.5)
        ax.set_ylabel('R²' if 'R2' in metric else 'RMSE (ml/min 或 RI)')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(frameon=False, ncol=6, fontsize=9, loc='upper center',
                  bbox_to_anchor=(0.5, -0.12))
        if 'R2' in metric:
            ax.axhline(0.6, color=RED, ls=':', lw=1.2)
            ax.text(len(tgts) - 0.5, 0.61, 'R²=0.6 目标线', color=RED, fontsize=8.5, ha='right')
        _save(fig, fname)


# ==================== 6. 权重条形图 ====================
def fig_weights(interp):
    for t, ir in interp.items():
        d = ir['detail'].head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9.6, 6.4))
        ax.barh(d['特征短名'], d['综合权重(%)'], color=BLUE, alpha=0.88)
        for i, v in enumerate(d['综合权重(%)']):
            ax.text(v + 0.08, i, f'{v:.2f}%', va='center', fontsize=8.5)
        ax.set_xlabel('综合权重 (%, 六模型归一化等权平均)')
        ax.set_title(f'图  {C.short(t)}：影响因素综合权重 Top15',
                     fontsize=12.5, fontweight='bold')
        ax.set_xlim(0, d['综合权重(%)'].max() * 1.2)
        _save(fig, f'07_权重_{C.short(t)}')


# ==================== 7. 森林图(标准化效应+95CI) ====================
def fig_forest(interp):
    for t, ir in interp.items():
        d = ir['detail'].head(15).iloc[::-1]
        y = np.arange(len(d))
        beta = d['Ridge标准化β'].values
        lo = beta - d['Ridge_95CI下限'].values
        hi = d['Ridge_95CI上限'].values - beta
        colors = [RED if b > 0 else BLUE for b in beta]
        fig, ax = plt.subplots(figsize=(9.8, 6.6))
        ax.errorbar(beta, y, xerr=[lo, hi], fmt='o', ms=7, ecolor='#555',
                    elinewidth=1.4, capsize=3, color='none')
        for yi, b, c in zip(y, beta, colors):
            ax.plot(b, yi, 'o', ms=8, color=c, zorder=3)
        ax.axvline(0, color='#333', lw=1)
        ax.set_yticks(y); ax.set_yticklabels(d['特征短名'])
        ax.set_xlabel('标准化回归系数 β (95% Bootstrap CI, 1000次重抽样)')
        ax.set_title(f'图  {C.short(t)}：Top15 因素标准化效应量森林图\n'
                     f'红=正向影响  蓝=负向影响',
                     fontsize=12, fontweight='bold')
        _save(fig, f'08_森林图_{C.short(t)}')


# ==================== 8. 相关性热力图 ====================
def fig_heatmap(bundle, interp):
    for t, ir in interp.items():
        r = bundle['targets'][t]
        top = ir['detail']['列名'].head(12).tolist()
        sub = r['X'][top].copy()
        sub['目标'] = r['y'].values
        corr = sub.corr()
        labels = [C.short(c) for c in corr.columns]
        fig, ax = plt.subplots(figsize=(9.4, 7.8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    vmin=-1, vmax=1, square=True, linewidths=0.5, linecolor='white',
                    xticklabels=labels, yticklabels=labels,
                    annot_kws={'fontsize': 7.5}, cbar_kws={'shrink': 0.7}, ax=ax)
        plt.setp(ax.get_xticklabels(), rotation=42, ha='right', fontsize=8.5)
        plt.setp(ax.get_yticklabels(), fontsize=8.5)
        ax.set_title(f'图  {C.short(t)}：Top12 因素 Pearson 相关矩阵',
                     fontsize=12.5, fontweight='bold')
        _save(fig, f'09_相关热力图_{C.short(t)}')


# ==================== 9. SHAP 蜂群图 ====================
def fig_shap(bundle, interp):
    import shap
    for t, ir in interp.items():
        si = ir['shap']
        if 'values' not in si:
            continue
        r = bundle['targets'][t]
        feats = r['features']
        names = [C.short(f) for f in feats]
        fig = plt.figure(figsize=(9.6, 6.6))
        shap.summary_plot(si['values'], si['background'], feature_names=names,
                          show=False, max_display=15, plot_size=None)
        plt.title(f'图  {C.short(t)}：SHAP 特征贡献蜂群图（最佳模型：{r["best_name"]}）',
                  fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        _save(fig, f'10_SHAP_{C.short(t)}')


# ==================== 10. PDP 剂量-反应 ====================
def fig_pdp(interp):
    for t, ir in interp.items():
        items = list(ir['pdp'].items())[:6]
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 7))
        for ax, (f, d) in zip(axes.ravel(), items):
            ax.plot(d['grid'], d['mean'], color=ORANGE, lw=2.2)
            ax.fill_between(d['grid'], d['lo'], d['hi'], color=ORANGE, alpha=0.2,
                            label='95% Bootstrap CI')
            rug = np.quantile(d['grid'], [0.1, 0.5, 0.9])
            for q in rug:
                ax.axvline(q, color=GREY, ls=':', lw=0.8)
            ax.set_title(C.short(f), fontsize=10.5, fontweight='bold')
            ax.tick_params(labelsize=8.5)
            ax.legend(frameon=False, fontsize=7.5)
        for ax in axes.ravel()[len(items):]:
            ax.axis('off')
        fig.suptitle(f'图  {C.short(t)}：Top6 因素剂量-反应曲线(PDP, 阴影=95%CI)',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, f'11_PDP_{C.short(t)}')


# ==================== 11. 预测 vs 观测 / 残差 ====================
def fig_prediction(bundle):
    for t, r in bundle['targets'].items():
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
        r2 = 1 - np.sum(resid**2) / np.sum((yt - yt.mean())**2)
        rmse = np.sqrt(np.mean(resid**2))
        ax.set_xlabel('观测值'); ax.set_ylabel('预测值')
        ax.set_title(f'预测 vs 观测 (测试集)\nR²={r2:.3f}  RMSE={rmse:.1f}',
                     fontsize=11, fontweight='bold')
        ax.legend(frameon=False, fontsize=9)
        ax = axes[1]
        sns.histplot(resid, bins=22, kde=True, ax=ax, color=PURPLE, edgecolor='white')
        ax.axvline(0, color=RED, ls='--', lw=1.4)
        ax.set_xlabel('残差(观测 − 预测)'); ax.set_ylabel('频数')
        ax.set_title(f'残差分布 (均值={resid.mean():.2f}, 偏度={stats.skew(resid):.2f})',
                     fontsize=11, fontweight='bold')
        fig.suptitle(f'图  {C.short(t)}：最佳模型({best}) 测试集预测诊断',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        _save(fig, f'12_预测诊断_{C.short(t)}')


# ==================== 12. ROC + 混淆矩阵 ====================
def fig_roc(bundle):
    cls = bundle['classification']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    for name, cv in cls['curves'].items():
        ax.plot(cv['fpr'], cv['tpr'], lw=1.8,
                label=f"{name} (AUC={cv['auc']:.3f})",
                color=MCOLORS.get(name, GREY))
    ax.plot([0, 1], [0, 1], '--', color='#999', lw=1)
    ax.set_xlabel('1 − 特异度'); ax.set_ylabel('敏感度')
    ax.set_title('应答者(ΔQ>0)识别：10折交叉验证 ROC', fontsize=11.5, fontweight='bold')
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    cm = cls['confusion_at_youden']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['预测:未改善', '预测:改善'],
                yticklabels=['实际:未改善', '实际:改善'],
                annot_kws={'fontsize': 13})
    best = cls['best']
    row = cls['metrics'].set_index('模型').loc[best]
    ax.set_title(f'最佳模型({best}) Youden截断下混淆矩阵\n'
                 f'敏感度={row["敏感度"]:.2f}, 特异度={row["特异度"]:.2f}, '
                 f'准确率={row["准确率"]:.2f}', fontsize=11, fontweight='bold')
    fig.suptitle('图13  血流改善应答者的分类识别', fontsize=13.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, '13_ROC与混淆矩阵')


# ==================== 13. 因素聚类 ====================
def fig_cluster_factor(fc):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    ax = axes[0]
    sel = fc['select']
    ax2 = ax.twinx()
    ax.plot(sel['K'], sel['SSE'], 'o-', color=BLUE, label='SSE(肘部法)')
    ax2.plot(sel['K'], sel['轮廓系数'], 's-', color=ORANGE, label='轮廓系数')
    ax2.plot(sel['K'], (sel['CH指数'] / sel['CH指数'].max()) * sel['轮廓系数'].max(),
             '^-', color=GREEN, label='CH指数(归一)')
    ax.axvline(fc['best_k'], color=RED, ls='--', lw=1.2)
    ax.set_xlabel('聚类数 K'); ax.set_ylabel('SSE', color=BLUE)
    ax2.set_ylabel('轮廓系数', color=ORANGE)
    ax.set_title(f'K 值选择(最优 K={fc["best_k"]})', fontsize=11.5, fontweight='bold')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc='upper right')

    ax = axes[1]
    tab = fc['table']
    tgt_cols = [c for c in tab.columns if c not in
                ('列名', 'cluster', 'PC1_plot', 'PC2_plot', 'PC1', 'PC2', 'PC3')]
    tab = tab.copy()
    tab['_tot'] = tab[tgt_cols].abs().sum(axis=1)
    label_set = set(tab.sort_values('_tot', ascending=False).head(18)['列名'])
    palette = sns.color_palette('Set2', fc['best_k'])
    cnames = fc.get('cluster_names', {})
    for k in range(fc['best_k']):
        sub = tab[tab['cluster'] == k]
        cname = cnames.get(k, f'群组{k+1}')
        lab = f'{cname} (n={len(sub)})'
        ax.scatter(sub['PC1_plot'], sub['PC2_plot'], s=55, alpha=0.8,
                   color=palette[k], label=lab,
                   edgecolor='white', linewidths=0.5)
        for _, rr in sub.iterrows():
            if rr['列名'] in label_set:
                ax.annotate(C.short(rr['列名']), (rr['PC1_plot'], rr['PC2_plot']),
                            fontsize=7.5, alpha=0.9,
                            xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel(f"PC1 ({fc['explained'][0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({fc['explained'][1]*100:.1f}%)")
    ax.set_title('影响因素作用模式聚类(PCA二维投影)', fontsize=11.5, fontweight='bold')
    ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle('图14  影响因素 K-Means 聚类分析', fontsize=13.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, '14_因素聚类')


# ==================== 14. 患者聚类 ====================
def fig_cluster_patient(pc):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    ax = axes[0]
    sel = pc['select']
    ax.plot(sel['K'], sel['SSE'], 'o-', color=BLUE)
    ax2 = ax.twinx()
    ax2.plot(sel['K'], sel['轮廓系数'], 's-', color=ORANGE)
    ax.axvline(pc['best_k'], color=RED, ls='--')
    ax.set_xlabel('K'); ax.set_ylabel('SSE', color=BLUE)
    ax2.set_ylabel('轮廓系数', color=ORANGE)
    ax.set_title(f'K 值选择(最优 K={pc["best_k"]})', fontsize=11, fontweight='bold')

    ax = axes[1]
    palette = sns.color_palette('Set2', pc['best_k'])
    pcnames = pc.get('cluster_names', {})
    for k in range(pc['best_k']):
        m = pc['labels'] == k
        pname = pcnames.get(k, '')
        lab = f'表型{k+1}{"·" + pname if pname else ""} (n={m.sum()})'
        ax.scatter(pc['xy'][m, 0], pc['xy'][m, 1], s=26, alpha=0.7,
                   color=palette[k], label=lab,
                   edgecolor='white', linewidths=0.4)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('患者血流动力学表型(PCA投影)', fontsize=11, fontweight='bold')
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[2]
    av = pc['anova'].copy()
    av = av[av['P值'] < 0.05].sort_values('P值').head(10)
    if len(av):
        colors = [RED if p < 0.001 else ORANGE for p in av['P值']]
        ax.barh(av['特征'][::-1], -np.log10(av['P值'][::-1]), color=colors[::-1], alpha=0.85)
        ax.axvline(-np.log10(0.05), color='#333', ls='--', lw=1)
        ax.set_xlabel('-log10(P值)')
    ax.set_title('组间差异最显著的特征(ANOVA)', fontsize=11, fontweight='bold')
    fig.suptitle('图15  患者血流动力学表型聚类', fontsize=13.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, '15_患者表型聚类')


def render_all(df, q_tables, bundle, interp, fc, pc):
    fig_target_dist(df)
    fig_paired(df)
    fig_quality(q_tables)
    fig_vif(bundle)
    fig_metrics(bundle)
    fig_heatmap(bundle, interp)
    fig_shap(bundle, interp)
    fig_pdp(interp)
    fig_prediction(bundle)
    fig_roc(bundle)
    fig_cluster_factor(fc)
    fig_cluster_patient(pc)
    print('全部图表已生成至 outputs/figures/')
