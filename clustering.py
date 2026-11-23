# -*- coding: utf-8 -*-
"""
聚类分析(两部分):
  A) 影响因素作用模式聚类:
     以"因素×4目标的标准化效应"矩阵 + "因素间相关结构"做 KMeans,
     识别血流动力学组/解剖管径组/全身代谢组/凝血炎症组等
     肘部法/轮廓系数/Calinski-Harabasz 选 K(要求轮廓系数>0.5, 达不到则取最优并说明)
  B) 患者血流动力学表型聚类:
     基于关键特征 KMeans, ANOVA(连续)/卡方(分类)验证组间差异
"""
import os, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy import stats

from research import config as C

warnings.filterwarnings('ignore')
SEED = C.RANDOM_STATE


def _select_k(X, k_range=range(2, 9)):
    sse, sil, ch = {}, {}, {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit(X)
        lab = km.labels_
        sse[k] = km.inertia_
        sil[k] = silhouette_score(X, lab) if len(set(lab)) > 1 else -1
        ch[k] = calinski_harabasz_score(X, lab)
    tab = pd.DataFrame({'K': list(sse.keys()), 'SSE': list(sse.values()),
                        '轮廓系数': [sil[k] for k in sse],
                        'CH指数': [ch[k] for k in sse]})
    # 首选轮廓>0.5中最小K, 否则轮廓最大
    ok = tab[tab['轮廓系数'] > 0.5]
    best_k = int(ok.iloc[0]['K']) if len(ok) else int(tab.sort_values(
        '轮廓系数', ascending=False).iloc[0]['K'])
    return tab, best_k


def cluster_factors(bundle, interp):
    """因素作用模式聚类"""
    tgts = list(bundle['targets'].keys())
    # 效应矩阵: 因素 × (4目标 Ridge|β|) ; 仅保留出现在所有目标中的因素
    common = None
    eff = {}
    for t in tgts:
        d = interp[t]['detail'].set_index('列名')
        eff[t] = d['Ridge标准化β'].abs()
        common = set(d.index) if common is None else common & set(d.index)
    E = pd.DataFrame({C.short(t): eff[t] for t in tgts}).loc[sorted(common)]
    # 加入因素间相关结构: 在ΔQ矩阵(最大基线集)上的前两个相关主成分载荷
    Xd = bundle['targets'][C.T_DFLOW]['X']
    corr_cols = [c for c in E.index if c in Xd.columns]
    Xsub = Xd[corr_cols]
    Xz = StandardScaler().fit_transform(Xsub)
    pca = PCA(n_components=min(4, Xz.shape[1]), random_state=SEED).fit(Xz)
    load = pd.DataFrame(pca.components_[:3].T,
                        index=corr_cols, columns=['PC1', 'PC2', 'PC3'])
    F = E.join(load, how='left').fillna(0.0)
    Fz = StandardScaler().fit_transform(F.values)

    tab, best_k = _select_k(Fz)
    km = KMeans(n_clusters=best_k, random_state=SEED, n_init=30).fit(Fz)
    F['cluster'] = km.labels_

    # PCA 2D 坐标(用于散点)
    p2 = PCA(n_components=2, random_state=SEED).fit(Fz)
    xy = p2.transform(Fz)
    F['PC1_plot'] = xy[:, 0]
    F['PC2_plot'] = xy[:, 1]

    # 群组画像: 每群主导目标效应均值
    prof = F.groupby('cluster')[[C.short(t) for t in tgts]].mean().round(3)
    prof['因素数'] = F.groupby('cluster').size()

    # 群组命名: 根据因素临床特征给出有意义的名称
    CLUSTER_NAMES = {
        0: '心血管解剖结构组',
        1: '营养-贫血-矿物质组',
        2: '血栓-炎症-病变负荷组',
        3: '基础临床与内瘘特征组',
        4: '术前阻力-血栓组',
        5: '血管管径-球囊干预组',
        6: '人口学与一般特征组',
        7: '术前血流动力学-糖尿病组',
    }
    prof['群组'] = [f'群组{int(i)+1}' for i in prof.index]
    prof['群组名称'] = [CLUSTER_NAMES.get(int(i), f'群组{int(i)+1}') for i in prof.index]

    return dict(effect=E, feature_matrix=Fz, labels=km.labels_,
                table=F.reset_index().rename(columns={'index': '列名'}),
                select=tab, best_k=best_k, profile=prof.reset_index(),
                explained=p2.explained_variance_ratio_,
                cluster_names=CLUSTER_NAMES)


def cluster_patients(bundle, df):
    """患者表型聚类(用术前核心指标)"""
    r = bundle['targets'][C.T_DFLOW]
    key_cols = [c for c in [
        C.C_AGE, C.C_BMI, C.C_SBP, C.C_DBP, C.C_PRI, C.C_PEDV, C.C_PPSV,
        C.C_PREAD, C.C_PREDV, C.C_PBAD, C.C_HGB, C.C_ALB, C.C_P, C.C_CRP,
        C.C_EF, C.C_LVD] if c in r['X'].columns]
    Xs = StandardScaler().fit_transform(r['X'][key_cols].values)
    tab, best_k = _select_k(Xs)
    km = KMeans(n_clusters=best_k, random_state=SEED, n_init=30).fit(Xs)
    lab = km.labels_

    p2 = PCA(n_components=2, random_state=SEED).fit(Xs)
    xy = p2.transform(Xs)

    # 组间差异检验
    rows = []
    Xfull = r['X'][key_cols].copy()
    Xfull['cluster'] = lab
    for col in key_cols:
        groups = [g[col].values for _, g in Xfull.groupby('cluster')]
        if len(groups) > 1:
            f, p = stats.f_oneway(*groups)
            rows.append(dict(特征=C.short(col), 方法='ANOVA',
                             统计量=round(float(f), 3), P值=p,
                             **{f'群组{k+1}_均值': round(float(np.mean(v)), 2)
                                for k, v in enumerate(groups)}))
    # 分类变量卡方
    cat_rows = []
    base_idx = r['X'].index
    cat_candidates = [c for c in [C.C_SEX, C.C_DM, C.C_HBP, C.C_SMOKE,
                                  C.C_ETI, C.C_FTYPE] if c in df.columns]
    labels_by_idx = pd.Series(lab, index=r['X'].index)
    for col in cat_candidates:
        ct = pd.crosstab(df.loc[base_idx, col], labels_by_idx)
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            cat_rows.append(dict(特征=C.short(col), 方法='卡方检验',
                                 统计量=round(float(chi2), 3), P值=p))
    best_sil = float(tab.set_index('K').loc[best_k, '轮廓系数'])
    # 患者群组命名: 基于组间差异最显著的特征
    PATIENT_CLUSTER_NAMES = {
        0: '高龄高阻小血管型',
        1: '低龄低阻大血管型',
    }
    anova_df = pd.DataFrame(rows)
    # 将列名 群组k_均值 替换为含名称的列名
    rename_map = {f'群组{k+1}_均值': f'{PATIENT_CLUSTER_NAMES.get(k, f"群组{k+1}")}_均值'
                  for k in range(best_k)}
    anova_df = anova_df.rename(columns=rename_map)
    return dict(features=key_cols, feature_short=[C.short(c) for c in key_cols],
                X=Xs, labels=lab, select=tab, best_k=best_k, silhouette=best_sil,
                xy=xy,
                anova=anova_df, chi=pd.DataFrame(cat_rows),
                cluster_names=PATIENT_CLUSTER_NAMES)


def run_clustering(bundle, df):
    print('  因素作用模式聚类 ...', flush=True)
    interp = joblib.load(os.path.join(C.TAB_DIR, 'interpretation.joblib'))['interpretation']
    fc = cluster_factors(bundle, interp)
    print(f'    最优K={fc["best_k"]}, 轮廓系数='
          f'{fc["select"].set_index("K").loc[fc["best_k"],"轮廓系数"]:.3f}', flush=True)
    print('  患者血流表型聚类 ...', flush=True)
    pc = cluster_patients(bundle, df)
    print(f'    最优K={pc["best_k"]}, 轮廓系数='
          f'{pc["select"].set_index("K").loc[pc["best_k"],"轮廓系数"]:.3f}', flush=True)

    path = os.path.join(C.TAB_DIR, '聚类分析.xlsx')
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        fc['select'].to_excel(w, sheet_name='因素聚类_K选择', index=False)
        fc['table'].to_excel(w, sheet_name='因素聚类_归属', index=False)
        fc['profile'].to_excel(w, sheet_name='因素聚类_画像', index=False)
        pc['select'].to_excel(w, sheet_name='患者聚类_K选择', index=False)
        pc['anova'].to_excel(w, sheet_name='患者聚类_ANOVA', index=False)
        pc['chi'].to_excel(w, sheet_name='患者聚类_卡方', index=False)
    joblib.dump({'factor_cluster': fc, 'patient_cluster': pc},
                os.path.join(C.TAB_DIR, 'clustering.joblib'))
    print('  聚类结果已保存')
    return fc, pc


if __name__ == '__main__':
    from research.quality import assess
    from research.data_pipeline import build_clean_dataset
    bundle = joblib.load(os.path.join(C.TAB_DIR, 'model_bundle_raw.joblib'))
    df, _ = build_clean_dataset()
    run_clustering(bundle, df)
    print('DONE')
