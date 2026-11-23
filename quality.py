# -*- coding: utf-8 -*-
"""
数据质量评估:
  - 缺失值分析(每列缺失比例, 原始/清洗后)
  - 异常值检测: Z-score(|z|>3) + IQR(1.5倍) + 临床合理区间
  - 分布检验: Shapiro-Wilk 正态性、偏度/峰度、转换建议
  - 填补前后关键统计量漂移(<5%)
  - VIF 多重共线性诊断(配合 data_pipeline.prepare_matrix)
所有结果落盘 outputs/tables, 供报告与审计使用。
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer

from research import config as C
from research.data_pipeline import build_clean_dataset, load_raw, feature_sets, prepare_matrix


def assess():
    raw = load_raw()
    df, audit = build_clean_dataset()
    fsets = feature_sets(df)
    tables = {}

    # ---------- 1) 缺失分析 ----------
    rows = []
    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        rows.append(dict(
            列名=col, 短名=C.short(col),
            数据类型='分类/编码' if col in C.CATEGORICAL_COLS else '连续',
            非空数=int(df[col].notna().sum()),
            缺失数=n_miss,
            缺失比例=round(n_miss / len(df), 4),
        ))
    miss_tab = pd.DataFrame(rows).sort_values('缺失比例', ascending=False)
    tables['01_缺失值分析'] = miss_tab

    # ---------- 2) 异常值检测(清洗后、填补前) ----------
    clin = audit['clinical_outliers'].copy()
    clin_map = dict(zip(clin['列名'], clin['越界数'])) if len(clin) else {}
    rows = []
    for col in df.columns:
        if col == C.C_GROUP_KEY:
            continue
        try:
            s = df[col].dropna().astype(float)
        except (ValueError, TypeError):
            continue
        if len(s) < 8 or s.nunique() < 3:
            continue
        z = np.abs((s - s.mean()) / s.std(ddof=0)) if s.std(ddof=0) > 0 else s*0
        n_z = int((z > 3).sum())
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        n_iqr = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()) if iqr > 0 else 0
        rows.append(dict(
            列名=col, 短名=C.short(col), n=len(s), 均值=round(s.mean(), 3),
            标准差=round(s.std(), 3), 最小值=s.min(), 最大值=s.max(),
            Zscore异常数_abs_z_gt3=n_z,
            IQR异常数_1p5倍=n_iqr,
            临床越界数=int(clin_map.get(col, 0)),
        ))
    out_tab = pd.DataFrame(rows).sort_values('临床越界数', ascending=False)
    tables['02_异常值检测'] = out_tab

    # ---------- 3) 正态性检验 & 转换建议 ----------
    rows = []
    for col in df.columns:
        if col == C.C_GROUP_KEY:
            continue
        if col in (C.T_PRE, C.T_POST, C.T_DFLOW, C.T_DRI):
            continue
        try:
            s = df[col].dropna().astype(float)
        except (ValueError, TypeError):
            continue
        if len(s) < 8 or s.nunique() < 5 or col in C.CATEGORICAL_COLS:
            continue
        try:
            W, p = stats.shapiro(s.values[:5000])
        except Exception:
            W, p = np.nan, np.nan
        skew = stats.skew(s)
        kurt = stats.kurtosis(s)
        if abs(skew) > 1 and (s > 0).all():
            trans = '建议 log1p 对数转换'
        elif abs(skew) > 1:
            trans = '右偏含非正值, 建议分位数/稳健方法'
        elif p < 0.05:
            trans = '轻度偏离正态, Z-score标准化即可'
        else:
            trans = '近似正态, Z-score标准化'
        rows.append(dict(列名=col, 短名=C.short(col), Shapiro_W=round(W, 4),
                         正态性P=('%.2e' % p) if p == p else 'NA',
                         偏度=round(skew, 3), 峰度=round(kurt, 3),
                         转换建议=trans))
    norm_tab = pd.DataFrame(rows)
    tables['03_分布检验'] = norm_tab

    # ---------- 4) 中位数填补前后统计量漂移 ----------
    key_cols = [c for c in [C.T_PRE, C.T_POST, C.C_PRI, C.C_POSTRI, C.C_PEDV,
                            C.C_POSTEDV, C.C_PBAD, C.C_AGE, C.C_BMI, C.C_HGB,
                            C.C_ALB, C.C_SBP, C.C_DBP, C.C_DD, C.C_CRP]
                if c in df.columns]
    rows = []
    for col in key_cols:
        s = df[col]
        n_miss = int(s.isna().sum())
        imp = SimpleImputer(strategy='median')
        s2 = pd.Series(imp.fit_transform(s.values.reshape(-1, 1)).ravel())
        for stat, f in [('均值', 'mean'), ('中位数', 'median'), ('标准差', 'std')]:
            v0, v1 = getattr(s, f)(), getattr(s2, f)()
            drift = abs(v1 - v0) / abs(v0) if v0 not in (0, np.nan) and v0 == v0 else 0
            rows.append(dict(列名=C.short(col), 统计量=stat, 填补前=round(v0, 3),
                             填补后=round(v1, 3), 缺失数=n_miss,
                             相对漂移=round(drift, 5),
                             是否小于5pct='是' if drift < 0.05 else '否'))
    imp_tab = pd.DataFrame(rows)
    tables['04_填补漂移'] = imp_tab

    # ---------- 5) VIF (4个目标各自的建模矩阵) ----------
    vif_rows = []
    dropped_rows = []
    matrices = {}
    # 已知数学同源: INR由PT计算 → 所有模型统一剔除PT
    global_drop = [C.C_PT]
    # 分组键: GroupKFold 保证同一患者所有PTA记录在同一折, 防泄露
    has_group = C.C_GROUP_KEY in df.columns
    for tgt, feats in fsets.items():
        X, y, dropped, vif = prepare_matrix(df, feats, tgt, drop_for_vif=global_drop)
        groups = df.loc[X.index, C.C_GROUP_KEY].values if has_group else None
        matrices[tgt] = (X, y, groups)
        for col, v in vif.items():
            vif_rows.append(dict(目标=C.short(tgt), 特征=C.short(col), VIF=round(v, 3),
                                 是否大于10='是' if v > 10 else '否'))
        for d in dropped:
            dropped_rows.append(dict(目标=C.short(tgt), 被剔除特征=C.short(d),
                                     原因='与其他特征数学同源/高度共线(VIF>10)'))
    tables['05_VIF诊断'] = pd.DataFrame(vif_rows)
    tables['06_VIF剔除记录'] = pd.DataFrame(dropped_rows)

    # ---------- 落盘 ----------
    xlsx_path = os.path.join(C.TAB_DIR, '数据质量评估.xlsx')
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as w:
        for name, tab in tables.items():
            tab.to_excel(w, sheet_name=name[:31], index=False)
    # 清洗后数据(供复用)
    df.to_excel(os.path.join(C.TAB_DIR, '清洗后分析数据集.xlsx'), index=False)

    n_patients = int(df[C.C_GROUP_KEY].nunique()) if C.C_GROUP_KEY in df.columns else len(df)
    summary = dict(
        n=len(df), n_patients=n_patients,
        n_features_raw=audit['n_cols_raw'], n_features_clean=df.shape[1],
        n_missing_cols=int((miss_tab['缺失数'] > 0).sum()),
        n_clinical_outlier_cols=int((out_tab['临床越界数'] > 0).sum()) if len(out_tab) else 0,
        n_clinical_outlier_vals=int(out_tab['临床越界数'].sum()) if len(out_tab) else 0,
        n_z_outlier_cols=int((out_tab['Zscore异常数_abs_z_gt3'] > 0).sum()) if len(out_tab) else 0,
        n_iqr_outlier_cols=int((out_tab['IQR异常数_1p5倍'] > 0).sum()) if len(out_tab) else 0,
        target_missing={C.short(k): int(df[k].isna().sum())
                        for k in C.TARGETS},
    )
    return df, tables, matrices, summary


if __name__ == '__main__':
    df, tables, matrices, summary = assess()
    print(summary)
    for k, t in tables.items():
        print(k, t.shape)
    print('模型矩阵:')
    for k, (X, y, g) in matrices.items():
        print(' ', C.short(k), X.shape, len(y),
              f'group={len(set(g)) if g is not None else "无"}')
