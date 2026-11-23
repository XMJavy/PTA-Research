# -*- coding: utf-8 -*-
"""
数据加载、清洗、特征工程
输出:
  clean_df      —— 清洗+特征工程后的分析数据集
  quality       —— 数据质量审计信息(缺失/异常/修改记录)
"""
import re
import numpy as np
import pandas as pd

from research import config as C


def _coerce_scalar(v, first_code=False):
    """单值清洗: '/'等→NaN; '<6'→3; '1+4'→求和或取首编码"""
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip().replace('\n', '')
    if s in ('', '/', '\\', 'nan', 'NaN', 'None', '－', '-', '--'):
        return np.nan
    if s.startswith('<'):
        try:
            return float(s[1:]) / 2.0
        except ValueError:
            return np.nan
    if s.startswith('>'):
        try:
            return float(s[1:])
        except ValueError:
            return np.nan
    if ('+' in s) or ('＋' in s):
        nums = []
        for p in re.split(r'[+＋]', s):
            try:
                nums.append(float(p.strip()))
            except ValueError:
                pass
        if not nums:
            return np.nan
        return nums[0] if first_code else float(np.sum(nums))
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_balloon_text(v):
    """解析形如 '凰泰球囊(20atm) 4mm与30大气压巴德球囊(30atm) 6mm' 的球囊直径文本"""
    if pd.isna(v):
        return dict(dia_max=np.nan, dia_mean=np.nan, n_balloon=0)
    if isinstance(v, (int, float)):
        return dict(dia_max=float(v), dia_mean=float(v), n_balloon=1)
    s = str(v)
    nums = [float(x) for x in re.findall(r'(\d+\.?\d*)\s*mm', s)]
    if not nums:
        try:
            fv = float(s.strip())
            return dict(dia_max=fv, dia_mean=fv, n_balloon=1)
        except ValueError:
            return dict(dia_max=np.nan, dia_mean=np.nan, n_balloon=0)
    return dict(dia_max=max(nums), dia_mean=float(np.mean(nums)), n_balloon=len(nums))


def _parse_pressure_text(v):
    """解析形如 '30大气压巴德球囊(30atm) 28bpm与...26bpm' 的压力文本"""
    if pd.isna(v):
        return dict(pmax=np.nan, pmean=np.nan, n_press=0)
    if isinstance(v, (int, float)):
        return dict(pmax=float(v), pmean=float(v), n_press=1)
    s = str(v)
    nums = [float(x) for x in re.findall(r'(\d+\.?\d*)\s*bpm', s)]
    if not nums:
        try:
            fv = float(s.strip())
            return dict(pmax=fv, pmean=fv, n_press=1)
        except ValueError:
            return dict(pmax=np.nan, pmean=np.nan, n_press=0)
    return dict(pmax=max(nums), pmean=float(np.mean(nums)), n_press=len(nums))


def load_raw():
    return pd.read_excel(C.DATA_FILE, sheet_name=C.DATA_SHEET)


def build_clean_dataset():
    raw = load_raw()
    df, audit = clean_frame(raw, with_targets=True, init_audit=True)
    audit['n_raw'] = len(raw)
    audit['n_cols_raw'] = raw.shape[1]
    return df, audit


def clean_frame(raw, with_targets=True, init_audit=False):
    """对任意原始格式 DataFrame 执行步骤1-7清洗(供研究与网页应用共用)"""
    df = raw.copy()
    audit = {'n_clean': len(df), 'actions': [],
             'clinical_outliers': pd.DataFrame(),
             'missing_required': []} if init_audit else None

    # 1) 删除ID/常量/日期列(保留 C_GROUP_KEY 分组键)
    drop_cols = [c for c in ['序号', '我院干预次序', C.C_SURDATE]
                 if c in df.columns and c != C.C_GROUP_KEY]
    df = df.drop(columns=drop_cols)
    if audit is not None:
        audit['actions'].append(f'删除标识/常量/日期列: {drop_cols}'
                                + (f', 保留分组键 {C.C_GROUP_KEY}' if C.C_GROUP_KEY in raw.columns else ''))
        # 记录分组键分布供 GroupKFold 使用
        if C.C_GROUP_KEY in raw.columns:
            audit['group_key'] = C.C_GROUP_KEY
            audit['group_counts'] = raw[C.C_GROUP_KEY].value_counts().to_dict()
            audit['n_groups'] = raw[C.C_GROUP_KEY].nunique()

    # 2) 球囊直径/压力文本解析为数值特征
    if C.C_BALDIA in df.columns:
        dia_ext = df[C.C_BALDIA].apply(_parse_balloon_text).apply(pd.Series)
        dia_ext.columns = ['最大球囊直径(mm)', '平均球囊直径(mm)', '球囊使用数量(个)']
        df = df.drop(columns=[C.C_BALDIA])
        df = pd.concat([df, dia_ext], axis=1)
    else:
        df['最大球囊直径(mm)'] = np.nan
        df['平均球囊直径(mm)'] = np.nan
        df['球囊使用数量(个)'] = 0
    if C.C_PRESS in df.columns:
        pre_ext = df[C.C_PRESS].apply(_parse_pressure_text).apply(pd.Series)
        pre_ext.columns = ['最大扩张压力(atm)', '平均扩张压力(atm)', '扩张次数']
        df = df.drop(columns=[C.C_PRESS])
        df = pd.concat([df, pre_ext], axis=1)
    else:
        df['最大扩张压力(atm)'] = np.nan
        df['平均扩张压力(atm)'] = np.nan
        df['扩张次数'] = 0

    # 3) 各列强制数值化(跳过 C_GROUP_KEY 字符串分组键)
    for col in df.columns:
        if col == C.C_GROUP_KEY:
            continue   # 分组键(如患者序号A001..A136)保留为字符串, 不 coerce
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        if not pd.api.types.is_numeric_dtype(df[col]):
            fc = col in C.FIRST_CODE_COLS
            df[col] = df[col].apply(lambda v: _coerce_scalar(v, first_code=fc))
    # 仅对非分组键列做数值 coerce
    num_cols = [c for c in df.columns if c != C.C_GROUP_KEY]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    # 4) 删除5处分段病变长度(与总病变长度存在完全线性依赖 W = X+Y+Z+AA+AB), 保留总长度
    drop2 = [c for c in C.C_SITE_LENS if c in df.columns]
    df = df.drop(columns=drop2)
    if audit is not None:
        audit['actions'].append(f'删除与"总病变长度"完全线性依赖的分段列: {drop2}')

    # 5) 临床合理区间: 越界值记为缺失(录入错误)
    outlier_log = []
    for col, (lo, hi) in C.CLINICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        n_bad = int(bad.sum())
        if n_bad:
            bad_vals = sorted(df.loc[bad, col].unique().tolist())[:6]
            outlier_log.append(dict(特征=C.short(col), 列名=col, 越界数=n_bad,
                                    合理区间=f'[{lo},{hi}]',
                                    越界值样例=str(bad_vals)))
            df.loc[bad, col] = np.nan
    if audit is not None:
        audit['clinical_outliers'] = pd.DataFrame(outlier_log)
        audit['actions'].append(f'按临床合理区间判为录入错误并置缺失: {len(outlier_log)} 列')

    # 6) 衍生特征
    df['脉压差(mmHg)'] = df[C.C_SBP] - df[C.C_DBP] if C.C_SBP in df.columns else np.nan
    if C.C_THROMB in df.columns:
        df['合并血栓(0/1)'] = (df[C.C_THROMB].fillna(0) > 0).astype(float)
    else:
        df['合并血栓(0/1)'] = 0.0
    df['是否高压球囊(>=30atm)'] = (df['最大扩张压力(atm)'] >= 30).astype(float)
    if C.C_BALTYPE in df.columns:
        df['是否药物涂层球囊'] = (df[C.C_BALTYPE] == 8).astype(float)
    else:
        df['是否药物涂层球囊'] = np.nan
    if C.C_SNAD in df.columns and C.C_PNAD in df.columns:
        df['动脉狭窄改善(mm)'] = df[C.C_SNAD] - df[C.C_PNAD]
    else:
        df['动脉狭窄改善(mm)'] = np.nan
    if C.C_SNVD in df.columns and C.C_PNVD in df.columns:
        df['静脉狭窄改善(mm)'] = df[C.C_SNVD] - df[C.C_PNVD]
    else:
        df['静脉狭窄改善(mm)'] = np.nan
    # 6a) V-AB 血管病变衍生(基于分段长度在 clean 前即已被 drop, 所以只依赖 C_NLESION / C_LLEN)
    if C.C_NLESION in df.columns and C.C_LLEN in df.columns:
        df['平均病变长度(mm)'] = np.where(df[C.C_NLESION] > 0,
                                          df[C.C_LLEN] / df[C.C_NLESION], np.nan)
    # 临床长病变: 总长度 >= 50mm 且病变数 >= 2 (至少两处, 或单处长病变)
    if C.C_LLEN in df.columns:
        df['长病变标志(>=50mm)'] = ((df[C.C_LLEN] >= 50)).astype(float)
    # 术前术后 RI/PSV/EDV 差 (供 PDP 扩展分析, 若同时作为目标 T_DRI 的补)
    if C.C_PRI in df.columns and C.C_POSTRI in df.columns:
        df['术前术后阻力指数差(RI差)'] = df[C.C_POSTRI] - df[C.C_PRI]
    if C.C_PPSV in df.columns and C.C_POSTPSV in df.columns:
        df['术前术后PSV差'] = df[C.C_POSTPSV] - df[C.C_PPSV]
    if C.C_PEDV in df.columns and C.C_POSTEDV in df.columns:
        df['术前术后EDV差'] = df[C.C_POSTEDV] - df[C.C_PEDV]

    # 7) 目标衍生变量(上传数据若无目标列则跳过)
    if with_targets and C.T_POST in df.columns and C.T_PRE in df.columns:
        df[C.T_DFLOW] = df[C.T_POST] - df[C.T_PRE]
    if with_targets and C.C_PRI in df.columns and C.C_POSTRI in df.columns:
        df[C.T_DRI] = df[C.C_PRI] - df[C.C_POSTRI]

    if audit is not None:
        audit['n_clean'] = len(df)
        audit['n_cols_clean'] = df.shape[1]
        return df, audit
    return df


# ---------------- 特征集划分 ----------------
def feature_sets(df):
    """返回建模目标的 (特征列表) 字典; 目前仅 ΔQ 一个目标;
    自动排除 C_GROUP_KEY 分组键 和 含术后信息的衍生变量"""
    allc = df.columns.tolist()
    # 术后原始超声列
    post_ultra = [c for c in [
        C.C_USTIME_POST, C.C_POSTAD, C.C_POSTVD, C.C_SNAD, C.C_SNVD,
        C.C_POSTRI, C.C_POSTPSV, C.C_POSTEDV] if c in allc]
    # "术前术后××差"——含术后信息, 绝不进 ΔQ 特征集
    post_derived = [c for c in ['术前术后阻力指数差(RI差)', '术前术后PSV差', '术前术后EDV差']
                    if c in allc]
    # 所有"含术后信息"的列, 合并排除
    post_info = list(set(post_ultra + post_derived))
    targets = [C.T_PRE, C.T_POST, C.T_DFLOW, C.T_DRI]
    # 永不进入特征集的列: 分组键 + 目标 + 含术后信息的列
    _never = set(targets) | set(post_info) | {C.C_GROUP_KEY}

    # ΔQ 模型: 仅基线 + 干预因素(不含术后超声和术后衍生差, 保证可术前预测改善程度)
    delta_exclude = set(targets) | set(post_info) | {
        '动脉狭窄改善(mm)', '静脉狭窄改善(mm)', C.C_GROUP_KEY}
    f_delta = [c for c in allc if c not in delta_exclude]
    return {
        C.T_DFLOW: f_delta,
    }


# ---------------- 建模矩阵准备(供 quality / modeling 共用) ----------------
def prepare_matrix(df, features, target, missing_threshold=0.5,
                   vif_threshold=10.0, drop_for_vif=None):
    """
    1. 剔除目标缺失行
    2. 剔除缺失率过高特征
    3. 中位数填补
    4. 迭代剔除 VIF>阈值 的特征(按 VIF 从大到小, 保留 drop_for_vif 黑名单优先剔除)
    返回 X(原始量纲 DataFrame), y(Series), 被剔除列列表, vif Series
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression

    cols = [c for c in features if c in df.columns]
    sub = df[cols + [target]].copy()
    sub = sub.dropna(subset=[target])
    X = sub[cols]
    y = sub[target].astype(float)

    keep = X.columns[X.isnull().mean() <= missing_threshold].tolist()
    X = X[keep]
    imp = SimpleImputer(strategy='median')
    X = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)

    # 删除常数列
    const_cols = [c for c in X.columns if X[c].std(ddof=0) < 1e-10]
    if const_cols:
        X = X.drop(columns=const_cols)

    def _vif_table(Xd):
        vif = {}
        colsx = Xd.columns.tolist()
        for col in colsx:
            others = [c for c in colsx if c != col]
            r2 = LinearRegression().fit(Xd[others].values, Xd[col].values).score(
                Xd[others].values, Xd[col].values)
            vif[col] = 1e6 if r2 >= 0.99999 else 1.0 / (1.0 - r2)
        return pd.Series(vif).sort_values(ascending=False)

    dropped = list(drop_for_vif or [])
    drop_have = [c for c in dropped if c in X.columns]
    X = X.drop(columns=drop_have)

    while True:
        vif = _vif_table(X)
        if vif.iloc[0] <= vif_threshold:
            break
        col = vif.index[0]
        X = X.drop(columns=[col])
        dropped.append(col)

    return X, y, dropped, _vif_table(X)


if __name__ == '__main__':
    d, a = build_clean_dataset()
    print('原始:', a['n_raw'], 'x', a['n_cols_raw'])
    print('清洗后:', a['n_clean'], 'x', a['n_cols_clean'])
    print('目标缺失:', d[C.T_DFLOW].isna().sum())
    fs = feature_sets(d)
    for k, v in fs.items():
        print(C.short(k), '->', len(v), '特征')
    print(a['clinical_outliers'].to_string())
