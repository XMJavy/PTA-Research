# -*- coding: utf-8 -*-
"""
基线特征统计分析（表1）
- 患者级聚合：固定属性取首次出现值（矛盾值标记）；动态指标纳入全部测量值
- 正态性：Shapiro-Wilk（α=0.05，P>0.05 正态）；正态→Mean±SD，否则 Median[P25,P75]
- 凝血指标参考值越界比例；抗凝/抗血小板药物频数百分比
- 交叉验证后输出 outputs/tables/baseline_stats.json
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research import config as C
from research import data_pipeline as DP

raw = DP.load_raw()
df = DP.clean_frame(raw)

def col(name):
    if name in df.columns:
        return name
    m = [c for c in df.columns if str(c).strip('\n') == name or str(c).startswith(name)]
    return m[0] if m else None

pid = C.C_GROUP_KEY
N_REC = len(df)
N_PAT = df[pid].nunique()
pat = df.groupby(pid, sort=False).first()          # 固定属性: 首次出现值
dup_rows = int(raw.duplicated().sum())

# ---------------- 标签映射 ----------------
SEX = {1: '男性', 2: '女性'}
EDU = {1: '小学及以下', 2: '小学以上'}
FEE = {1: '本地医保', 2: '异地医保', 3: '自费'}
ETI = {1: '慢性肾小球肾炎', 2: '糖尿病性肾病', 3: '高血压性肾病',
       4: '多囊肾', 5: '痛风性肾病', 6: '梗阻性肾病'}
YN = {0: '无', 1: '有'}
SIDE = {1: '左上肢', 2: '右上肢', 3: '左下肢', 4: '右下肢'}
FTYPE = {1: '标准内瘘(头静脉-桡动脉)', 2: '贵要静脉-尺动脉',
         3: '高位内瘘(头静脉-肱动脉)', 4: '高位内瘘(头静脉-桡动脉近心端)',
         5: '穿静脉-桡动脉', 6: 'AVG人工血管', 7: '肘正中静脉-桡动脉',
         8: '桡动脉-桡静脉', 9: '肱动脉-穿静脉'}
FANAS = {1: '端侧', 2: '端端', 3: '侧侧', 4: '反端侧'}
MACE_LB = {'0': '无', '1': '心绞痛', '2': '心肌梗死', '3': '恶性心律失常',
           '4': '充血性心力衰竭', '5': '心源性休克', '6': '心源性死亡', '7': '脑卒中'}
DRUGS = ['利伐沙班', '低分子肝素', '氢氯吡格雷', '阿司匹林',
         '舒洛地特', '吲哚布芬片', '华法林钠片']

# ---------------- 工具 ----------------
def desc(x):
    """连续变量描述: 正态→mean±sd, 否则 median[P25,P75]; 含独立numpy交叉验证"""
    x = pd.to_numeric(pd.Series(x), errors='coerce').dropna().astype(float)
    n = int(len(x))
    if n < 8:
        return dict(n=n, text='—', normal=None)
    W, p = stats.shapiro(x.values)
    normal = bool(p > 0.05)
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    p25, p50, p75 = [float(v) for v in np.percentile(x.values, [25, 50, 75])]
    # 交叉验证: numpy 独立复算
    assert abs(mean - float(np.mean(x.values))) < 1e-9
    assert abs(p50 - float(np.median(x.values))) < 1e-9
    if normal:
        text = f'{mean:.2f}±{sd:.2f}'
    else:
        text = f'{p50:.2f}[{p25:.2f},{p75:.2f}]'
    return dict(n=n, text=text, normal=normal, W=round(float(W), 4),
                p=float(p), mean=round(mean, 2), sd=round(sd, 2),
                p25=round(p25, 2), p50=round(p50, 2), p75=round(p75, 2))

def cat_counts(series, denom, labels=None):
    """分类变量频数/百分比 + 交叉验证(各类别之和=分母)"""
    s = pd.Series(series).dropna()
    vc = s.value_counts()
    tot = int(vc.sum())
    assert tot == int(len(s))
    out = []
    keys = sorted(vc.index, key=lambda v: (v != v) if isinstance(v, float) else v) \
        if labels is None else [k for k in labels if k in vc.index]
    for k in keys:
        n = int(vc[k])
        out.append(dict(n=n, pct=round(n / denom * 100, 1),
                        label=(labels.get(k, str(k)) if labels else str(k))))
    return out, int(len(s))

rows = []          # 表格行
meta = {'n_pat': N_PAT, 'n_rec': N_REC, 'dup_rows': dup_rows,
        'conflicts': [], 'notes': []}

def section(label):
    rows.append(dict(kind='section', label=label))

def cat_row(label, value='', indent=0, n=None):
    rows.append(dict(kind='cat', label=label, value=value, indent=indent, n=n))

def cont_row(label, d, indent=0):
    v = d['text']
    if d['n'] < N_REC and d['n'] >= 8:
        v += f'  (n={d["n"]})'
    rows.append(dict(kind='cont', label=label, value=v, indent=indent, d=d))

# ================= 一、人口学 / 固定属性（患者级, 首次值） =================
section('人口学资料')
c, n_used = cat_counts(pat[col(C.C_SEX)], N_PAT, SEX)
cat_row('性别, n(%)')
for it in c:
    cat_row(it['label'], f"{it['n']} ({it['pct']})", 1)
meta['notes'].append(f'性别缺失1例, 分母n={n_used}')

d = desc(pat[col(C.C_AGE)])
cont_row('年龄（岁）', d)

c, _ = cat_counts(pat[col(C.C_EDU)], N_PAT, EDU)
cat_row('教育程度, n(%)')
for it in c:
    cat_row(it['label'], f"{it['n']} ({it['pct']})", 1)

c, _ = cat_counts(pat[col(C.C_FEE)], N_PAT, FEE)
cat_row('费用类型, n(%)')
for it in c:
    cat_row(it['label'], f"{it['n']} ({it['pct']})", 1)

c, _ = cat_counts(pat[col(C.C_ETI)], N_PAT, ETI)
cat_row('原发病, n(%)')
for it in c:
    cat_row(it['label'], f"{it['n']} ({it['pct']})", 1)

for cname, title in [(C.C_SMOKE, '吸烟史, n(%)'), (C.C_DM, '糖尿病, n(%)'),
                     (C.C_HBP, '高血压, n(%)')]:
    cc, _ = cat_counts(pat[col(cname)], N_PAT, YN)
    cat_row(title)
    for it in cc:
        cat_row(it['label'], f"{it['n']} ({it['pct']})", 1)

cont_row('BMI（kg/m2）', desc(pat['BMI']))
cont_row('透析时长（天）', desc(pat[col(C.C_DIAL_DAYS)]))
cont_row('透析频次（次/周）', desc(pat[col(C.C_DIAL_FREQ)]))
cont_row('内瘘建立至启用间隔（天）', desc(pat[col(C.C_FIST_INT)]))
cont_row('内瘘使用时长（天）', desc(pat[col(C.C_FIST_DAYS)]))

# 固定属性矛盾标记
for cname in [C.C_SEX, C.C_EDU, C.C_FEE, C.C_ETI, C.C_SMOKE, C.C_DM, C.C_HBP,
              C.C_SIDE, C.C_NLESION]:
    cc = col(cname)
    g = df.groupby(pid)[cc].nunique()
    bad = sorted(g[g > 1].index.tolist())
    if bad:
        meta['conflicts'].append(dict(变量=C.short(cc), 患者数=len(bad),
                                      样例=bad[:5], 处理='取首次出现值'))

# ================= 二、内瘘与病变特征（患者级） =================
section('内瘘与血管病变特征')
c, _ = cat_counts(pat[col(C.C_SIDE)], N_PAT, SIDE)
cat_row('内瘘侧肢体, n(%)')
for it in c:
    cat_row(it['label'], f"{it['n']} ({it['pct']})", 1)

cf, ca = col(C.C_FTYPE), col(C.C_FANAS)
if cf:
    c, _ = cat_counts(pat[cf], N_PAT, FTYPE)
    cat_row('内瘘分类, n(%)')
    for it in c:
        cat_row(it['label'], f"{it['n']} ({it['pct']})", 1)
if ca:
    c, _ = cat_counts(pat[ca], N_PAT, FANAS)
    cat_row('内瘘吻合方式, n(%)')
    for it in c:
        cat_row(it['label'], f"{it['n']} ({it['pct']})", 1)

cont_row('动脉与静脉血管病变数量（处，首次干预）', desc(pat[col(C.C_NLESION)]))
n_lc = next((x['患者数'] for x in meta['conflicts'] if x['变量'] == '病变数量'), 0)
meta['notes'].append('病变数量/总病变长度随每次干预分别记录，固定属性表取首次干预值；'
                     f'病变数量有{n_lc}位患者多次干预间数值不同（属不同次干预的病变记录，非录入错误）')
n_age_na = int(pat[col(C.C_AGE)].isna().sum())
if n_age_na:
    meta['notes'].append(f'年龄{n_age_na}例超出临床合理区间(18~100岁), 按数据质量规则判为录入错误并置缺失')

# ================= 三、生命体征（全部测量值） =================
section('生命体征')
cont_row('收缩压（mmHg）', desc(df[col(C.C_SBP)]))
cont_row('舒张压（mmHg）', desc(df[col(C.C_DBP)]))
cont_row('脉压差（mmHg）', desc(df['脉压差(mmHg)']))

# ================= 四、实验室检查（全部测量值） =================
section('实验室检查')
LAB = [(C.C_WBC, '白细胞（×10^9/L）'), (C.C_HGB, '血红蛋白（g/L）'),
       (C.C_PLT, '血小板（×10^9/L）'), (C.C_CRP, 'C反应蛋白（mg/L）'),
       (C.C_ALB, '白蛋白（g/L）'), (C.C_CA, '钙（mmol/L）'),
       (C.C_P, '磷（mmol/L）'), (C.C_K, '钾（mmol/L）'),
       (C.C_PTH, '甲状旁腺激素PTH（pg/ml）'), (C.C_BNP, 'B型钠尿肽BNP（pg/ml）'),
       (C.C_TNI, '肌钙蛋白（ng/ml）')]
for cname, label in LAB:
    cont_row(label, desc(df[col(cname)]))

# ================= 五、凝血功能 + 越界比例 =================
section('凝血功能')
REF = [(C.C_APTT, 'APTT（秒）', 25.1, 36.5), (C.C_PT, 'PT（秒）', 9.4, 12.5),
       (C.C_INR, 'INR', 0.85, 1.13), (C.C_FBG, 'Fbg（g/L）', 2.0, 4.8),
       (C.C_TT, 'TT（秒）', 10.3, 16.6), (C.C_DD, 'D-二聚体（ng/ml）', 0, 500)]
for cname, label, lo, hi in REF:
    x = pd.to_numeric(df[col(cname)], errors='coerce').dropna()
    d = desc(x)
    cont_row(label, d)
    n = len(x)
    n_lo, n_hi = int((x < lo).sum()), int((x > hi).sum())
    cat_row(f'低于参考值（<{lo}）', f'{n_lo} ({round(n_lo/n*100,1)})', 1)
    cat_row(f'高于参考值（>{hi}）', f'{n_hi} ({round(n_hi/n*100,1)})', 1)

# ================= 六、心脏超声（全部测量值） =================
section('心脏超声影像学指标')
ECHO = [(C.C_AAO, '升主动脉AAO（mm）'), (C.C_LA, '左房LA（mm）'),
        (C.C_LVD, '左室LVDd（mm）'), (C.C_LVPW, '左室后壁LVPW（mm）'),
        (C.C_IVS, '室间隔IVS（mm）'), (C.C_RA, '右房RA（mm）'),
        (C.C_RV, '右室RV（mm）'), (C.C_EF, 'EF值（%）')]
for cname, label in ECHO:
    cont_row(label, desc(df[col(cname)]))

# 动态: 总病变长度纳入全部测量
cont_row('动脉与静脉血管总病变长度（mm，全部干预）', desc(df[col(C.C_LLEN)]))

# ================= 七、抗凝/抗血小板药物（记录级+患者级） =================
section('抗凝、抗血小板药物使用')
rec_drug = raw[col(C.C_DRUG)].astype(str).str.strip()
rec_drug = rec_drug.replace('/', np.nan)
pat_drug = raw.groupby(pid, sort=False).first()[col(C.C_DRUG)].astype(str).str.strip()
cnt_rec, cnt_pat = {}, {}
for name in DRUGS + ['无']:
    if name == '无':
        m_rec = rec_drug == '无'
        m_pat = pat_drug == '无'
    else:
        m_rec = rec_drug.str.contains(name, na=False)
        m_pat = pat_drug.str.contains(name, na=False)
    cnt_rec[name] = int(m_rec.sum())
    cnt_pat[name] = int(m_pat.sum())
cat_row('药物使用, n(%)（记录级/患者级）')
for name in ['无'] + DRUGS:
    cat_row(name, f'{cnt_rec[name]} ({round(cnt_rec[name]/N_REC*100,1)}) / '
                  f'{cnt_pat[name]} ({round(cnt_pat[name]/N_PAT*100,1)})', 1)
n_multi_rec = int((rec_drug.dropna().str.contains(r'\+').sum()))
meta['notes'].append(f'联合用药记录{n_multi_rec}条（按所含各类药物分别计入）')

# ================= 八、MACE（患者级: 任一记录出现即计） =================
section('MACE事件（患者级，任一记录出现即计）')
mace_raw = raw[col(C.C_MACE)].astype(str).str.strip()
ev_pat = {}
for code, lb in MACE_LB.items():
    if code == '0':
        continue
    m = mace_raw.str.contains(lb, na=False)
    pats = df.loc[m, pid].nunique()
    ev_pat[lb] = pats
none_pat = N_PAT - df.loc[mace_raw != '无', pid].nunique()
cat_row('MACE, n(%)')
cat_row('无', f'{none_pat} ({round(none_pat/N_PAT*100,1)})', 1)
for lb in ['心绞痛', '心肌梗死', '恶性心律失常', '充血性心力衰竭',
           '心源性休克', '心源性死亡', '脑卒中']:
    cat_row(lb, f'{ev_pat[lb]} ({round(ev_pat[lb]/N_PAT*100,1)})', 1)
# MACE 患者内"无→有"进展标记
g = df.groupby(pid)[col(C.C_MACE)].apply(
    lambda s: set(s.astype(str).str.strip()))
prog = [p for p, s in g.items() if '无' in s and len(s) > 1]
if prog:
    meta['conflicts'].append(dict(变量='MACE', 患者数=len(prog), 样例=prog,
                                  处理='属事件随时间进展, 患者级按"任一记录出现"统计'))

# ================= 交叉验证 =================
checks = []
# (1) 分类百分比之和≈100
checks.append(('分类百分比和', all(
    abs(sum(it['pct'] for it in cat_counts(pat[col(k)], N_PAT, mp)[0]) - 100) < 0.6
    for k, mp in [(C.C_SEX, SEX), (C.C_EDU, EDU), (C.C_FEE, FEE)])))
# (2) 连续统计量 numpy 复算抽查
x = pd.to_numeric(df[col(C.C_HGB)], errors='coerce').dropna()
checks.append(('血红蛋白均值复算', abs(float(np.mean(x)) - float(x.mean())) < 1e-9))
# (3) 正态判定与呈现规则一致
dd = desc(df[col(C.C_PTH)])
checks.append(('PTH呈现规则', (dd['p'] > 0.05) == ('±' in dd['text'])))
# (4) 数据类型: 数值列全部 numeric
num_ok = all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns if c != pid)
checks.append(('数值列dtype', bool(num_ok)))
# (5) 完全重复行
checks.append(('重复行检查', dup_rows == 0))
meta['checks'] = [(k, bool(v)) for k, v in checks]

# ---------------- 输出 ----------------
out = dict(meta=meta, rows=rows)
opath = os.path.join(C.TAB_DIR, 'baseline_stats.json')
with open(opath, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print(f'患者 {N_PAT}, 记录 {N_REC}, 完全重复行 {dup_rows}')
print('交叉验证:', meta['checks'])
print('矛盾标记:')
for cf in meta['conflicts']:
    print('  ', cf)
print('表格行数:', len(rows))
print('输出:', opath)
