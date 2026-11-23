# -*- coding: utf-8 -*-
"""
全局配置：列名、特征分组、临床合理区间、路径
"""
import os

# ---------------- 路径 ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(ROOT, 'PTA.xlsx')
DATA_SHEET = 'data'
OUT_DIR = os.path.join(ROOT, 'outputs')
TAB_DIR = os.path.join(OUT_DIR, 'tables')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
ARTIFACT_DIR = os.path.join(ROOT, 'webapp', 'artifacts')
for d in (OUT_DIR, TAB_DIR, FIG_DIR, ARTIFACT_DIR):
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42

# ---------------- 目标变量 ----------------
T_PRE = '术前肱动脉血流量(ml/m)'
T_POST = '术后肱动脉血流量(ml/m)'
T_DFLOW = '血流量变化量(术后-术前)'
T_DRI = '阻力指数变化量(术前-术后)'

# 建模目标：仅分析"血流量变化量ΔQ"，代表术后血流动力学好转程度
# 好转定义：术后流量>术前流量(ΔQ>0) 且 术后RI<术前RI(ΔRI>0)
TARGETS = [T_DFLOW]

# ---------------- 关键列全名 ----------------
C_PATIENT_ID = '患者序号（序号相同代表同一人）'   # 分组键, GroupKFold 防泄露
C_GROUP_KEY = C_PATIENT_ID                       # 建模分组锚点
C_SEX = '性别：1=男，2=女'
C_AGE = '年龄（岁）'
C_EDU = '教育程度：1=小学及以下，2=小学以上'
C_FEE = '费用类型：1=本地医保，2=异地医保，3=自费'
C_BMI = 'BMI'
C_ETI = '原发病:1=慢性肾小球肾炎，2=糖尿病性肾病，3=高血压性肾病，4=多囊肾，5=痛风性肾病，6=梗阻性肾病'
C_SMOKE = '吸烟史:0=无 ,1=有'
C_DM = '糖尿病：0=无，1=有'
C_HBP = '高血压：0=无，1=有'
C_DIAL_DAYS = '透析时长（天）'
C_DIAL_FREQ = '透析频次（周）'
C_FIST_INT = '内瘘建立至启用的时间间隔（天）'
C_FIST_DAYS = '内瘘使用时长（天）'
C_MACE = 'MACE：0=无，1=心绞痛，2=心肌梗死，3=恶性心律失常，4=充血性心力衰竭，5=心源性休克，6=心源性死亡，7=脑卒中'
C_SIDE = '内瘘侧肢体：1=左上肢，2=右上肢，3=左下肢，4=右下肢'
C_FTYPE = '内瘘分类：1=标准内瘘头静脉与桡动脉；2=贵要静脉与尺动脉；3=高位内瘘头静脉与肱动脉；4=高位内瘘头静脉与桡动脉近心端；5=穿静脉与桡动脉；6=AVG；7=肘正中静脉与桡动脉；8=桡动脉与桡静脉；9=肱动脉与穿静脉'
C_FANAS = '内瘘吻合方式：1=端侧（静脉端，动脉侧）；2=端端；3=侧侧；4=反端侧（动脉侧端，静脉端）'
C_INDIC = 'PTA临床指征类型：1=低流量（狭窄）；2=血栓；3=成熟不良'
C_STEN = '狭窄类型：1=I型狭窄（动静脉吻合口及距离吻合口2cm内的静脉血管）；2=II型狭窄（动静脉内瘘血管穿刺部位）；3=III型狭窄（动静脉内瘘相关静脉汇入深静脉处）；4=IV型狭窄（动脉流入道）'
# V-AB 新血管病变特征 (Excel 列 V..AB, idx 21..27)
C_NLESION = '动脉与静脉血管病变数量（处）'           # V
C_LLEN = '动脉与静脉血管总病变长度（mm）'             # W
C_SITE_LEN1 = '动脉或静脉血管第一处病变长度（mm）'     # X
C_SITE_LEN2 = '动脉或静脉血管第二处病变长度（mm）'     # Y
C_SITE_LEN3 = '动脉或静脉血管第三处病变长度（mm）'     # Z
C_SITE_LEN4 = '动脉或静脉血管第四处病变长度（mm）'     # AA
C_SITE_LEN5 = '动脉或静脉血管第五处病变长度（mm）'     # AB
C_SITE_LENS = [C_SITE_LEN1, C_SITE_LEN2, C_SITE_LEN3, C_SITE_LEN4, C_SITE_LEN5]
C_THROMB = '血栓长度（mm）'
C_USTIME_PRE = '术前超声与透析的相对时间：1=透析前，2=透析后，3=非透析日'
C_ALEN = '吻合口长度（mm）'
C_AWID = '吻合口宽度（mm）'
C_PREAD = '术前正常参考动脉直径(mm)'
C_PREDV = '术前正常参考静脉直径(mm)'
C_PNAD = '术前最狭窄动脉内径(mm)'
C_PNVD = '术前最狭窄静脉内径(mm)'
C_PBAD = '术前肱动脉内径(mm)'
C_PRI = '术前阻力指数'
C_PPSV = '术前峰值收缩期流速（PSV cm/s）'
C_PEDV = '术前舒张末期流速（EDV cm/s）'
C_INTERV = '本次干预方式：1=PTA，2=溶栓，3=切开取栓，4=人工血管'
C_SURDATE = '本次PTA手术日期'
C_BALTYPE = '球囊种类、长度、额定爆破压：1=27大气压巴德球囊（27atm），2=30大气压巴德球囊(30atm)，3=40大气压巴德球囊(40atm)，4=波科球囊(20atm)，5=凰泰球囊(20atm)，6=鼎科球囊(20atm)，7=鼎科刻痕球囊(20atm)，8=紫杉醇药物涂层球囊(20atm)'
C_BALDIA = '球囊直径（mm）'
C_PRESS = '打开狭窄最低压力(bpm)'
C_USTIME_POST = '术后超声与透析的相对时间：1=透析前，2=透析后，3=非透析日'
C_POSTAD = '术后正常参考动脉直径(mm)'
C_POSTVD = '术后正常参考静脉直径(mm)'
C_SNAD = '狭窄动脉术后内径(mm)'
C_SNVD = '狭窄静脉术后内径(mm)'
C_POSTRI = '术后阻力指数'
C_POSTPSV = '术后峰值收缩期流速（PSV cm/s）'
C_POSTEDV = '术后舒张末期流速（EDV cm/s）'
C_DRUG = '抗凝、抗血小板药物使用：0=无，1=利伐沙班，2=低分子肝素，3=氢氯吡格雷，4=阿司匹林，5=舒洛地特,6=吲哚布芬片；7=华法林钠片'
C_WBC = '白细胞(×10^9/L)'
C_HGB = '血红蛋白(g/L)'
C_PLT = '血小板(×10^9/L)'
C_CRP = 'C反应蛋白(mg/L)'
C_ALB = '白蛋白(g/L)'
C_CA = '钙(mmol/L)'
C_P = '磷(mmol/L)'
C_K = '钾(mmol/L)'
C_PTH = '甲状旁腺激素PTH（pg/ml）'
C_BNP = 'B型钠尿肽BNP(pg/ml)'
C_TNI = '肌钙蛋白(ng/ml)'
C_APTT = '活化部分凝血活酶时间APTT（参考值25.1-36.5秒）'
C_PT = '凝血酶原时间PT（参考值9.4-12.5秒）'
C_INR = '国际标准化比值INR（参考值0.85-1.13）'
C_FBG = '纤维蛋白原Fbg（参考值2.0-4.8g/L）'
C_TT = '凝血酶时间TT（参考值10.3-16.6秒）'
C_DD = 'D二聚体 D-Dimer（参考值0-500ng/ml）'
C_SBP = '收缩压(mmHg)'
C_DBP = '舒张压(mmHg)'
C_AAO = '升主动脉(mm)AAO'
C_LA = '左房(mm)LA'
C_LVD = '左室(mm)LVDd'
C_LVPW = '左室后壁(mm)LVPW'
C_IVS = '室间隔(mm)IVS'
C_RA = '右房(mm)RA'
C_RV = '右室(mm)RV'
C_EF = 'EF值%'

# ---------------- 多标签/组合编码列 (取首个编码=主类别) ----------------
FIRST_CODE_COLS = [C_MACE, C_INDIC, C_STEN, C_INTERV, C_BALTYPE, C_DRUG]

# 分类变量(名义/有序编码, CatBoost按类别处理)
CATEGORICAL_COLS = [
    C_SEX, C_EDU, C_FEE, C_ETI, C_SMOKE, C_DM, C_HBP, C_MACE,
    C_SIDE, C_FTYPE, C_FANAS, C_INDIC, C_STEN, C_USTIME_PRE,
    C_INTERV, C_BALTYPE, C_USTIME_POST, C_DRUG, C_DIAL_FREQ,
]

# ---------------- 临床合理取值区间(超出判为录入错误→缺失) ----------------
CLINICAL_BOUNDS = {
    C_AGE: (18, 100), C_BMI: (10, 45), C_DIAL_DAYS: (0, 12000),
    C_DIAL_FREQ: (1, 4), C_FIST_INT: (0, 8000), C_FIST_DAYS: (0, 12000),
    # V-AB 血管病变特征: 数量 1..5, 长度 0..500mm (最大记录值 W=500)
    C_NLESION: (1, 5), C_LLEN: (1, 500),
    C_SITE_LEN1: (0, 500), C_SITE_LEN2: (0, 200), C_SITE_LEN3: (0, 200),
    C_SITE_LEN4: (0, 50),  C_SITE_LEN5: (0, 20),
    C_THROMB: (0, 1000),
    C_ALEN: (1, 40), C_AWID: (1, 40),
    C_PREAD: (0.5, 12), C_PREDV: (0.5, 15), C_PNAD: (0, 12), C_PNVD: (0, 15),
    C_PBAD: (1, 15), C_PRI: (0.2, 1.2), C_PPSV: (5, 800), C_PEDV: (-5, 600),
    T_PRE: (0, 2500),
    C_POSTAD: (0.5, 12), C_POSTVD: (0.5, 15), C_SNAD: (0, 12), C_SNVD: (0, 15),
    C_POSTRI: (0.2, 1.2), C_POSTPSV: (5, 900), C_POSTEDV: (-5, 700),
    T_POST: (50, 3000),
    C_WBC: (1, 50), C_HGB: (50, 200), C_PLT: (20, 800), C_CRP: (0, 500),
    C_ALB: (15, 60), C_CA: (1, 4), C_P: (0.1, 5), C_K: (2, 8),
    C_PTH: (0, 5000), C_BNP: (0, 30000), C_TNI: (0, 50),
    C_APTT: (10, 120), C_PT: (5, 40), C_INR: (0.5, 5),
    C_FBG: (0.5, 15), C_TT: (5, 60), C_DD: (0, 20000),
    C_SBP: (60, 260), C_DBP: (30, 160),
    C_AAO: (10, 60), C_LA: (10, 80), C_LVD: (20, 90), C_LVPW: (4, 25),
    C_IVS: (4, 25), C_RA: (10, 80), C_RV: (10, 80), C_EF: (10, 90),
}

# ---------------- 特征短名(图表/报告) ----------------
SHORT = {
    T_PRE: '术前肱动脉血流量', T_POST: '术后肱动脉血流量',
    T_DFLOW: '血流量变化量ΔQ', T_DRI: '阻力指数变化量ΔRI',
    C_PRI: '术前阻力指数', C_POSTRI: '术后阻力指数',
    C_PEDV: '术前EDV', C_POSTEDV: '术后EDV',
    C_PPSV: '术前PSV', C_POSTPSV: '术后PSV',
    C_PREAD: '术前参考动脉直径', C_POSTAD: '术后参考动脉直径',
    C_PREDV: '术前参考静脉直径', C_POSTVD: '术后参考静脉直径',
    C_PNAD: '术前最窄动脉内径', C_PNVD: '术前最窄静脉内径',
    C_PBAD: '术前肱动脉内径', C_SNAD: '狭窄动脉术后内径', C_SNVD: '狭窄静脉术后内径',
    C_SEX: '性别', C_AGE: '年龄', C_BMI: 'BMI', C_ETI: '原发病',
    C_DM: '糖尿病', C_HBP: '高血压', C_SMOKE: '吸烟史',
    C_P: '血磷', C_CA: '血钙', C_K: '血钾', C_CRP: 'C反应蛋白',
    C_DD: 'D-二聚体', C_HGB: '血红蛋白', C_ALB: '白蛋白', C_PLT: '血小板',
    C_WBC: '白细胞', C_PTH: 'PTH', C_BNP: 'BNP', C_TNI: '肌钙蛋白',
    C_APTT: 'APTT', C_PT: 'PT', C_INR: 'INR', C_FBG: '纤维蛋白原', C_TT: 'TT',
    C_EF: 'EF值', C_AAO: '升主动脉AAO', C_LA: '左房LA', C_LVD: '左室LVDd',
    C_LVPW: '左室后壁LVPW', C_IVS: '室间隔IVS', C_RA: '右房RA', C_RV: '右室RV',
    C_SBP: '收缩压', C_DBP: '舒张压', '脉压差(mmHg)': '脉压差',
    C_DIAL_DAYS: '透析时长', C_DIAL_FREQ: '透析频次', C_FIST_DAYS: '内瘘使用时长',
    C_FIST_INT: '内瘘建立至启用间隔', C_FTYPE: '内瘘分类', C_FANAS: '内瘘吻合方式',
    C_SIDE: '内瘘侧肢体', C_INDIC: 'PTA指征', C_STEN: '狭窄类型', C_MACE: 'MACE事件',
    C_NLESION: '病变数量', C_LLEN: '总病变长度', C_THROMB: '血栓长度',
    C_SITE_LEN1: '第1处病变长度', C_SITE_LEN2: '第2处病变长度',
    C_SITE_LEN3: '第3处病变长度', C_SITE_LEN4: '第4处病变长度',
    C_SITE_LEN5: '第5处病变长度',
    '病变部位数': '病变部位数', '平均病变长度(mm)': '平均病变长度',
    '长病变标志(>=50mm)': '长病变标志',
    '术前术后阻力指数差(RI差)': 'RI差', '术前术后PSV差': 'PSV差',
    '术前术后EDV差': 'EDV差',
    C_INTERV: '干预方式', C_BALTYPE: '球囊类型', C_ALEN: '吻合口长度', C_AWID: '吻合口宽度',
    C_DRUG: '抗凝/抗血小板药', C_EDU: '教育程度', C_FEE: '费用类型',
    C_USTIME_PRE: '术前超声时机', C_USTIME_POST: '术后超声时机',
    '最大球囊直径(mm)': '最大球囊直径', '球囊使用数量(个)': '球囊数量',
    '最大扩张压力(atm)': '最大扩张压力', '平均扩张压力(atm)': '平均扩张压力',
    '是否高压球囊(>=30atm)': '高压球囊', '是否药物涂层球囊': '药物涂层球囊',
    '合并血栓(0/1)': '合并血栓', '动脉狭窄改善(mm)': '动脉狭窄改善',
    '静脉狭窄改善(mm)': '静脉狭窄改善',
}

def short(name):
    if name in SHORT:
        return SHORT[name]
    return str(name).split('（')[0].split('(')[0].split('：')[0][:14]
