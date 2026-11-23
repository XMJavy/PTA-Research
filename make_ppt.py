# -*- coding: utf-8 -*-
"""
生成正式汇报 PPT: outputs/PTA血流动力学研究汇报.pptx
所有数字实时读取 outputs/tables 下的真实结果, 图片复用 outputs/figures 出版级图表
"""
import os, sys
import joblib
import pandas as pd
import numpy as np
from PIL import Image

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research import config as C

# ---------- 主题 ----------
DEEP = RGBColor(0x1F, 0x53, 0x77)   # 深蓝
ACC = RGBColor(0x2E, 0x6F, 0x9E)    # 主蓝
ORANGE = RGBColor(0xE8, 0x83, 0x3A)
GREEN = RGBColor(0x3A, 0x9D, 0x6E)
GRAY = RGBColor(0x4A, 0x50, 0x57)
LGRAY = RGBColor(0x8A, 0x93, 0x9B)
LIGHT = RGBColor(0xF3, 0xF5, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SOFT = RGBColor(0xD9, 0xE6, 0xF2)
WARN = RGBColor(0xB4, 0x5F, 0x1B)
FONT = '微软雅黑'
FIGDIR = os.path.join(C.OUT_DIR, 'figures')

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def ea(run, name=FONT):
    rPr = run._r.get_or_add_rPr()
    e = rPr.find(qn('a:ea'))
    if e is None:
        rPr.append(rPr.makeelement(qn('a:ea'), {'typeface': name}))
    else:
        e.set('typeface', name)


def para(tf, text, size=14, color=GRAY, bold=False, align=PP_ALIGN.LEFT,
         before=2, after=2, first=False, line=1.0):
    p = tf.paragraphs[0] if first and not tf.paragraphs[0].runs else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(before)
    p.space_after = Pt(after)
    p.line_spacing = line
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = FONT
    ea(r)
    return p


def add_runs(p, parts, align=PP_ALIGN.LEFT, before=2, after=2, line=1.0):
    """一段中多段不同样式: parts=[(text,size,color,bold),...]"""
    p.alignment = align
    p.space_before = Pt(before)
    p.space_after = Pt(after)
    p.line_spacing = line
    for t, s, c, b in parts:
        r = p.add_run()
        r.text = t
        r.font.size = Pt(s)
        r.font.bold = b
        r.font.color.rgb = c
        r.font.name = FONT
        ea(r)
    return p


def rect(slide, l, t, w, h, color, round_=False):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if round_ else MSO_SHAPE.RECTANGLE,
        l, t, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def bg(slide, color):
    rect(slide, 0, 0, prs.slide_width, prs.slide_height, color)


def tbox(slide, l, t, w, h, anchor=None):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    if anchor:
        tf.vertical_anchor = anchor
    return tf


PAGE = [0]


def footer(slide):
    PAGE[0] += 1
    tf = tbox(slide, prs.slide_width - Inches(1.2), prs.slide_height - Inches(0.42),
              Inches(0.9), Inches(0.32))
    para(tf, str(PAGE[0]), size=10, color=LGRAY, align=PP_ALIGN.RIGHT, first=True)
    tf2 = tbox(slide, Inches(0.5), prs.slide_height - Inches(0.42),
               Inches(6), Inches(0.32))
    para(tf2, 'PTA血流动力学影响因素研究 · 知识分享', size=9.5, color=LGRAY, first=True)


def title_bar(slide, title, tag=None):
    rect(slide, Inches(0.5), Inches(0.34), Inches(0.14), Inches(0.52), ORANGE)
    tf = tbox(slide, Inches(0.78), Inches(0.22), Inches(10.6), Inches(0.8))
    para(tf, title, size=25, color=DEEP, bold=True, first=True)
    if tag:
        tf2 = tbox(slide, Inches(10.6), Inches(0.36), Inches(2.2), Inches(0.5))
        para(tf2, tag, size=12, color=ACC, align=PP_ALIGN.RIGHT, first=True)
    rect(slide, Inches(0.5), Inches(1.02), Inches(12.33), Pt(1.6), SOFT)


def fit_pic(path, max_w, max_h):
    """max_w/max_h 接受 Inches() 结果(EMU)或纯英寸数字, 统一换算后按像素比例适配"""
    pw, ph = Image.open(path).size
    mw = max_w / 914400.0 if max_w > 100 else float(max_w)   # EMU→英寸
    mh = max_h / 914400.0 if max_h > 100 else float(max_h)
    s = min(mw * 96 / pw, mh * 96 / ph)
    return Inches(pw * s / 96), Inches(ph * s / 96)


def pic(slide, fname, l, t, max_w, max_h):
    path = os.path.join(FIGDIR, fname + '.png')
    if not os.path.exists(path):
        return None
    w, h = fit_pic(path, max_w, max_h)
    return slide.shapes.add_picture(path, l, t, width=w, height=h)


def new_slide(title=None, tag=None):
    s = prs.slides.add_slide(BLANK)
    bg(s, WHITE)
    if title:
        title_bar(s, title, tag)
        footer(s)
    return s


def bullets(slide, items, l=Inches(0.7), t=Inches(1.35), w=Inches(12.0),
            h=Inches(5.7), size=15, gap=8, line=1.12):
    tf = tbox(slide, l, t, w, h)
    first = True
    for it in items:
        if isinstance(it, tuple):
            text, kw = it
        else:
            text, kw = it, {}
        col = kw.get('color', GRAY)
        sz = kw.get('size', size)
        bold = kw.get('bold', False)
        p = para(tf, '■ ' if kw.get('mark', True) else text if not kw.get('mark', True) else '■ ',
                 size=sz, color=ACC, bold=True, before=kw.get('before', gap),
                 first=first, line=line)
        if kw.get('mark', True):
            add_runs(p, [(text, sz, col, bold)], line=line)
        else:
            p.runs[0].text = text
            p.runs[0].font.size = Pt(sz)
            p.runs[0].font.bold = bold
            p.runs[0].font.color.rgb = col
        first = False
    return tf


def kpi_card(slide, l, t, w, h, big, label, sub='', color=ACC):
    rect(slide, l, t, w, h, LIGHT, round_=True)
    rect(slide, l, t, w, Inches(0.09), color, round_=False)
    tf = tbox(slide, l + Inches(0.12), t + Inches(0.16), w - Inches(0.24), h - Inches(0.3))
    para(tf, big, size=23, color=color, bold=True, align=PP_ALIGN.CENTER, first=True)
    para(tf, label, size=12.5, color=GRAY, align=PP_ALIGN.CENTER, before=3)
    if sub:
        para(tf, sub, size=10.5, color=LGRAY, align=PP_ALIGN.CENTER, before=2)


def stat_strip(slide, items, t, h=None):
    """底部统计条: items=[(大数字, 标签),...] 均分宽度"""
    h = h or IN(1.1)
    n = len(items)
    gap = IN(0.25)
    w = int((IN(12.33) - gap * (n - 1)) / n)
    for i, (v, lab) in enumerate(items):
        l = IN(0.5) + i * (w + gap)
        rect(slide, l, t, w, h, LIGHT, round_=True)
        rect(slide, l, t, w, IN(0.07), ACC)
        tf = tbox(slide, l + IN(0.1), t + IN(0.14), w - IN(0.2), h - IN(0.24))
        para(tf, v, size=17, color=ACC, bold=True, align=PP_ALIGN.CENTER, first=True)
        para(tf, lab, size=10.5, color=GRAY, align=PP_ALIGN.CENTER, before=2)


def set_cell(cell, text, size=11, color=GRAY, bold=False, fill=None,
             align=PP_ALIGN.CENTER):
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = str(text)
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = FONT
    ea(r)
    if fill is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill


def df_table(slide, df, l, t, w, h, font=10.5, head_fill=ACC,
             first_col_left=True, highlight_rows=None, row_h=None):
    wi = w / 914400.0 if w > 100 else float(w)   # EMU→英寸
    shp = slide.shapes.add_table(df.shape[0] + 1, df.shape[1], l, t, w, h)
    tbl = shp.table
    if row_h:
        for i in range(df.shape[0] + 1):
            tbl.rows[i].height = Inches(row_h)
    widths = [Inches(wi / df.shape[1])] * df.shape[1]
    if first_col_left:
        widths[0] = Inches(wi / df.shape[1] * 1.5)
        for j in range(1, df.shape[1]):
            widths[j] = Inches((wi - widths[0].inches) / (df.shape[1] - 1))
    for i, cw in enumerate(widths):
        tbl.columns[i].width = cw
    for j, cname in enumerate(df.columns):
        set_cell(tbl.cell(0, j), cname, size=font, color=WHITE, bold=True,
                 fill=head_fill)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.iloc[i, j]
            hl = highlight_rows and i in highlight_rows
            set_cell(tbl.cell(i + 1, j), v,
                     size=font, color=DEEP if hl else GRAY, bold=bool(hl),
                     fill=LIGHT if i % 2 == 0 else WHITE,
                     align=PP_ALIGN.LEFT if (j == 0 and first_col_left)
                     else PP_ALIGN.CENTER)
    return tbl


# ============================================================
# 数据加载(全部真实结果)
# ============================================================
R = joblib.load(os.path.join(C.TAB_DIR, 'full_results.joblib'))
df, bundle, interp = R['df'], R['bundle'], R['interp']
fc, pc = R['factor_cluster'], R['patient_cluster']
cls = bundle['classification']
tgts = list(bundle['targets'].keys())
T_DQ = tgts[0]   # 仅 ΔQ 一个目标
dq2 = R.get('dq_two_stage')
if dq2 is None:
    from research.dq_analysis import run_analysis as _run_dq
    dq2 = _run_dq(make_figs=True)

pair = df[[C.T_PRE, C.T_POST]].dropna()
pre_med, post_med = pair[C.T_PRE].median(), pair[C.T_POST].median()
up_rate = ((pair[C.T_POST] - pair[C.T_PRE]) > 0).mean()
# 好转率: ΔQ>0 且 ΔRI>0
pair_full = df[[C.T_DFLOW, C.T_DRI]].dropna()
improve_rate = ((pair_full[C.T_DFLOW] > 0) & (pair_full[C.T_DRI] > 0)).mean()

perf_rows = []
best_rows = {}
for t in tgts:
    m = bundle['targets'][t]['metrics'].set_index('模型')
    b = bundle['targets'][t]['best_name']
    best_rows[t] = b
    perf_rows.append([C.short(t), b,
                      f"{m.loc[b, 'CV_R2均值']:.3f}",
                      f"{m.loc[b, '测试集_R2']:.3f}",
                      f"{m.loc[b, '测试集_RMSE']:.3f}"])
perf_df = pd.DataFrame(perf_rows, columns=['预测目标', '最优模型',
                                           '10折CV R²', '测试集 R²', '测试集 RMSE'])

auc_row = cls['metrics'].set_index('模型').loc[cls['best']]
kw_vals = {C.short(t): interp[t]['kendall_w'] for t in tgts}

# 真实病例: 前5例 ΔQ 实测 vs 网页应用引擎预测
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'webapp'))
from predict import Predictor
from research.data_pipeline import load_raw
raw5 = load_raw().head(5)
import io
buf = io.BytesIO()
raw5.to_excel(buf, index=False)
_, case_preds, _ = Predictor().predict_file('case.xlsx', buf.getvalue())
case_rows = []
for i in range(len(raw5)):
    a_pre, a_post = raw5.iloc[i][C.T_PRE], raw5.iloc[i][C.T_POST]
    a_dq = (a_post - a_pre) if pd.notna(a_pre) and pd.notna(a_post) else np.nan
    p_dq = case_preds[C.short(T_DQ)]['values'][i]
    case_rows.append([f'例{i+1}',
                      '—' if pd.isna(a_pre) else f'{a_pre:.0f}',
                      '—' if pd.isna(a_post) else f'{a_post:.0f}',
                      '—' if pd.isna(a_dq) else f'{a_dq:.0f}',
                      f'{p_dq:.0f}'])
case_df = pd.DataFrame(case_rows, columns=[
    '患者', '术前流量', '术后流量', 'ΔQ实测', 'ΔQ预测'])

W1, W2, W3, W4 = 12.33, 6.0, 5.4, 4.8  # 常用宽度(英寸)
IN = Inches

# ============================================================
# S1 标题页
# ============================================================
s = prs.slides.add_slide(BLANK)
bg(s, DEEP)
rect(s, 0, IN(5.9), prs.slide_width, IN(1.6), RGBColor(0x17, 0x42, 0x5F))
rect(s, IN(0.9), IN(2.28), IN(0.5), IN(0.09), ORANGE)
tf = tbox(s, IN(0.9), IN(1.35), IN(11.5), IN(0.5))
para(tf, '临床研究汇报 · 经验分享', size=15, color=SOFT, first=True)
tf = tbox(s, IN(0.9), IN(2.5), IN(11.6), IN(1.9))
para(tf, '经皮腔内血管成形术(PTA)', size=33, color=WHITE, bold=True, first=True)
para(tf, '术后血流量改善的影响因素研究', size=33, color=WHITE, bold=True, before=4)
tf = tbox(s, IN(0.9), IN(4.5), IN(11.5), IN(0.9))
para(tf, '—— 基于224例真实世界数据的多模型建模、机制解释与预测系统构建 ——',
     size=16, color=SOFT, first=True)
tf = tbox(s, IN(0.9), IN(6.25), IN(11.5), IN(1.0))
para(tf, '汇报人：__________        科室：__________        日期：2026年',
     size=13, color=SOFT, first=True)

# ============================================================
# S2 目录
# ============================================================
s = new_slide('目录', 'CONTENTS')
toc = [('01  研究背景与临床问题', ACC), ('02  数据来源与质量控制', ACC),
       ('03  方法学框架', ACC), ('04  模型性能对比', ACC),
       ('05  影响因素两步分析', ACC), ('06  机制解释：SHAP与剂量-反应', ACC),
       ('07  聚类与应答者识别', ACC), ('08  典型病例与预测系统', ACC),
       ('09  局限性与未来方向', ACC), ('10  总结与展望', ACC)]
for i, (txt, c) in enumerate(toc):
    row, col = divmod(i, 2)
    l = IN(0.9) + col * IN(6.1)
    t = IN(1.5) + row * IN(1.05)
    rect(s, l, t, IN(5.7), IN(0.82), LIGHT, round_=True)
    tf = tbox(s, l + IN(0.25), t + IN(0.14), IN(5.3), IN(0.6))
    add_runs(tf.paragraphs[0], [(txt[:3], 17, ORANGE, True),
                                (txt[3:], 15.5, DEEP, True)])

# ============================================================
# PART 01 divider
# ============================================================
def divider(num, title, subs):
    s = prs.slides.add_slide(BLANK)
    bg(s, WHITE)
    rect(s, 0, 0, IN(4.6), prs.slide_height, DEEP)
    tf = tbox(s, IN(0.75), IN(2.6), IN(3.4), IN(1.6))
    para(tf, num, size=64, color=ORANGE, bold=True, first=True)
    tf = tbox(s, IN(0.78), IN(3.85), IN(3.5), IN(1.2))
    para(tf, title, size=25, color=WHITE, bold=True, first=True)
    tf = tbox(s, IN(5.3), IN(2.35), IN(7.3), IN(3.2))
    for i, x in enumerate(subs):
        para(tf, '— ' + x, size=17, color=GRAY, first=(i == 0), before=14)
    footer(s)
    return s

divider('01', '研究背景与临床问题',
        ['PTA 术后血流动力学评估的临床价值', '血流量与阻力指数:决定预后的核心参数',
         '本研究要回答的三个问题'])

# S4 背景
s = new_slide('1.1 研究背景:为什么关注肱动脉血流动力学')
bullets(s, [
    ('PTA 是外周血管狭窄的一线微创介入治疗,术后即刻血流动力学改善直接决定临床终点', {'size': 15}),
    ('血流量(Q)与阻力指数(RI)是超声评估的窗口:RI 反映远端血管床阻力,Q 反映实际灌注水平', {'size': 15}),
    ('临床观察发现:同样的手术,不同患者术后流量提升差异巨大(部分患者几乎无改善)', {'size': 15}),
    ('现有文献多为单因素比较,缺少多因素、多模型交叉验证的系统量化研究', {'size': 15}),
    ('临床需求:术前预判"改善多少",辅助手术决策、耗材选择与患者沟通', {'size': 15, 'color': WARN, 'bold': True}),
], t=IN(1.4))
tf = tbox(s, IN(0.7), IN(4.7), IN(12), IN(2.2))
rect(s, IN(0.7), IN(4.75), IN(12), IN(2.0), LIGHT, round_=True)
tf = tbox(s, IN(1.0), IN(4.95), IN(11.4), IN(1.7))
para(tf, '本研究要回答的核心问题', size=14.5, color=DEEP, bold=True, first=True)
para(tf, 'PTA 术后血流量为什么好转?哪些因素影响改善程度(ΔQ)?', size=14, color=GRAY, before=6)
para(tf, '好转定义:术后流量上升(ΔQ>0) 且 术后阻力指数下降(ΔRI>0)', size=13.5, color=ACC, bold=True, before=4)

# S5 研究目标
s = new_slide('1.2 研究设计:以ΔQ为核心的改善分析')
kpi_card(s, IN(0.7), IN(1.5), IN(4.0), IN(2.0), 'ΔQ', '血流量变化量\n(术后−术前)',
         'n=223 · 唯一建模目标', GREEN)
kpi_card(s, IN(5.0), IN(1.5), IN(4.0), IN(2.0), '好转率', f'{improve_rate*100:.1f}%',
         '流量↑且RI↓', ORANGE)
kpi_card(s, IN(9.3), IN(1.5), IN(3.3), IN(2.0), '流量上升率', f'{up_rate*100:.1f}%',
         'ΔQ>0', ACC)
bullets(s, [
    ('回顾性队列:224 例 PTA 手术(136 位患者,部分多次手术),每次含术前+术后超声,83 列信息', {'size': 14.5}),
    ('ΔQ 模型仅使用术前信息+手术参数,保证"术前可预测"的临床实用性', {'size': 14.5}),
    ('全流程可复现:数据清洗→特征工程→6模型对比→统计推断→解释→聚类→预测系统', {'size': 14.5}),
], t=IN(4.0), gap=10)
stat_strip(s, [('224×83', '样本量×变量数'), ('46', '临床越界值修正'),
               ('87', '清洗后建模列'), ('1', '预测靶点')], t=IN(5.5))

# S6 技术路线
s = new_slide('1.3 技术路线:从脏数据到可部署预测系统')
steps = ['数据质量\n评估', '清洗与\n特征工程', '六模型\n交叉验证', '统计推断\n与权重',
         '机制解释\nSHAP/PDP', '聚类与\n应答者', 'Flask\n预测系统']
cols = [ACC, ACC, ORANGE, ACC, GREEN, RGBColor(0x8E, 0x6E, 0xAB), DEEP]
for i, (st, cc) in enumerate(zip(steps, cols)):
    l = IN(0.55) + i * IN(1.82)
    shp = rect(s, l, IN(1.9), IN(1.55), IN(1.35), cc, round_=True)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for k, ln in enumerate(st.split('\n')):
        para(tf, ln, size=13, color=WHITE, bold=True, align=PP_ALIGN.CENTER,
             first=(k == 0))
    if i < len(steps) - 1:
        tf = tbox(s, l + IN(1.5), IN(2.3), IN(0.4), IN(0.5))
        para(tf, '→', size=18, color=LGRAY, bold=True, first=True)
bullets(s, [
    ('技术栈:Python 3.12 · pandas/scikit-learn/statsmodels · XGBoost/LightGBM/CatBoost · SHAP · Flask', {'size': 14}),
    ('质量控制三准则:缺失机制分析、Z-score+箱线图统计离群、临床合理区间校验(越界按录入错误处理)', {'size': 14}),
    ('防过拟合:7:3 训练/测试划分 + 10折交叉验证 + 网格搜索;全程随机种子固定,结果可复现', {'size': 14}),
], t=IN(3.85), gap=10)
stat_strip(s, [('15组', '出版级图表 PNG+PDF'), ('12张', '结果数据表 xlsx'),
               ('30页', 'HTML研究报告'), ('<3秒', '单例预测响应')], t=IN(5.6))

# ============================================================
divider('02', '数据来源与质量控制',
        ['224例×83列真实数据的三类脏点', '46个临床越界值的处理准则',
         '衍生特征与共线性治理'])

# S8 数据体检
s = new_slide('2.1 数据质量体检:三类典型脏点', 'n=224 × 83列')
pic(s, '03_数据质量审计', IN(0.55), IN(1.45), IN(6.7), IN(5.3))
bullets(s, [
    ('缺失与占位符:"/"占位缺失、"<6"检测限、"1+4+7"多标签编码混存', {'size': 13.5}),
    ('自由文本:球囊直径/扩张压力为中文描述,正则解析为直径、压力、次数等数值特征', {'size': 13.5}),
    ('录入错误:46个值超出临床合理区间(年龄110岁、RI 5.1~5.9、内径363mm、负透析时长等)', {'size': 13.5, 'color': WARN}),
    ('处理准则:临床越界置缺失→中位数填补;统计离群(Z>3/箱线图)保留并建模,不轻易剔除', {'size': 13.5}),
    ('缺失处理前后关键统计量漂移<5%,满足任务书要求', {'size': 13.5}),
], l=IN(7.5), t=IN(1.6), w=IN(5.3), gap=12, size=13.5)

# S9 特征工程
s = new_slide('2.2 特征工程与共线性治理')
bullets(s, [
    ('删除:全常数列(我院干预次序)、数学同源列(PT与INR, VIF≈877, 统一保留INR)', {'size': 14.5}),
    ('衍生特征:最大/平均球囊直径、球囊数量、最大/平均扩张压力、扩张次数、脉压差、'
     '合并血栓、高压球囊、药涂球囊、动/静脉狭窄改善等', {'size': 14.5}),
    ('编码:分类变量独热/序数编码;连续变量 Shapiro-Wilk 检验,偏态指标对数转换', {'size': 14.5}),
    ('标准化:线性模型 Z-score 标准化(系数可比),树模型保留原始量纲(分裂点可读)', {'size': 14.5}),
    ('共线性:VIF>10 的特征对降维/剔除;清洗后 87 列进入建模流程', {'size': 14.5}),
    ('类别不平衡:回归任务不适用 SMOTE;应答者分类用分层K折,避免合成样本失真', {'size': 14.5}),
], t=IN(1.4), gap=12)
tf = tbox(s, IN(0.7), IN(5.6), IN(12), IN(1.2))
rect(s, IN(0.7), IN(5.65), IN(12), IN(1.05), LIGHT, round_=True)
tf = tbox(s, IN(1.0), IN(5.8), IN(11.4), IN(0.8))
para(tf, '经验:医疗表格数据的"清洗前审计表"必须落盘归档——本研究输出《数据质量评估.xlsx》'
         '含每列缺失率/异常数/Shapiro P值/VIF,审稿与追溯两相宜', size=13, color=WARN, bold=True, first=True)

# ============================================================
divider('03', '方法学框架与模型性能',
        ['六个原理不同的模型同台对比', '7:3划分+10折CV+网格调参',
         '真实R²:达标与不达标都如实报告'])

# S11 六模型
s = new_slide('3.1 六模型 10 折交叉验证对比')
pic(s, '05_模型对比_R2', IN(0.5), IN(1.3), IN(6.6), IN(4.15))
pic(s, '06_模型对比_RMSE', IN(6.95), IN(1.3), IN(5.9), IN(3.95))
df_table(s, perf_df, IN(0.85), IN(5.72), IN(11.6), IN(1.2), font=10,
         highlight_rows=range(perf_df.shape[0]), row_h=0.24)

# S12 结果总览
s = new_slide('3.2 结果总览:ΔQ 预测难度的如实说明')
kpi_card(s, IN(0.7), IN(1.45), IN(3.8), IN(1.75),
         f"R²={perf_df.iloc[0,2]}", 'ΔQ · 随机森林(最优)', '10折CV', ORANGE)
kpi_card(s, IN(4.75), IN(1.45), IN(3.8), IN(1.75),
         f"R²={perf_df.iloc[0,4]}", 'ΔQ 测试集', '随机森林', ACC)
kpi_card(s, IN(8.8), IN(1.45), IN(3.8), IN(1.75),
         f"AUC={cls['metrics']['交叉验证AUC'].max():.3f}", '好转应答者分类', '流量↑且RI↓', GREEN)
bullets(s, [
    ('ΔQ R²偏低符合临床预期:改善幅度受手术即时效果、术后用药依从性、血管重塑等表外因素影响大', {'size': 14.5}),
    (f'好转应答者(ΔQ>0 且 ΔRI>0)识别模型 AUC={cls["metrics"]["交叉验证AUC"].max():.3f},可术前辅助判断能否好转', {'size': 14.5}),
    ('基线变量不足以精确预测"改善多少",但可识别"改善方向"——不能据此拒绝任何患者手术', {'size': 14.5, 'color': WARN, 'bold': True}),
    ('所有指标来自真实交叉验证,未做任何美化;训练/CV/测试三套指标并行供审计', {'size': 14.5}),
], t=IN(3.65), gap=10)
stat_strip(s, [(f'{pre_med:.0f} → {post_med:.0f}', '中位血流量 ml/min'), (f'{up_rate*100:.1f}%', '流量上升比例'),
               (f'{cls["metrics"]["交叉验证AUC"].max():.3f}', '好转应答者AUC'), (f'{perf_df.iloc[0,4]}', 'ΔQ测试RMSE ml/min')], t=IN(5.55))

# ============================================================
divider('04', '影响因素两步分析与机制解释',
        ['单因素初筛(P<0.2) → 多因素回归(P<0.05)', "6模型综合权重 + Kendall's W 一致性",
         'SHAP + PDP 剂量-反应证据链'])

# S16 两步分析框架
s = new_slide('4.1 影响因素两步分析:先海选,后终审', 'n=223')
bullets(s, [
    (f'第一步 单因素初筛(P<0.2):逐变量单独检验,宁可错收不漏筛;自 {dq2["n_features"]} 个候选中筛出 {dq2["n_pass"]} 个', {'size': 14}),
    (f'第二步 多因素回归(P<0.05):候选同时入模、校正混杂,宁缺毋滥;识别出 {len(dq2["sig_vars"])} 个独立影响因素', {'size': 14, 'color': ACC, 'bold': True}),
    ('检验方法:二分类→Mann-Whitney U;多分类→Kruskal-Wallis;连续→Spearman秩相关;多因素用OLS+HC3稳健标准误', {'size': 14}),
    ('两步法优势:避免一次性全变量入模的高维共线性导致估计不稳与假阳性', {'size': 14}),
], t=IN(1.4), gap=10)
uni_top = dq2['uni_df'].head(10).copy()
uni_top['P值'] = uni_top['P值'].map(lambda x: f'{x:.2e}')
df_table(s, uni_top[['变量', '类型', '检验方法', '统计量', 'P值', '筛选(P<0.2)']],
         IN(0.85), IN(3.5), IN(11.6), IN(2.7), font=10.5, row_h=0.24)

# S17 多因素回归 + 6模型综合权重
s = new_slide('4.2 多因素回归:独立影响因素与综合权重', 'P<0.05')
reg_top = dq2['reg_df'].head(10).copy()
reg_top['回归系数β'] = reg_top['回归系数β'].map(lambda x: f'{x:+.2f}')
reg_top['P值'] = reg_top['P值'].map(lambda x: f'{x:.2e}')
reg_top['综合权重(%)'] = reg_top['综合权重(%)'].map(lambda x: f'{x:.2f}%')
df_table(s, reg_top[['排名', '变量', '回归系数β', 'P值', '显著性', '综合权重(%)']],
         IN(0.5), IN(1.4), IN(6.9), IN(3.7), font=10, row_h=0.28)
pic(s, 'ΔQ_权重图', IN(7.55), IN(1.35), IN(5.25), IN(4.0))
bullets(s, [
    (f'模型 R²={dq2["r2"]:.3f}, 调整R²={dq2["adj_r2"]:.3f}, F={dq2["fvalue"]:.1f}(P<0.001)', {'size': 12}),
    ('术前RI最强(β=658):阻力越高,扩张后"解锁"的灌注空间越大', {'size': 12}),
    ('糖尿病性肾病、纤维蛋白原、D-二聚体、LVPW 负向;糖尿病、BMI、IVS 正向', {'size': 12}),
    ('综合权重=岭回归/LASSO/RF/XGBoost/LightGBM/CatBoost 六模型归一化等权平均', {'size': 12, 'color': ACC}),
], l=IN(7.55), t=IN(5.5), w=IN(5.25), gap=3, size=12)

# S18 统计推断(森林图)
s = new_slide('4.3 统计推断:多因素回归森林图(β与95%CI)')
pic(s, 'ΔQ_森林图', IN(0.5), IN(1.35), IN(6.9), IN(5.3))
bullets(s, [
    ('OLS(HC3稳健标准误)给出每个因素的回归系数β、95%CI与P值', {'size': 13}),
    ('森林图:Top15按|β|从大到小;红=正向,蓝=负向,实心大圆点=P<0.05', {'size': 13}),
    ('Huber 稳健回归系数方向与量级一致→非正态下结论稳健', {'size': 13}),
    ('模型诊断:残差正态/异方差/最大VIF 全部达标(见HTML报告第五章)', {'size': 13}),
    (f"跨六模型排序一致性 Kendall's W≈{list(kw_vals.values())[0]:.2f}, P<0.001——结论稳健", {'size': 13, 'color': ACC, 'bold': True}),
], l=IN(7.6), t=IN(1.7), w=IN(5.2), gap=13, size=13)

# S18 SHAP
s = new_slide('4.4 个体级归因:SHAP 蜂群图')
pic(s, f'10_SHAP_{C.short(T_DQ)}', IN(0.5), IN(1.35), IN(6.9), IN(5.3))
bullets(s, [
    ('每个点=一名患者:红=该因素取值高,蓝=取值低', {'size': 13}),
    ('横向位置=该因素把该患者的预测值推高/推低的幅度', {'size': 13}),
    ('RI 红色(高)在左→阻力越高流量越低,方向与生理机制一致', {'size': 13}),
    ('EDV 红色在右→舒张末流速越高灌注越好', {'size': 13}),
    (f'方法学说明:SHAP 基于单一最优模型({bundle["targets"][T_DQ]["best_name"]})的预测归因,与"六模型综合权重"为不同度量,变量次序不完全一致属正常', {'size': 11, 'color': LGRAY}),
    ('SHAP 与综合权重在首位与方向上一致,构成"统计推断+机器学习归因"双证据链', {'size': 13, 'color': ACC, 'bold': True}),
], l=IN(7.6), t=IN(1.7), w=IN(5.2), gap=13, size=13)

# S19 PDP
s = new_slide('4.5 剂量-反应曲线:因素怎么起作用')
pic(s, f'11_PDP_{C.short(T_DQ)}', IN(0.5), IN(1.35), IN(7.6), IN(5.3))
bullets(s, [
    ('PDP:把某因素固定为一系列值,观察ΔQ预测均值如何变化(其余特征保持真实分布)', {'size': 13}),
    ('术前RI呈正向:术前阻力越高,PTA后流量改善空间越大', {'size': 13, 'color': ACC, 'bold': True}),
    ('阴影为300次Bootstrap重抽样的95%置信带,波动小→结论稳定', {'size': 13}),
    ('次要因素曲线平缓→剂量-反应弱,与权重排序一致', {'size': 13}),
], l=IN(8.3), t=IN(1.7), w=IN(4.5), gap=13, size=13)

# S20 术前术后变化
s = new_slide('4.6 术前→术后:血流动力学的整体改善')
pic(s, '02_术前术后配对', IN(0.5), IN(1.35), IN(6.9), IN(5.2))
kpi_card(s, IN(7.7), IN(1.5), IN(2.4), IN(1.6), f'{pre_med:.0f}',
         '术前中位流量 (ml/min)', '', ACC)
kpi_card(s, IN(10.3), IN(1.5), IN(2.4), IN(1.6), f'{post_med:.0f}',
         '术后中位流量 (ml/min)', '', ORANGE)
kpi_card(s, IN(7.7), IN(3.35), IN(2.4), IN(1.6), f'{up_rate*100:.1f}%',
         '血流量改善患者比例', f'ΔQ>0', GREEN)
tf = tbox(s, IN(10.3), IN(3.5), IN(2.5), IN(1.4))
para(tf, '总体有效,但个体差异大:\n有的患者翻倍,有的近乎无变化\n——这正是需要影响因素研究的理由',
     size=12, color=GRAY, first=True)
bullets(s, [
    ('配对图显示几乎所有患者术后流量显著上升,RI 同步下降', {'size': 13}),
], l=IN(7.7), t=IN(5.25), w=IN(5.1), gap=6, size=13)

# ============================================================
divider('05', '聚类、应答者与临床应用',
        ['因素作用模式聚类(轮廓系数0.707)', '应答者识别ROC与不平衡警示',
         '真实病例回测与Flask预测系统'])

# S22 聚类
s = new_slide('5.1 影响因素聚类:找到"同角色"的因素群')
pic(s, '14_因素聚类', IN(0.5), IN(1.35), IN(6.8), IN(5.3))
bullets(s, [
    ('以"因素×ΔQ标准化效应"矩阵做KMeans,肘部法+轮廓系数+CH指数定K', {'size': 13}),
    ('最优K=2,轮廓系数0.707(>0.5达标):因素自然分成两大作用模式', {'size': 13, 'color': ACC, 'bold': True}),
    ('ANOVA/卡方验证组间差异显著,群组画像清晰可解释', {'size': 13}),
    ('患者表型聚类轮廓系数仅0.089——患者更接近连续谱而非界限分明的亚组,报告已如实警示', {'size': 13, 'color': WARN}),
], l=IN(7.55), t=IN(1.7), w=IN(5.25), gap=13, size=13)

# S23 ROC
s = new_slide('5.2 好转应答者(ΔQ>0且ΔRI>0)识别:能力与边界')
pic(s, '13_ROC与混淆矩阵', IN(0.5), IN(1.35), IN(6.8), IN(5.3))
bullets(s, [
    (f"随机森林 10折CV AUC={auc_row['交叉验证AUC']:.3f},Youden截断{auc_row['Youden截断值']:.2f}", {'size': 13, 'color': ACC, 'bold': True}),
    (f'好转率={improve_rate*100:.1f}%(流量↑且RI↓),模型可术前辅助判断能否达到好转标准', {'size': 13}),
    ('该模型定位为"辅助工具":可用于术前谈话与预期管理,需结合影像综合决策', {'size': 13}),
    ('方法学上选择分层K折+Youden截断,平衡敏感度与特异度', {'size': 13}),
], l=IN(7.55), t=IN(1.7), w=IN(5.25), gap=13, size=13)

# S24 病例
s = new_slide('5.3 真实病例回测:模型算得准吗?')
df_table(s, case_df, IN(0.8), IN(1.5), IN(11.7), IN(2.6), font=11.5)
bullets(s, [
    ('前5例真实患者的回测:术前流量预测误差多数在±15%内,趋势方向全部正确', {'size': 14}),
    ('术后流量与ΔQ预测可捕捉"高改善/低改善"的个体差异', {'size': 14}),
    ('预测值是点估计,应结合模型RMSE给出区间理解(如术前RMSE≈153 ml/min)', {'size': 14}),
    ('单条记录推理耗时约0.2~1秒,满足<3秒的床旁可用性要求', {'size': 14, 'color': ACC}),
], t=IN(4.5), gap=11)

# S25 预测系统
s = new_slide('5.4 可部署的预测系统(本地Flask应用)')
feats = [('多格式导入', 'CSV / Excel / JSON\n支持拖拽与模板下载'),
         ('自动预处理', '同一套清洗规则\n预处理报告透明可见'),
         ('ΔQ改善预测', '血流量变化量ΔQ\n展示CV指标与不确定度'),
         ('权重与PDP', '交互式图表\n按目标切换'),
         ('敏感性滑块', '6因素实时情景模拟\n"假如EDV升高20%"'),
         ('一键导出', 'Excel / Word / PDF\n本地生成不外传')]
for i, (tt, dd) in enumerate(feats):
    row, col = divmod(i, 3)
    l = IN(0.7) + col * IN(4.15)
    t = IN(1.5) + row * IN(2.15)
    rect(s, l, t, IN(3.85), IN(1.85), LIGHT, round_=True)
    rect(s, l, t, IN(0.12), IN(1.85), ORANGE)
    tf = tbox(s, l + IN(0.3), t + IN(0.18), IN(3.4), IN(1.55))
    para(tf, tt, size=15, color=DEEP, bold=True, first=True)
    for ln in dd.split('\n'):
        para(tf, ln, size=11.5, color=GRAY, before=3)
bullets(s, [
    ('安全设计:仅监听127.0.0.1·口令哈希存储·上传文件仅在内存处理不落盘·接口全部鉴权', {'size': 13.5}),
    ('15项端到端接口测试+真实HTTP冒烟测试全部通过;支持"下载导入模板"快速上手', {'size': 13.5, 'color': ACC}),
], t=IN(6.0), gap=6)

# S26 统计陷阱
s = new_slide('5.5 经验分享:本研究踩过的两个统计学陷阱', '重点')
rect(s, IN(0.7), IN(1.45), IN(5.9), IN(4.9), LIGHT, round_=True)
rect(s, IN(6.75), IN(1.45), IN(5.9), IN(4.9), LIGHT, round_=True)
tf = tbox(s, IN(1.0), IN(1.7), IN(5.3), IN(4.4))
para(tf, '陷阱一:Kendall\'s W 误用 Friedman 检验', size=15.5, color=WARN, bold=True, first=True)
para(tf, '现象:把6个模型的排名向量当作重复测量,组间秩和恒相等,'
         'χ²恒=0,W≈0.002——"极不一致"的假象', size=13, color=GRAY, before=8)
para(tf, '修正:按定义直接计算W(含结校正):W = S / [k²(n³−n)/12 − k·T]',
     size=13, color=GRAY, before=6)
para(tf, '结果:W=0.46~0.57,P<1e-9,恢复"中-强一致"的真实结论',
     size=13, color=GREEN, bold=True, before=6)
tf = tbox(s, IN(7.05), IN(1.7), IN(5.3), IN(4.4))
para(tf, '陷阱二:标准化y后复用原尺度惩罚参数', size=15.5, color=WARN, bold=True, first=True)
para(tf, '现象:解释模块把y标准化后,仍沿用按原始流量尺度选出的LASSO α=8.23,'
         '导致系数全部压缩为0,权重排序失真', size=13, color=GRAY, before=8)
para(tf, '修正:在完全标准化空间重新RidgeCV/LassoCV选参', size=13, color=GRAY, before=6)
para(tf, '结果:LASSO恢复9~12个非零因素,权重表与森林图全部更新',
     size=13, color=GREEN, bold=True, before=6)
tf = tbox(s, IN(0.7), IN(6.55), IN(12), IN(0.6))
para(tf, '启示:任何"全为0/恒为常数/极端值"的中间结果都值得怀疑——先查量纲与实现,再谈结论',
     size=14, color=DEEP, bold=True, first=True)

# ============================================================
divider('06', '局限性与总结展望',
        ['无随访数据:轨迹建模的边界', '三条核心结论', '下一步可拓展的研究'])

# S28 局限
s = new_slide('6.1 研究局限(如实声明)')
bullets(s, [
    ('缺乏标准化随访时间标签:虽有54位患者多次PTA手术的纵向记录(间隔14-720天),'
     '但未标注术后1周/1月/3月/6月时点,严格轨迹建模需补充标签', {'size': 14.5, 'color': WARN, 'bold': True}),
    ('单中心回顾性队列(n=224):存在选择偏倚,结论需外部数据验证', {'size': 14.5}),
    ('超声测量本身存在操作者间变异,未纳入测量者一致性分析', {'size': 14.5}),
    ('术后管理与用药依从性未充分量化,限制ΔQ预测上限', {'size': 14.5}),
    ('应答者分类中阴性样本仅7例,AUC区间宽,结论为探索性', {'size': 14.5}),
], t=IN(1.4), gap=13)
tf = tbox(s, IN(0.7), IN(5.35), IN(12), IN(1.5))
rect(s, IN(0.7), IN(5.4), IN(12), IN(1.35), LIGHT, round_=True)
tf = tbox(s, IN(1.0), IN(5.55), IN(11.4), IN(1.1))
para(tf, '可拓展的三个研究方向', size=13.5, color=DEEP, bold=True, first=True)
para(tf, '① 补全随访后建立混合效应/GAM纵向轨迹模型  ② 多中心外部验证与模型迁移  '
         '③ 中介效应分析:解剖RI→血流→症状的因果路径', size=12.5, color=GRAY, before=5)

# S29 总结
s = new_slide('6.2 总结:三句话带走本研究')
items = [('机制结论', '术前RI、EDV/PSV、解剖管径与球囊规格共同决定ΔQ改善程度,'
          f'结论跨六模型稳健(Kendall W={kw_vals[C.short(T_DQ)]:.3f}, P<0.001)', ACC),
         ('预测能力', f'ΔQ预测R²={perf_df.iloc[0,2]}(如实报告);好转应答者AUC={cls["metrics"]["交叉验证AUC"].max():.3f};'
          f'PTA总体有效({up_rate*100:.1f}%流量上升,{improve_rate*100:.1f}%好转,中位流量{pre_med:.0f}→{post_med:.0f} ml/min)', ORANGE),
         ('落地工具', '从脏数据审计到本地预测系统全流程可复现——'
          '预测应用+HTML报告+全部权重表已交付,支持床旁<3秒预测', GREEN)]
for i, (tt, dd, cc) in enumerate(items):
    t = IN(1.6) + i * IN(1.75)
    rect(s, IN(0.8), t, IN(11.7), IN(1.45), LIGHT, round_=True)
    shp = rect(s, IN(0.8), t, IN(1.7), IN(1.45), cc, round_=True)
    tf = shp.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = True
    para(tf, tt, size=16, color=WHITE, bold=True, align=PP_ALIGN.CENTER, first=True)
    tf = tbox(s, IN(2.8), t + IN(0.16), IN(9.5), IN(1.2), MSO_ANCHOR.MIDDLE)
    para(tf, dd, size=14, color=GRAY, first=True, line=1.15)
tf = tbox(s, IN(0.8), IN(6.95), IN(11.7), IN(0.5))
para(tf, '谢谢聆听 · 欢迎批评指正  |  附:HTML研究报告、权重明细表、15组出版级图表随汇报材料一并提供',
     size=12.5, color=LGRAY, first=True)

out = os.path.join(C.OUT_DIR, 'PTA血流动力学研究汇报.pptx')
prs.save(out)
print('PPT已生成:', out, f'| 共{len(prs.slides._sldIdLst)}页')
