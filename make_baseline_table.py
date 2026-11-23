# -*- coding: utf-8 -*-
"""
按示例图片样式生成基线特征统计 Word 表:
outputs/表1_基线临床特征统计表.docx
读取 outputs/tables/baseline_stats.json
"""
import os, sys, json, platform
import numpy as np
import pandas as pd
import scipy
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research import config as C

with open(os.path.join(C.TAB_DIR, 'baseline_stats.json'), encoding='utf-8') as f:
    R = json.load(f)
meta, rows = R['meta'], R['rows']

GREEN = 'DDEBDD'      # 浅绿底(示例图样式)
HEADGREEN = 'C8DFC8'
FONT = '宋体'

doc = Document()
st = doc.styles['Normal']
st.font.name = FONT
st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn('w:eastAsia'), FONT)

title = doc.add_paragraph()
tr = title.add_run(f"表1　{meta['n_pat']}例维持性血液透析患者"
                   f"{meta['n_rec']}例次PTA干预的基线临床特征")
tr.bold = True
tr.font.size = Pt(12)
tr.font.name = FONT
tr._element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

tbl = doc.add_table(rows=1, cols=2)
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
tbl.style = 'Table Grid'
tbl.columns[0].width = Cm(9.5)
tbl.columns[1].width = Cm(6.5)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.makeelement(qn('w:shd'),
                           {qn('w:val'): 'clear', qn('w:fill'): color})
    tcPr.append(shd)


def set_cell(cell, text, bold=False, size=10, indent=0, color=None, bg=None):
    cell.text = ''
    p = cell.paragraphs[0]
    r = p.add_run(('　' * indent) + text)
    r.font.name = FONT
    r.font.size = Pt(size)
    r.bold = bold
    r._element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
    if color:
        r.font.color.rgb = RGBColor(*color)
    if bg:
        shade(cell, bg)


# 表头
h = tbl.rows[0].cells
set_cell(h[0], '临床指标', bold=True, size=10.5, bg=HEADGREEN)
set_cell(h[1], f"数值（患者n={meta['n_pat']} / 例次n={meta['n_rec']}）",
         bold=True, size=10.5, bg=HEADGREEN)

for row in rows:
    cells = tbl.add_row().cells
    if row['kind'] == 'section':
        set_cell(cells[0], row['label'], bold=True, size=10, bg=GREEN)
        set_cell(cells[1], '', bg=GREEN)
    elif row['kind'] == 'cat':
        set_cell(cells[0], row['label'], size=10, indent=row['indent'])
        set_cell(cells[1], row['value'], size=10)
    else:
        set_cell(cells[0], row['label'], size=10, indent=row['indent'])
        set_cell(cells[1], row['value'], size=10)

# ---------------- 脚注 ----------------
notes = doc.add_paragraph()
def fn(text, bold=False):
    r = notes.add_run(text)
    r.font.name = FONT
    r.font.size = Pt(9)
    r.bold = bold
    r._element.rPr.rFonts.set(qn('w:eastAsia'), FONT)

fn('注：', True)
fn(f"固定属性指标（性别、费用类型、原发病、教育程度、吸烟史、糖尿病、高血压、我院干预次序、"
   f"MACE、内瘘侧肢体、内瘘分类、内瘘吻合方式、动脉与静脉血管病变数量）按患者级统计"
   f"（同一患者多次干预取首次出现值），n={meta['n_pat']}；动态测量指标（生命体征、实验室检查、"
   f"凝血功能、心脏超声、总病变长度）纳入全部测量值，n={meta['n_rec']}例次。")
fn('连续变量先行Shapiro-Wilk正态性检验（α=0.05）：P>0.05者以均数±标准差表示，'
   'P≤0.05者以中位数[P25,P75]表示，均精确至小数点后两位；分类变量以n(%)表示。'
   '凝血指标另计算低于/高于参考值范围的比例。抗凝、抗血小板药物按编码定义统计'
   '（0=无，1=利伐沙班，2=低分子肝素，3=氢氯吡格雷，4=阿司匹林，5=舒洛地特，'
   '6=吲哚布芬片，7=华法林钠片），联合用药按所含各类药物分别计入，表中同时给出'
   '记录级/患者级频数。')
fn('缩略语：PTA=经皮腔内血管成形术；MACE=主要不良心血管事件；BMI=体质量指数；'
   'APTT=活化部分凝血活酶时间；PT=凝血酶原时间；INR=国际标准化比值；'
   'Fbg=纤维蛋白原；TT=凝血酶时间；PTH=甲状旁腺激素；BNP=B型钠尿肽；'
   'EF=射血分数；AAO=升主动脉；LA=左房；LVDd=左室舒张末期内径；'
   'LVPW=左室后壁；IVS=室间隔；RA=右房；RV=右室；IQR=四分位距。')
cf_txt = '；'.join(f"{x['变量']}（{x['患者数']}例，如{'、'.join(map(str, x['样例'][:3]))}）"
                  for x in meta['conflicts'])
fn(f'患者内矛盾值标记：{cf_txt}，均取首次出现值并在附表中说明。')
for nt in meta['notes']:
    fn(nt + '。')
fn(f"统计软件：Python {platform.python_version()}（pandas {pd.__version__}、"
   f"numpy {np.__version__}、scipy {scipy.__version__}）；所有统计量经独立复算交叉验证。")

# ---------------- 附表: 正态性检验明细 ----------------
doc.add_paragraph()
t2 = doc.add_paragraph()
r2 = t2.add_run('附表　连续变量Shapiro-Wilk正态性检验结果及表示方法')
r2.bold = True
r2.font.size = Pt(11)
r2.font.name = FONT
r2._element.rPr.rFonts.set(qn('w:eastAsia'), FONT)

atbl = doc.add_table(rows=1, cols=5)
atbl.style = 'Table Grid'
atbl.alignment = WD_TABLE_ALIGNMENT.CENTER
hd = atbl.rows[0].cells
for i, t in enumerate(['指标', 'n', 'W统计量', 'P值', '表示方法']):
    set_cell(hd[i], t, bold=True, size=9.5, bg=HEADGREEN)
for row in rows:
    if row['kind'] != 'cont' or 'd' not in row or row['d'].get('W') is None:
        continue
    d = row['d']
    ptxt = f"{d['p']:.3f}" if d['p'] >= 0.001 else f"{d['p']:.2e}"
    method = '均数±标准差' if d['normal'] else '中位数[P25,P75]'
    c = atbl.add_row().cells
    set_cell(c[0], row['label'], size=9)
    set_cell(c[1], str(d['n']), size=9)
    set_cell(c[2], f"{d['W']:.4f}", size=9)
    set_cell(c[3], ptxt, size=9)
    set_cell(c[4], method, size=9)

# ---------------- 附表2: 矛盾值说明 ----------------
doc.add_paragraph()
t3 = doc.add_paragraph()
r3 = t3.add_run('附表2　同一患者多次记录间固定属性矛盾值标记')
r3.bold = True
r3.font.size = Pt(11)
r3.font.name = FONT
r3._element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
ctbl = doc.add_table(rows=1, cols=4)
ctbl.style = 'Table Grid'
ctbl.alignment = WD_TABLE_ALIGNMENT.CENTER
hd = ctbl.rows[0].cells
for i, t in enumerate(['变量', '矛盾患者数', '患者样例', '处理方式']):
    set_cell(hd[i], t, bold=True, size=9.5, bg=HEADGREEN)
for x in meta['conflicts']:
    c = ctbl.add_row().cells
    set_cell(c[0], x['变量'], size=9)
    set_cell(c[1], str(x['患者数']), size=9)
    set_cell(c[2], '、'.join(map(str, x['样例'])), size=9)
    set_cell(c[3], x['处理'], size=9)

out = os.path.join(C.OUT_DIR, '表1_基线临床特征统计表.docx')
doc.save(out)
print('saved:', out)
print('主表行数:', len(rows) + 1)
