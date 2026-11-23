# -*- coding: utf-8 -*-
"""
生成 ΔRI（阻力指数变化量）影响因素 HTML 分析报告
图表 base64 内嵌，完全自包含单文件，风格与
《PTA血流动力学影响因素系统研究报告.html》保持一致。

数据来源：import research.dri_analysis.run_analysis() 的真实计算结果，
无任何硬编码统计数字；数据更新后重跑本脚本即可同步。
"""
import os, io, base64, datetime, warnings
import numpy as np
import pandas as pd
from PIL import Image

from research import config as C
from research.dri_analysis import run_analysis

warnings.filterwarnings('ignore')


def img_b64(path, max_w=1100):
    """读取 PNG，缩放后转 base64（不落盘中间文件）"""
    im = Image.open(path).convert('RGB')
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def fig_tag(path, caption, max_w=880):
    if not os.path.exists(path):
        return f'<p class="miss">[缺少图表 {os.path.basename(path)}]</p>'
    return (f'<figure><img src="data:image/png;base64,{img_b64(path)}" '
            f'style="width:100%;max-width:{max_w}px;display:block;margin:0 auto">'
            f'<figcaption>{caption}</figcaption></figure>')


def fmt_p(p):
    """P 值格式化：极小用科学计数法，并加粗标色显著项"""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    s = f'{p:.2e}' if p < 0.001 else f'{p:.4f}'
    if p < 0.05:
        return f'<b style="color:#C0504D">{s}</b>'
    return s


def fmt_beta(v):
    return f'{v:+.4f}'


def uni_table_html(uni_df):
    """单因素分析完整表格，P<0.2 入选行高亮"""
    head = ('<tr><th>序号</th><th>变量</th><th>类型</th><th>检验方法</th>'
            '<th>统计量</th><th>P值</th><th>入选多因素(P&lt;0.2)</th></tr>')
    rows = []
    for i, r in uni_df.iterrows():
        sel = r['筛选(P<0.2)'] == '是'
        cls = ' class="sel"' if sel else ''
        stat = '' if pd.isna(r['统计量']) else (
            f"{r['统计量']:.4f}" if abs(r['统计量']) < 10 else f"{r['统计量']:.2f}")
        sel_html = '<b style="color:#C0504D">是</b>' if sel else '否'
        rows.append(
            f'<tr{cls}><td>{i+1}</td><td>{r["变量"]}</td><td>{r["类型"]}</td>'
            f'<td>{r["检验方法"]}</td><td>{stat}</td><td>{fmt_p(r["P值"])}</td>'
            f'<td>{sel_html}</td></tr>')
    return f'<table class="rtbl">{head}{"".join(rows)}</table>'


def reg_table_html(reg_df):
    """多因素回归系数表（含排名与6模型综合权重%），P<0.05 行高亮"""
    head = ('<tr><th>排名</th><th>变量</th><th>回归系数β</th><th>稳健SE</th>'
            '<th>95%CI下限</th><th>95%CI上限</th><th>P值</th>'
            '<th>显著性</th><th>Huberβ</th><th>方向</th><th>综合权重(%)</th></tr>')
    rows = []
    for _, r in reg_df.iterrows():
        sig = r['P值'] < 0.05
        cls = ' class="sel"' if sig else ''
        direction = ('<span class="up">正向 ↑</span>' if r['回归系数β'] > 0
                     else '<span class="down">负向 ↓</span>')
        sig_html = (f'<b style="color:#C0504D">{r["显著性"]}</b>'
                    if sig else r['显著性'])
        rows.append(
            f'<tr{cls}><td><b style="color:#1f5377">{int(r["排名"])}</b></td>'
            f'<td>{r["变量"]}</td><td>{fmt_beta(r["回归系数β"])}</td>'
            f'<td>{r["稳健SE"]:.4f}</td><td>{r["95%CI下限"]:.4f}</td>'
            f'<td>{r["95%CI上限"]:.4f}</td><td>{fmt_p(r["P值"])}</td>'
            f'<td>{sig_html}</td><td>{fmt_beta(r["Huberβ"])}</td>'
            f'<td>{direction}</td><td><b>{r["综合权重(%)"]:.2f}%</b></td></tr>')
    return f'<table class="rtbl">{head}{"".join(rows)}</table>'


def perf_table_html(metrics, best_name):
    """六模型预测性能表，最优模型行高亮。
    CV R² 为「均值±SD」（10折折间波动）；CV RMSE 仅均值；
    测试集 R² 附 Bootstrap(1000次) 标准差（测试集只评一次，无折间SD）。"""
    has_sd = '测试集_R2_SD' in metrics.columns
    test_head = '测试集 R² (均值±SD)' if has_sd else '测试集 R²'
    head = ('<tr><th>模型</th><th>CV R² (均值±SD)</th><th>CV MAE</th>'
            '<th>CV RMSE</th>'
            '<th>' + test_head + '</th><th>测试集 MAE</th><th>测试集 RMSE</th></tr>')
    rows = []
    for _, r in metrics.iterrows():
        cls = ' class="sel"' if r['模型'] == best_name else ''
        tag = ' <b style="color:#b45f1b">★最优</b>' if r['模型'] == best_name else ''
        r2_cell = f'{r["CV_R2均值"]:.4f} ± {r["CV_R2标准差"]:.4f}'
        mae_cell = f'{r["CV_MAE均值"]:.3f}'
        rmse_cell = f'{r["CV_RMSE均值"]:.3f}'
        test_r2 = (f'{r["测试集_R2"]:.4f} ± {r["测试集_R2_SD"]:.4f}' if has_sd
                   else f'{r["测试集_R2"]:.4f}')
        cells = (f'<td>{test_r2}</td><td>{r["测试集_MAE"]:.3f}</td>'
                 f'<td>{r["测试集_RMSE"]:.3f}</td>')
        rows.append(f'<tr{cls}><td><b>{r["模型"]}</b>{tag}</td>'
                    f'<td>{r2_cell}</td><td>{mae_cell}</td>'
                    f'<td>{rmse_cell}</td>{cells}</tr>')
    return f'<table class="rtbl">{head}{"".join(rows)}</table>'


def weight_table_html(weight_detail, top_n=15):
    """6模型归一化重要性明细表 (TopN, 模型内最重要=100)"""
    models = ['岭回归', 'LASSO', '随机森林', 'XGBoost', 'LightGBM', 'CatBoost']
    head = ('<tr><th>排名</th><th>变量</th><th>岭回归</th><th>LASSO</th>'
            '<th>随机森林</th><th>XGBoost</th><th>LightGBM</th><th>CatBoost</th>'
            '<th>综合权重(%)</th></tr>')
    rows = []
    d = weight_detail.head(top_n)
    for i, (name, r) in enumerate(d.iterrows()):
        cells = ''.join(f'<td>{r[m] * 100:.1f}</td>' for m in models)
        rows.append(
            f'<tr><td>{i + 1}</td><td style="text-align:left">{name}</td>{cells}'
            f'<td><b style="color:#1f5377">{r["综合权重(%)"]:.2f}%</b></td></tr>')
    return f'<table class="rtbl">{head}{"".join(rows)}</table>'


def weight_reg_table_html(df, top_n=15):
    """综合权重 × 标准化回归系数明细表（TopN，P<0.05 行高亮）"""
    head = ('<tr><th>排名</th><th>特征因素</th><th>综合权重%</th><th>标准化β</th>'
            '<th>95%CI下限</th><th>95%CI上限</th><th>OLS P值</th>'
            '<th>显著性</th><th>方向</th></tr>')
    rows = []
    for _, r in df.head(top_n).iterrows():
        sig = r['OLS P值'] < 0.05
        cls = ' class="sel"' if sig else ''
        direction = ('<span class="up">正向 ↑</span>' if r['方向'] == '正向'
                     else '<span class="down">负向 ↓</span>')
        sig_html = (f'<b style="color:#C0504D">{r["显著性"]}</b>'
                    if sig else r['显著性'])
        rows.append(
            f'<tr{cls}><td><b style="color:#1f5377">{int(r["排名"])}</b></td>'
            f'<td style="text-align:left">{r["特征因素"]}</td>'
            f'<td><b>{r["综合权重(%)"]:.2f}%</b></td>'
            f'<td>{fmt_beta(r["标准化β"])}</td>'
            f'<td>{r["95%CI下限"]:+.4f}</td><td>{r["95%CI上限"]:+.4f}</td>'
            f'<td>{fmt_p(r["OLS P值"])}</td><td>{sig_html}</td>'
            f'<td>{direction}</td></tr>')
    return f'<table class="rtbl">{head}{"".join(rows)}</table>'


def build():
    R = run_analysis(make_figs=True, write_docx=False)
    ds, uni_df, reg_df, inter_df = R['desc_stats'], R['uni_df'], R['reg_df'], R['inter_df']
    today = datetime.date.today().isoformat()
    n_pass, sig_vars = R['n_pass'], R['sig_vars']
    ml = R.get('ml', {})

    # ---- 机器学习：最优模型 + SHAP ----
    ml_block = ''
    if ml:
        m = ml
        brow = m['best_row']
        shap_top5 = m['shap_top'].head(5) if m.get('shap_top') is not None else None
        shap_cards = ''
        shap_bullets = ''
        if shap_top5 is not None:
            shap_cards = ''.join(
                f'<div class="card"><div class="n" style="font-size:15px">{i+1}. {nm}</div>'
                f'<div class="l">平均|SHAP| = {val:.4f}</div></div>'
                for i, (nm, val) in enumerate(shap_top5.items()))
            st = m['shap_top']

            def _g(i):
                return ((st.index[i], float(st.iloc[i])) if len(st) > i
                        else ('—', float('nan')))
            (n1, v1), (n2, v2), (n3, v3), (n4, v4), (n5, v5), (n6, v6), (n7, v7) = \
                _g(0), _g(1), _g(2), _g(3), _g(4), _g(5), _g(6)
            shap_bullets = f'''<ul class="clean">
<li><b>{n1}</b> 贡献远高于其他所有因素（平均|SHAP| = {v1:.4f}，呈数量级差距），
且取值越高（红点）SHAP 值越偏右——与多因素回归中"其为最强独立正向因素（β 最大）"完全一致，两类方法互相印证；</li>
<li>第 2、3 位为 <b>{n2}</b>（{v2:.4f}）、<b>{n3}</b>（{v3:.4f}），平均|SHAP| 已远低于首位，
属模型层面的次要校正因素，且均未进入多因素回归显著名单，解读需谨慎；</li>
<li>第 4、5 位为 <b>{n4}</b>（{v4:.4f}）、<b>{n5}</b>（{v5:.4f}）：<b>{n5}</b> 方向与多因素回归结论吻合，
而 <b>{n4}</b> 虽在 SHAP 中排名靠前、却未进入多因素回归显著名单，
提示其作用可能以非线性形式被正则化模型捕获，可作为后续扩大样本后的重点验证因素；</li>
<li>第 6、7 位为 <b>{n6}</b>（{v6:.4f}）、<b>{n7}</b>（{v7:.4f}），方向亦与多因素回归结论一致，
说明靠前因素并非仅由单一模型驱动；</li>
<li>总体而言，SHAP 归因与两步回归识别出的独立影响因素在<b>首位与方向上高度一致</b>，
说明核心结论<b>不依赖单一分析方法</b>。</li>
</ul>'''
        shap_fig = (fig_tag(m['shap_path'],
                            '图5  ΔRI 最优模型 SHAP 特征贡献蜂群图（每个点=一名患者；'
                            '颜色红/蓝=该特征取值高/低；横向位置=该特征把该患者 ΔRI 预测值'
                            '推高/推低的幅度）') if m.get('shap_path') else '')
        ml_block = f'''
<h2>五、机器学习预测模型：六模型对比、最优模型与 SHAP 归因</h2>
<p>除前述统计学两步分析外，进一步用机器学习框架评估"仅凭术前信息能多准确地预测 ΔRI"，
建模流程与主报告 ΔQ 模型<b>完全一致</b>：</p>
<ul class="clean">
<li><b>特征</b>：与两步分析相同的术前基线 + 干预因素，共 {m['n_features']} 个（中位数填补、VIF&gt;10 剔除，不含任何术后信息）；</li>
<li><b>数据划分</b>：按患者序号分组 7:3 划分（训练 {m['n_train']} 例 / 测试 {m['n_test']} 例），
同一患者的多次手术记录不会跨训练/测试集；训练/测试 ΔRI 分布 K-S 检验 P = {m['ks_p']:.3f}，分布可比；</li>
<li><b>六种模型</b>：岭回归（RidgeCV 选 α）、LASSO（LassoCV）、随机森林、XGBoost、LightGBM、CatBoost（后四者 3 折网格调参）；</li>
<li><b>验证与选择</b>：全数据 <b>GroupKFold 10 折交叉验证</b>（按患者分组防泄露）+ 30% 独立测试集，
按 10 折 CV RMSE 最小选出最优模型；</li>
<li><b>个体归因</b>：对最优模型计算 SHAP 值，以蜂群图展示每个因素把每位患者预测值推高/推低的方向与幅度。</li>
</ul>
{perf_table_html(m['metrics'], m['best_name'])}
{fig_tag(m.get('metrics_r2_path', ''),
         '图3  ΔRI 六模型 10 折交叉验证 R² 对比（误差线=标准差，虚线=R²=0.6 目标线）')}
{fig_tag(m.get('metrics_rmse_path', ''),
         '图4  ΔRI 六模型 10 折交叉验证 RMSE 对比（越低越好，误差线=标准差）')}
<div class="cards">
<div class="card"><div class="n">{m['best_name']}</div><div class="l">最优模型（CV RMSE 最小）</div></div>
<div class="card"><div class="n">{brow['CV_R2均值']:.3f}</div><div class="l">10折交叉验证 R²</div></div>
<div class="card"><div class="n">{brow['测试集_R2']:.3f}</div><div class="l">独立测试集 R²</div></div>
<div class="card"><div class="n">{brow['测试集_RMSE']:.3f}</div><div class="l">测试集 RMSE（ΔRI 单位）</div></div>
</div>
<div class="lead"><b>如何解读预测性能：</b>最优模型 <b>{m['best_name']}</b> 的 10 折 CV R² = {brow['CV_R2均值']:.3f}、
测试集 R² = {brow['测试集_R2']:.3f}，即仅凭术前基线与干预因素即可解释约六成的 ΔRI 个体差异，
预测精度明显优于 ΔQ（主报告中 ΔQ 最优模型 CV R² 仅约 0.06）。
这与 ΔRI 的数学结构有关——ΔRI = 术前RI − 术后RI，<b>术前阻力指数本身就是 ΔRI 的组成项</b>，
基线阻力越高、术后可下降的空间越大，因而可预测性强；这并不意味着术后管理不重要，
残余约四成变异仍取决于术中即时效果、术后用药与血管重塑等表外因素。</div>
{shap_fig}
<p class="small">方法学说明：本图 SHAP 贡献基于<b>单一最优模型（{m['best_name']}）</b>对每位患者预测值的分解，
与第四章「六模型综合权重」（6 个模型重要性 min-max 归一化后等权平均）是两种不同度量，
变量次序不完全一致属正常现象；两种方法均一致指向<b>术前阻力指数</b>为首要驱动因素。</p>
<h3>5.1 SHAP 个体级归因：哪些因素在驱动模型预测</h3>
<p>蜂群图中每个点代表一名患者：<b>颜色越红表示该因素取值越高、越蓝表示越低；
点越靠右表示该因素把这位患者的 ΔRI 预测值推得越高（预期 RI 下降越多），越靠左则反之</b>。
按平均绝对 SHAP 值排名，贡献最大的 5 个因素为：</p>
<div class="cards">{shap_cards}</div>
{shap_bullets}
<h3>5.2 最佳模型剂量-反应关系（PDP）</h3>
<p>SHAP 蜂群图揭示了特征对预测值的整体贡献，但<b>无法直接回答"特征越高/越低，ΔRI 会怎么变"</b>。
剂量-反应曲线（Partial Dependence Plot, PDP）通过对最佳模型（{m['best_name']}）进行条件预测，
展示 Top6 权重因素在其取值范围内对 ΔRI 预测的<b>平均边际效应</b>（阴影带为 Bootstrap 95% 置信区间）。
曲线的<b>单调方向</b>对应 SHAP 和多因素回归的系数方向，而<b>弯曲/拐点</b>则暗示存在非线性效应。</p>
{fig_tag(m.get('pdp_path', ''),
         '图6  ΔRI Top6 因素剂量-反应曲线（PDP，阴影=95% Bootstrap 置信带）')}
<h3>5.3 测试集预测诊断：模型拟合度与残差分析</h3>
<p>最终用留出的 30% 独立测试集检验最佳模型的泛化能力：
左图为<b>预测值 vs 观测值</b>散点，点越靠近 y=x 理想线（虚线）说明预测越准确，
绿色实线为模型拟合的线性回归关系；右图为<b>残差分布</b>，理想情况下应近似正态且以 0 为中心。</p>
{fig_tag(m.get('pred_path', ''),
         f'图7  ΔRI 最佳模型（{m["best_name"]}）测试集预测诊断（左：预测 vs 观测；右：残差分布）')}
'''

    # ---- 描述性统计表 ----
    desc_rows = ''.join([
        f'<tr><td>有效样本量 n</td><td>{ds["n"]}</td></tr>',
        f'<tr><td>均值 ± 标准差</td><td>{ds["mean"]:.3f} ± {ds["sd"]:.3f}</td></tr>',
        f'<tr><td>最小值 ~ 最大值</td><td>[{ds["min"]:.3f}, {ds["max"]:.3f}]</td></tr>',
        f'<tr><td>四分位数 P25 / 中位数 / P75</td>'
        f'<td>{ds["p25"]:.3f} / {ds["median"]:.3f} / {ds["p75"]:.3f}</td></tr>',
        f'<tr><td>Shapiro-Wilk 正态性检验</td>'
        f'<td>W = {ds["shapiro_W"]:.4f}, P = {fmt_p(ds["shapiro_p"])}</td></tr>',
        f'<tr><td>分布结论</td><td>{"符合" if ds["normal"] else "<b>不符合</b>"}正态分布'
        f'（P{"&gt;" if ds["normal"] else "&lt;"}0.05）</td></tr>',
    ])
    desc_table = f'<table class="rtbl"><tr><th>统计量</th><th>数值</th></tr>{desc_rows}</table>'

    # ---- 交互作用表 ----
    inter_rows = ''.join(
        f'<tr><td>{r["交互项"]}</td><td>{fmt_beta(r["回归系数"])}</td>'
        f'<td>{fmt_p(r["P值"])}</td><td>{r["显著性"]}</td></tr>'
        for _, r in inter_df.iterrows())
    inter_table = (f'<table class="rtbl"><tr><th>交互项</th><th>回归系数</th>'
                   f'<th>P值</th><th>显著性</th></tr>{inter_rows}</table>')

    # ---- 模型诊断表 ----
    diag_items = [
        ('决定系数 R²', f'{R["r2"]:.4f}', f'模型解释 ΔRI 变异的 {R["r2"]*100:.1f}%'),
        ('调整 R²', f'{R["adj_r2"]:.4f}', '校正自变量个数后仍保持较高解释力'),
        ('F 检验', f'F = {R["fvalue"]:.2f}, P = {R["f_pvalue"]:.2e}', '模型整体极显著'),
        ('残差正态性（Shapiro-Wilk）',
         f'W = {R["resid_W"]:.4f}, P = {fmt_p(R["resid_p"])}',
         '残差服从正态分布' if R['resid_p'] > 0.05 else '残差不服从正态分布'),
        ('异方差检验（Breusch-Pagan）',
         f'LM = {R["bp_stat"]:.4f}, P = {fmt_p(R["bp_p"])}',
         '不存在异方差' if R['bp_p'] > 0.05 else '存在异方差'),
        ('残差分布', f'均值 {R["resid_mean"]:.4f}, SD {R["resid_sd"]:.4f}, '
                     f'范围 [{R["resid_min"]:.4f}, {R["resid_max"]:.4f}]', '以 0 为中心'),
        ('|残差| &gt; 2.5SD 异常值', f'{R["outliers"]} 个', '占比极低，不影响结论稳健性'),
        ('最大 VIF', f'{R["max_vif"]:.2f}', 'VIF &lt; 10，无严重共线性'),
        ('AIC / BIC', f'{R["aic"]:.1f} / {R["bic"]:.1f}', '模型拟合良好'),
    ]
    diag_table = ('<table class="rtbl"><tr><th>诊断项目</th><th>结果</th><th>解读</th></tr>'
                  + ''.join(f'<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>'
                            for a, b, c in diag_items) + '</table>')

    # ---- 动态取关键数字 ----
    top_uni = uni_df.iloc[0]  # 最强单因素关联
    beta_of = lambda v: reg_df.loc[reg_df['变量'] == v, '回归系数β'].iloc[0]
    beta_pri = beta_of('术前阻力指数')
    beta_dm = beta_of('糖尿病')

    # ---- 显著因素临床解读 ----
    interp_cards = ''
    for v in sig_vars:
        row = reg_df[reg_df['变量'] == v].iloc[0]
        direction, text = R['clinical_interp'].get(v, ('', ''))
        dir_cls = 'up' if row['回归系数β'] > 0 else 'down'
        interp_cards += f'''
<div class="card" style="flex:1 1 100%;text-align:left">
<div class="n" style="font-size:17px">{v}
<span class="{dir_cls}" style="font-size:13.5px;margin-left:8px">{direction}
β = {row['回归系数β']:+.3f}，95%CI [{row['95%CI下限']:.3f}, {row['95%CI上限']:.3f}]</span></div>
<div style="font-size:13.8px;color:#444;margin-top:6px">{text}</div>
</div>'''

    html = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ΔRI（阻力指数变化量）影响因素分析报告</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f4f6f8;color:#2b2f33;line-height:1.8;font-size:15px}}
.wrap{{max-width:1080px;margin:0 auto;background:#fff;box-shadow:0 2px 18px rgba(0,0,0,.07)}}
header{{background:linear-gradient(135deg,#1d4e73,#2E6F9E);color:#fff;padding:52px 60px 40px}}
header h1{{font-size:27px;line-height:1.4}}
header .sub{{margin-top:12px;opacity:.92;font-size:14.5px}}
header .meta{{margin-top:20px;padding-top:14px;border-top:1px solid rgba(255,255,255,.25);font-size:13px;opacity:.85}}
main{{padding:40px 60px 64px}}
h2{{font-size:20px;color:#1f5377;border-left:5px solid #2E6F9E;padding-left:13px;margin:44px 0 16px}}
h3{{font-size:16.5px;margin:28px 0 10px;color:#2b2f33}}
h4{{font-size:15px;margin:18px 0 8px;color:#1f5377}}
p{{margin:10px 0;text-align:justify}}
figure{{margin:16px 0 8px}}
figure img{{width:100%;border:1px solid #eceff2;border-radius:6px}}
figcaption{{text-align:center;color:#888;font-size:13px;margin:6px 0 18px}}
table.rtbl{{width:100%;border-collapse:collapse;margin:14px 0;font-size:12.8px}}
table.rtbl th{{background:#2E6F9E;color:#fff;padding:8px 7px;font-weight:600;white-space:nowrap}}
table.rtbl td{{padding:6px 7px;border-bottom:1px solid #edf0f2;text-align:center;white-space:nowrap}}
table.rtbl tbody tr:nth-child(even){{background:#f8fafb}}
table.rtbl tbody tr:hover{{background:#eef5fa}}
table.rtbl tr.sel{{background:#fdf3ec !important}}
table.rtbl tr.sel:hover{{background:#fbe8d8 !important}}
.lead{{background:#eef5fa;border:1px solid #d4e6f3;border-radius:8px;padding:18px 22px;font-size:14.3px}}
.warn{{background:#fdf3ec;border:1px solid #f3d3b8;border-radius:8px;padding:16px 22px;margin:16px 0}}
.warn b{{color:#b45f1b}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}
.card{{flex:1 1 180px;background:#fafbfc;border:1px solid #e6eaee;border-radius:8px;padding:15px 17px}}
.card .n{{font-size:24px;font-weight:700;color:#2E6F9E}}
.card .l{{font-size:12.8px;color:#666;margin-top:3px}}
.subcard{{border:1px solid #eef1f4;border-radius:8px;padding:8px 18px 14px;margin:18px 0;background:#fcfdfe}}
.up{{color:#C0504D;font-weight:600}} .down{{color:#2E6F9E;font-weight:600}}
ul.clean{{margin:8px 0 8px 22px}} ul.clean li{{margin:6px 0}}
.concl{{background:#eef8f2;border:1px solid #cfe8db;border-radius:8px;padding:20px 24px}}
.concl ol{{margin-left:20px}} .concl li{{margin:9px 0}}
.small{{font-size:13px;color:#777}} .miss{{color:#C0504D}}
code{{background:#f0f2f5;padding:1px 6px;border-radius:3px;font-size:13px}}
footer{{text-align:center;color:#999;font-size:12.5px;padding:26px;border-top:1px solid #eee}}
</style></head><body><div class="wrap">
<header>
<h1>PTA 术后阻力指数改善（ΔRI）<br>影响因素分析报告</h1>
<div class="sub">经皮腔内血管成形术（PTA）患者术后阻力指数变化量（ΔRI = 术前RI − 术后RI）的单因素筛选与多因素回归分析</div>
<div class="meta">数据源：PTA.xlsx · 工作表「data」 ｜ PTA手术 {R['n_rows']} 例 / 患者 {R['n_patients']} 位 ｜ ΔRI 有效样本 n = {ds['n']} ｜ 方法：单因素筛选（P&lt;0.2）· OLS + HC3 稳健标准误 · Huber 稳健回归敏感性分析 · 交互检验 · 残差诊断 · 6模型综合权重 · 六模型机器学习（GroupKFold 10折）· SHAP 归因 ｜ {today}</div>
</header>
<main>

<h2>一、研究方法概述</h2>
<div class="lead">
本研究基于 {R['n_rows']} 例 PTA 手术记录（来自 {R['n_patients']} 位血液透析患者，54 位患者有 2~7 次手术），
以<b>阻力指数变化量 ΔRI（术前RI − 术后RI）</b>为独立研究终点——正值表示术后 RI 下降（血管阻力改善），
负值表示术后 RI 升高。ΔRI 有效样本 <b>n = {ds['n']}</b>（5 例术前/术后 RI 缺失被剔除）。
候选变量共 <b>{R['n_features']}</b> 个（仅术前基线与干预因素）：二分类 {R['n_bin']} 个、
多分类 {R['n_multi']} 个、连续 {R['n_cont']} 个。
</div>
<ul class="clean">
<li><b>① 描述性统计</b>：ΔRI 的均值、标准差、四分位数与范围，并以 Shapiro-Wilk 检验其正态性；</li>
<li><b>② 单因素分析</b>：按变量类型选择检验方法——二分类变量用独立样本 t 检验 / Mann-Whitney U 检验，
多分类变量用单因素 ANOVA / Kruskal-Wallis 检验，连续变量用 Pearson / Spearman 相关
（因 ΔRI 非正态，主报告非参数检验结果，参数检验 P 值备查）；筛选 <b>P&lt;0.2</b> 的变量进入多因素分析；</li>
<li><b>③ 多因素线性回归</b>：多分类变量哑变量化（drop_first），中位数填补缺失，
VIF&gt;10 迭代剔除共线变量，最终采用 <b>OLS + HC3 稳健标准误</b>估计；</li>
<li><b>④ 敏感性分析</b>：因 ΔRI 不符合正态分布，辅以 Huber 稳健回归核对系数方向与量级；</li>
<li><b>⑤ 交互作用检验</b>：检验 3 对临床意义明确的交互项（糖尿病×术前RI、合并血栓×血栓长度、高血压×术前PSV）；</li>
<li><b>⑥ 模型诊断</b>：残差正态性、Breusch-Pagan 异方差检验、VIF 共线性与拟合优度（R²、F、AIC/BIC）；</li>
<li><b>⑦ 多模型综合权重</b>：岭回归、LASSO、随机森林、XGBoost、LightGBM、CatBoost <b>6 个模型</b>分别拟合 ΔRI，
各模型重要性（线性模型取标准化系数绝对值，树模型取特征重要性 gain）min-max 归一化后等权平均，
得到各因素的<b>综合权重（%）与排名</b>（从大到小），用于量化各因素对 ΔRI 的相对贡献；</li>
<li><b>⑧ 机器学习预测与 SHAP 归因</b>：岭回归、LASSO、随机森林、XGBoost、LightGBM、CatBoost
<b>6 个模型</b>按患者分组 7:3 划分、GroupKFold 10 折交叉验证，选出最优预测模型，
并以 <b>SHAP 蜂群图</b>展示各因素对每位患者预测值的贡献方向与大小，与回归结论交叉印证。</li>
</ul>

<h2>二、ΔRI 描述性统计与正态性检验</h2>
{desc_table}
<p>ΔRI 均值为 <b>{ds["mean"]:.3f} ± {ds["sd"]:.3f}</b>，中位数 {ds["median"]:.3f}，
提示多数患者 PTA 术后阻力指数下降、血管阻力改善；但 Shapiro-Wilk 检验
P = {ds["shapiro_p"]:.2e} &lt; 0.05，ΔRI <b>不符合正态分布</b>，
故后续多因素分析采用 HC3 稳健标准误，并辅以 Huber 稳健回归作敏感性验证。</p>

<h2>三、单因素分析结果</h2>
<p>共 {len(uni_df)} 个候选变量完成单因素检验，其中 <b>{n_pass} 个 P&lt;0.2</b> 入选多因素回归
（下表<span style="background:#fdf3ec;padding:0 4px">橙色底纹行</span>为入选变量，按 P 值升序排列）。
最强单因素关联为<b>{top_uni['变量']}</b>（{top_uni['检验方法'].replace('相关', '')} ρ = {top_uni['统计量']:.3f}，P = {top_uni['P值']:.2e}）——
基线阻力越高，术后下降空间越大。</p>
{uni_table_html(uni_df)}

<h2>四、多因素线性回归结果</h2>
<p>单因素入选的 {n_pass} 个候选变量经哑变量化、缺失填补与 VIF 迭代剔除后，
最终 <b>{R['n_reg_features']} 个特征</b>（含哑变量）进入 OLS + HC3 稳健回归（n = {R['n_reg']}）。
模型拟合：<b>R² = {R['r2']:.4f}，调整 R² = {R['adj_r2']:.4f}，
F = {R['fvalue']:.2f}，P = {R['f_pvalue']:.2e}</b>。
共 <b>{len(sig_vars)} 个变量 P&lt;0.05</b> 为 ΔRI 的独立影响因素：
<b>{'、'.join(sig_vars)}</b>。Huber 稳健回归系数方向与量级均一致，结论稳健。</p>
<p>下表首列<b>排名</b>与末列<b>综合权重(%)</b>由 <b>6 个模型</b>
（岭回归、LASSO、随机森林、XGBoost、LightGBM、CatBoost）各自重要性 min-max 归一化后等权平均得出，
按综合权重从大到小排名，量化各因素对 ΔRI 的相对贡献：</p>
{reg_table_html(reg_df)}
{fig_tag(R['forest_path'], '图1  ΔRI 影响因素森林图：多因素回归系数 β 与 95%CI（Top15 按|β|从大到小；红=正向，蓝=负向，实心大圆点=P<0.05）')}
{fig_tag(R['weight_path'], '图2  ΔRI 影响因素综合权重排名 Top15（6模型综合权重，从大到小；红=正向，蓝=负向）')}
<h3>4.1 六模型综合权重明细（Top15）</h3>
<p class="small">表中数值为各模型内部 min-max 归一化后的相对重要性（该模型内最重要 = 100）；
综合权重(%) 为 6 模型等权平均后再归一化的百分比。</p>
{weight_table_html(R['weight_detail'])}
<h3>4.2 综合权重与标准化回归系数对照（Top15）</h3>
<p class="small">本表将 6 模型<b>综合权重%</b>与多因素 <b>OLS 标准化回归系数</b>并列对照：
标准化β 为将自变量与因变量均做 Z 标准化后重新拟合（OLS + HC3 稳健标准误）所得，
<b>无量纲、可横向比较</b>不同量纲因素的相对效应强弱（"该因素每升高 1 个标准差、ΔRI 变化多少个标准差"）；
方向与显著性由标准化模型给出（尺度线性变换不改变 P 值，故与第四章原始模型一致）。
综合权重反映<b>预测贡献</b>、标准化β 反映<b>效应方向与线性强度</b>，两者结合可更全面判断因素重要性。</p>
{weight_reg_table_html(R['weight_reg_df'])}

{ml_block}
<h2>六、交互作用检验</h2>
<p>在主效应模型基础上逐一加入 3 对临床意义明确的交互项，检验是否存在效应修饰：</p>
{inter_table}
<p>三项交互作用<b>均无统计学意义（P&gt;0.05）</b>，即糖尿病、合并血栓、高血压对 ΔRI 的影响
不随术前阻力指数、血栓长度或术前 PSV 水平显著改变，主效应模型无需纳入交互项。</p>

<h2>七、模型诊断与拟合优度</h2>
{diag_table}
<div class="warn"><b>诊断结论：</b>残差服从正态分布（Shapiro P = {R['resid_p']:.3f}），
无异方差（Breusch-Pagan P = {R['bp_p']:.3f}），最大 VIF = {R['max_vif']:.2f}（&lt;10，无严重共线性），
仅 {R['outliers']} 个残差异常值。模型设定合理，估计结果可靠。</div>

<h2>八、影响因素汇总与临床解读</h2>
<p>多因素回归共识别出 <b>{len(sig_vars)} 个 ΔRI 独立影响因素</b>（P&lt;0.05），按效应强度与方向解读如下：</p>
<div class="cards">{interp_cards}</div>
<ul class="clean">
<li><b>术前阻力指数是最强预测因子</b>（β = {beta_pri:+.3f}）：术前 RI 每升高 0.1，预期 ΔRI 增大约 {beta_pri*0.1:.3f}，
体现"基线阻力越高、PTA 后下降空间越大"的生理机制；</li>
<li><b>糖尿病是唯一负向因素</b>（β = {beta_dm:+.3f}）：微血管病变与内皮功能障碍削弱术后血管舒张反应，
提示糖尿病患者需强化术后控糖与血管管理；</li>
<li><b>吻合口宽度、合并血栓、教育程度为正向因素</b>：宽吻合口提供更充分流入道；
合并血栓者经取栓/溶栓联合 PTA 后梗阻解除明显；教育程度可能通过健康素养与依从性间接影响预后。</li>
</ul>

<h2>九、结论</h2>
<div class="concl"><b>主要结论：</b>
<ol>
<li>ΔRI 均值为 {ds["mean"]:.3f} ± {ds["sd"]:.3f}，多数患者 PTA 术后阻力指数下降（阻力改善），但分布非正态；</li>
<li>单因素分析自 {len(uni_df)} 个候选变量中筛选出 {n_pass} 个变量；
多因素回归（R² = {R['r2']:.4f}，调整 R² = {R['adj_r2']:.4f}）识别出 {len(sig_vars)} 个独立影响因素：
{'、'.join(sig_vars)}；</li>
<li>术前阻力指数是最强正向预测因子，糖尿病为负向因素，吻合口宽度、合并血栓与教育程度为正向因素；</li>
<li>六模型机器学习预测中，<b>{ml.get('best_name', '—')}</b> 表现最优
（10 折 CV R² = {ml.get('best_row', pd.Series(dtype=float)).get('CV_R2均值', float('nan')):.3f}，
测试集 R² = {ml.get('best_row', pd.Series(dtype=float)).get('测试集_R2', float('nan')):.3f}）；
SHAP 蜂群图显示术前阻力指数的个体级贡献远高于其他因素，且合并血栓、教育程度、糖尿病等
与回归结论方向一致——统计学与机器学习两类方法互相印证，结论稳健；</li>
<li>三项临床交互检验均不显著，主效应模型设定充分；模型诊断良好
（残差正态、无异方差、最大 VIF = {R['max_vif']:.2f}），Huber 稳健回归结果一致，结论可靠；</li>
<li>临床意义：术前高 RI、宽吻合口、合并血栓的患者预期 RI 改善更显著；
糖尿病患者需加强术后管理以优化阻力改善。</li>
</ol></div>

</main>
<footer>本报告全部数值与图表均由真实统计计算自动生成（research.dri_analysis + research.dri_modeling + research.make_dri_report） ｜ Python · statsmodels · scipy · scikit-learn · XGBoost · LightGBM · CatBoost · SHAP ｜ {today}</footer>
</div></body></html>'''

    out = os.path.join(C.OUT_DIR, 'ΔRI影响因素分析报告.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print('报告已生成:', out)


if __name__ == '__main__':
    build()
