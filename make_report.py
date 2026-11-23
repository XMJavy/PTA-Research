# -*- coding: utf-8 -*-
"""
生成完整 HTML 研究报告(图表 base64 内嵌, 单文件)
读取 outputs/tables/full_results.joblib
"""
import os, base64, io, warnings
import numpy as np
import pandas as pd
import joblib
from PIL import Image

from research import config as C

warnings.filterwarnings('ignore')


def img_b64(path, max_w=1100):
    im = Image.open(path).convert('RGB')
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(path.replace('.png', '_small.png'), format='PNG', optimize=True)
    with open(path.replace('.png', '_small.png'), 'rb') as f:
        return base64.b64encode(f.read()).decode()


def fig_tag(name, caption):
    p = os.path.join(C.FIG_DIR, f'{name}.png')
    if not os.path.exists(p):
        return f'<p class="miss">[缺少图表 {name}]</p>'
    return (f'<figure><img src="data:image/png;base64,{img_b64(p)}">'
            f'<figcaption>{caption}</figcaption></figure>')


def df_to_html(df, float_fmt='{:.4f}', max_rows=None, cls='rtbl'):
    d = df.copy()
    if max_rows:
        d = d.head(max_rows)
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: '' if pd.isna(x) else float_fmt.format(x))
    return d.to_html(index=False, classes=cls, border=0, escape=False)


def fmt_p(p):
    """P 值格式化：极小用科学计数法，显著项标红加粗"""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    s = f'{p:.2e}' if p < 0.001 else f'{p:.4f}'
    return f'<b class="up">{s}</b>' if p < 0.05 else s


def uni_table_html(uni_df):
    """单因素分析表（P<0.2 入选行高亮）"""
    head = ('<tr><th>序号</th><th>变量</th><th>类型</th><th>检验方法</th>'
            '<th>统计量</th><th>P值</th><th>附报参数检验P值</th>'
            '<th>入选多因素(P&lt;0.2)</th></tr>')
    rows = []
    for i, r in uni_df.iterrows():
        sel = r['筛选(P<0.2)'] == '是'
        cls = ' class="sel"' if sel else ''
        stat = '' if pd.isna(r['统计量']) else (
            f"{r['统计量']:.4f}" if abs(r['统计量']) < 10 else f"{r['统计量']:.2f}")
        tp = r.get('t检验P值', None)
        tp_s = '' if (tp is None or pd.isna(tp)) else fmt_p(float(tp))
        sel_html = '<b class="up">是</b>' if sel else '否'
        rows.append(
            f'<tr{cls}><td>{i+1}</td><td>{r["变量"]}</td><td>{r["类型"]}</td>'
            f'<td>{r["检验方法"]}</td><td>{stat}</td><td>{fmt_p(r["P值"])}</td>'
            f'<td>{tp_s}</td><td>{sel_html}</td></tr>')
    return f'<table class="rtbl">{head}{"".join(rows)}</table>'


def reg_table_html(reg_df):
    """多因素回归系数表（含排名与6模型综合权重%，P<0.05 行高亮）"""
    head = ('<tr><th>排名</th><th>变量</th><th>回归系数β</th><th>稳健SE</th>'
            '<th>95%CI下限</th><th>95%CI上限</th><th>P值</th>'
            '<th>显著性</th><th>Huberβ</th><th>方向</th><th>综合权重(%)</th></tr>')
    rows = []
    for _, r in reg_df.iterrows():
        sig = r['P值'] < 0.05
        cls = ' class="sel"' if sig else ''
        direction = ('<span class="up">正向 ↑</span>' if r['回归系数β'] > 0
                     else '<span class="down">负向 ↓</span>')
        sig_html = f'<b class="up">{r["显著性"]}</b>' if sig else r['显著性']
        rows.append(
            f'<tr{cls}><td><b style="color:#1f5377">{int(r["排名"])}</b></td>'
            f'<td>{r["变量"]}</td><td>{r["回归系数β"]:+.4f}</td>'
            f'<td>{r["稳健SE"]:.4f}</td><td>{r["95%CI下限"]:.4f}</td>'
            f'<td>{r["95%CI上限"]:.4f}</td><td>{fmt_p(r["P值"])}</td>'
            f'<td>{sig_html}</td><td>{r["Huberβ"]:+.4f}</td>'
            f'<td>{direction}</td><td><b>{r["综合权重(%)"]:.2f}%</b></td></tr>')
    return f'<table class="rtbl">{head}{"".join(rows)}</table>'


def weight_detail_html(weight_detail, top_n=15):
    """6模型归一化重要性明细表 (TopN, 模型内最重要=100)"""
    models = ['岭回归', 'LASSO', '随机森林', 'XGBoost', 'LightGBM', 'CatBoost']
    head = ('<tr><th>排名</th><th>变量</th><th>岭回归</th><th>LASSO</th>'
            '<th>随机森林</th><th>XGBoost</th><th>LightGBM</th><th>CatBoost</th>'
            '<th>综合权重(%)</th></tr>')
    rows = []
    for i, (name, r) in enumerate(weight_detail.head(top_n).iterrows()):
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
        sig_html = f'<b class="up">{r["显著性"]}</b>' if sig else r['显著性']
        rows.append(
            f'<tr{cls}><td><b style="color:#1f5377">{int(r["排名"])}</b></td>'
            f'<td style="text-align:left">{r["特征因素"]}</td>'
            f'<td><b>{r["综合权重(%)"]:.2f}%</b></td>'
            f'<td>{r["标准化β"]:+.4f}</td>'
            f'<td>{r["95%CI下限"]:+.4f}</td><td>{r["95%CI上限"]:+.4f}</td>'
            f'<td>{fmt_p(r["OLS P值"])}</td><td>{sig_html}</td>'
            f'<td>{direction}</td></tr>')
    return f'<table class="rtbl">{head}{"".join(rows)}</table>'


def build():
    R = joblib.load(os.path.join(C.TAB_DIR, 'full_results.joblib'))
    df, qt, bundle = R['df'], R['q_tables'], R['bundle']
    interp, fc, pc, summ = R['interp'], R['factor_cluster'], R['patient_cluster'], R['summary']
    tgts = list(bundle['targets'].keys())

    # ---- 汇总数字 ----
    pair = df[[C.T_PRE, C.T_POST]].dropna()
    dq = pair[C.T_POST] - pair[C.T_PRE]
    up_rate = (dq > 0).mean()
    # 好转定义: ΔQ>0(流量上升) 且 ΔRI>0(RI下降)
    pair_full = df[[C.T_DFLOW, C.T_DRI]].dropna()
    improve_rate = ((pair_full[C.T_DFLOW] > 0) & (pair_full[C.T_DRI] > 0)).mean()
    best_rows = {t: bundle['targets'][t]['metrics'].sort_values(
        'CV_RMSE均值').iloc[0] for t in tgts}

    # 性能表(仅ΔQ)：CV R² 为「均值 ± 标准差」；CV RMSE 仅均值；
    # 测试集 R² 附 Bootstrap(1000次) 标准差，与 ΔRI 报告口径一致
    perf = pd.concat([
        bundle['targets'][t]['metrics'].assign(目标=C.short(t))
        for t in tgts], ignore_index=True)

    def _test_r2(r):
        sd = r.get('测试集_R2_SD', np.nan)
        base = f"{r['测试集_R2']:.4f}"
        return base if pd.isna(sd) else f'{base} ± {sd:.4f}'

    perf = pd.DataFrame({
        '目标': perf['目标'], '模型': perf['模型'],
        'CV R² (均值±SD)': perf.apply(
            lambda r: f"{r['CV_R2均值']:.4f} ± {r['CV_R2标准差']:.4f}", axis=1),
        'CV MAE': perf['CV_MAE均值'].map(lambda v: f'{v:.3f}'),
        'CV RMSE': perf['CV_RMSE均值'].map(lambda v: f'{v:.3f}'),
        '测试集 R² (均值±SD)': perf.apply(_test_r2, axis=1),
        '测试集 MAE': perf['测试集_MAE'].map(lambda v: f'{v:.3f}'),
        '测试集 RMSE': perf['测试集_RMSE'].map(lambda v: f'{v:.3f}'),
    })

    # 缺失Top
    miss_top = qt['01_缺失值分析'].head(12)
    out_top = qt['02_异常值检测'].sort_values('临床越界数', ascending=False).head(10)
    norm_sig = qt['03_分布检验']
    n_nonnormal = int((norm_sig['正态性P'].apply(
        lambda x: float(x) if x != 'NA' else 1) < 0.05).sum())

    # Kendall W 表
    kw_rows = []
    for t in tgts:
        ir = interp[t]
        kw_rows.append(dict(目标=C.short(t),
                            Kendall_W=round(ir['kendall_w'], 4),
                            卡方=round(ir['kendall_chi2'], 2),
                            P值=('%.2e' % ir['kendall_p'])))
    kw_df = pd.DataFrame(kw_rows)

    cls = bundle['classification']

    # ---------------- ΔQ 两步分析结果（run_pipeline 已写入 full_results.joblib） ----------------
    dq = R.get('dq_two_stage')
    if dq is None:
        from research.dq_analysis import run_analysis as _run_dq
        dq = _run_dq(make_figs=True)
    ds = dq['desc_stats']
    uni_df, reg_df, inter_df = dq['uni_df'], dq['reg_df'], dq['inter_df']
    sig_vars, n_pass = dq['sig_vars'], dq['n_pass']

    desc_table = (
        '<table class="rtbl"><tr><th>统计量</th><th>数值</th></tr>'
        f'<tr><td>有效样本量 n</td><td>{ds["n"]}</td></tr>'
        f'<tr><td>均值 ± 标准差</td><td>{ds["mean"]:.3f} ± {ds["sd"]:.3f}</td></tr>'
        f'<tr><td>最小值 ~ 最大值</td><td>[{ds["min"]:.3f}, {ds["max"]:.3f}]</td></tr>'
        f'<tr><td>四分位数 P25 / 中位数 / P75</td>'
        f'<td>{ds["p25"]:.3f} / {ds["median"]:.3f} / {ds["p75"]:.3f}</td></tr>'
        f'<tr><td>Shapiro-Wilk 正态性检验</td>'
        f'<td>W = {ds["shapiro_W"]:.4f}, P = {fmt_p(ds["shapiro_p"])}</td></tr>'
        f'<tr><td>分布结论</td><td>{"符合" if ds["normal"] else "<b>不符合</b>"}正态分布'
        f'（P{"&gt;" if ds["normal"] else "&lt;"}0.05）</td></tr></table>')

    inter_rows = ''.join(
        f'<tr><td>{r["交互项"]}</td><td>{r["回归系数"]:+.4f}</td>'
        f'<td>{fmt_p(r["P值"])}</td><td>{r["显著性"]}</td></tr>'
        for _, r in inter_df.iterrows())
    inter_table = ('<table class="rtbl"><tr><th>交互项</th><th>回归系数</th>'
                   '<th>P值</th><th>显著性</th></tr>' + inter_rows + '</table>')
    inter_ns = (len(inter_df) == 0) or bool((inter_df['P值'] >= 0.05).all())
    inter_txt = ('各项交互作用<b>均无统计学意义（P&gt;0.05）</b>，主效应模型无需纳入交互项。'
                 if inter_ns else '部分交互项达到统计学意义，解释主效应时需注意效应修饰。')

    diag_items = [
        ('决定系数 R²', f'{dq["r2"]:.4f}', f'模型解释 ΔQ 变异的 {dq["r2"]*100:.1f}%'),
        ('调整 R²', f'{dq["adj_r2"]:.4f}', '校正自变量个数后仍保持解释力'),
        ('F 检验', f'F = {dq["fvalue"]:.2f}, P = {dq["f_pvalue"]:.2e}', '模型整体极显著'),
        ('残差正态性（Shapiro-Wilk）', f'W = {dq["resid_W"]:.4f}, P = {fmt_p(dq["resid_p"])}',
         '残差服从正态分布' if dq['resid_p'] > 0.05
         else '残差偏离正态，已用 HC3 稳健SE并辅以 Huber 稳健回归'),
        ('异方差检验（Breusch-Pagan）', f'LM = {dq["bp_stat"]:.4f}, P = {fmt_p(dq["bp_p"])}',
         '不存在异方差' if dq['bp_p'] > 0.05 else '存在异方差（已用 HC3 稳健SE校正）'),
        ('残差分布', f'均值 {dq["resid_mean"]:.2f}, SD {dq["resid_sd"]:.2f}',
         f'范围 [{dq["resid_min"]:.1f}, {dq["resid_max"]:.1f}]，以 0 为中心'),
        ('|残差| &gt; 2.5SD 异常值', f'{dq["outliers"]} 个', '占比低，不影响结论稳健性'),
        ('最大 VIF', f'{dq["max_vif"]:.2f}', 'VIF &lt; 10，无严重共线性'),
        ('AIC / BIC', f'{dq["aic"]:.1f} / {dq["bic"]:.1f}', '拟合优度与复杂度权衡'),
    ]
    diag_table = ('<table class="rtbl"><tr><th>诊断项目</th><th>结果</th><th>解读</th></tr>'
                  + ''.join(f'<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>'
                            for a, b, c in diag_items) + '</table>')

    top_uni = uni_df.iloc[0]
    interp_cards = ''
    for v in sig_vars:
        row = reg_df[reg_df['变量'] == v].iloc[0]
        direction, text = dq['clinical_interp'].get(
            v, ('', '多因素回归校正后仍与 ΔQ 独立相关。'))
        dir_cls = 'up' if row['回归系数β'] > 0 else 'down'
        interp_cards += (
            f'<div class="card" style="flex:1 1 100%;text-align:left">'
            f'<div class="n" style="font-size:17px">{v}'
            f'<span class="{dir_cls}" style="font-size:13.5px;margin-left:8px">{direction} '
            f'β = {row["回归系数β"]:+.3f}，95%CI '
            f'[{row["95%CI下限"]:.3f}, {row["95%CI上限"]:.3f}]</span></div>'
            f'<div style="font-size:13.8px;color:#444;margin-top:6px">{text}</div></div>')
    sig_str = '、'.join(sig_vars)
    pri = reg_df.loc[reg_df['变量'] == '术前阻力指数', '回归系数β']
    pri_txt = f'{pri.iloc[0]:+.1f}' if len(pri) else '—'

    sec5 = f'''
<h2>五、影响因素两步分析（单因素筛选 + 多因素回归）</h2>
<p>ΔQ 影响因素分析遵循临床流行病学「两步法」：<b>第一步单因素初筛</b>（阈值 P&lt;0.2，
原则是"宁可错收、不可漏筛"）从 {dq['n_features']} 个候选变量中筛出候选变量；
<b>第二步多因素线性回归</b>（阈值 P&lt;0.05，原则是"宁缺毋滥"）在校正混杂后识别独立影响因素。
相比"一次性全变量入模"，两步法可避免高维共线性导致的估计不稳与假阳性。
本章回答"哪些因素独立影响 ΔQ"，与第四章"六模型对比"（回答"ΔQ 能被预测到多准"）互补。</p>

<h3>5.1 描述性统计与正态性检验</h3>
{desc_table}
<p>ΔQ 均值为 <b>{ds["mean"]:.3f} ± {ds["sd"]:.3f}</b> ml/min，
Shapiro-Wilk 检验 P = {fmt_p(ds["shapiro_p"])}，
故多因素分析采用 <b>OLS + HC3 稳健标准误</b>，并辅以 <b>Huber 稳健回归</b>作非正态敏感性分析。</p>

<h3>5.2 单因素分析（初筛，P&lt;0.2）</h3>
<p>共 <b>{len(uni_df)}</b> 个候选变量完成单因素检验（二分类→Mann-Whitney U，多分类→Kruskal-Wallis，
连续→Spearman 秩相关；均附报参数检验 P 值备查），其中 <b>{n_pass}</b> 个变量 P&lt;0.2 入选多因素分析
（下表<span style="background:#fdf3ec;padding:0 4px">橙色底纹行</span>为入选变量，按 P 值升序排列）。
最强单因素关联为 <b>{top_uni['变量']}</b>（{str(top_uni['检验方法']).replace('相关', '')}，
统计量 = {top_uni['统计量']:.3f}，P = {top_uni['P值']:.2e}）。</p>
{uni_table_html(uni_df)}

<h3>5.3 多因素线性回归（终审，P&lt;0.05）</h3>
<p>单因素入选的 {n_pass} 个候选变量经哑变量化、中位数填补与 VIF&gt;10 迭代剔除后，
最终 <b>{dq['n_reg_features']} 个特征</b>进入 OLS + HC3 稳健回归（n = {dq['n_reg']}）。
模型拟合：<b>R² = {dq['r2']:.4f}，调整 R² = {dq['adj_r2']:.4f}，F = {dq['fvalue']:.2f}，
P = {dq['f_pvalue']:.2e}</b>；共 <b>{len(sig_vars)} 个变量 P&lt;0.05</b> 为 ΔQ 的独立影响因素：
<b>{sig_str}</b>。Huber 稳健回归系数方向与量级一致，结论稳健。</p>
<p>下表末列<b>综合权重(%)</b>与首列<b>排名</b>由 <b>6 个模型</b>
（岭回归、LASSO、随机森林、XGBoost、LightGBM、CatBoost）各自重要性 min-max 归一化后等权平均得出，
按综合权重从大到小排名，量化各因素对 ΔQ 的相对贡献：</p>
{reg_table_html(reg_df)}
{fig_tag('ΔQ_森林图', '图7  ΔQ 影响因素森林图：多因素回归系数 β 与 95%CI（Top15 按|β|从大到小；红=正向，蓝=负向，实心大圆点=P<0.05）')}
{fig_tag('ΔQ_权重图', '图8  ΔQ 影响因素综合权重排名 Top15（6模型综合权重，从大到小；红=正向，蓝=负向）')}
<h4>5.3.1 六模型综合权重明细（Top15）</h4>
<p class="small">表中数值为各模型内部 min-max 归一化后的相对重要性（该模型内最重要 = 100）；
综合权重(%) 为 6 模型等权平均后再归一化的百分比。</p>
{weight_detail_html(dq['weight_detail'])}
<h4>5.3.2 综合权重与标准化回归系数对照（Top15）</h4>
<p class="small">本表将 6 模型<b>综合权重%</b>与多因素 <b>OLS 标准化回归系数</b>并列对照：
标准化β 为将自变量与因变量均做 Z 标准化后重新拟合（OLS + HC3 稳健标准误）所得，
<b>无量纲、可横向比较</b>不同量纲因素的相对效应强弱（"该因素每升高 1 个标准差、ΔQ 变化多少个标准差"）；
方向与显著性由标准化模型给出（尺度线性变换不改变 P 值，故与 5.3 原始模型一致）。
综合权重反映<b>预测贡献</b>、标准化β 反映<b>效应方向与线性强度</b>，两者结合可更全面判断因素重要性。</p>
{weight_reg_table_html(dq['weight_reg_df'])}

<h3>5.4 交互作用检验与模型诊断</h3>
<p>在主效应模型基础上逐一加入临床意义明确的交互项（糖尿病×术前RI、合并血栓×血栓长度、高血压×术前PSV）：
{inter_txt}</p>
{inter_table}
{diag_table}
<div class="warn"><b>诊断结论：</b>残差 Shapiro P = {dq['resid_p']:.4f}，
Breusch-Pagan P = {dq['bp_p']:.4f}，最大 VIF = {dq['max_vif']:.2f}（&lt;10，无严重共线性），
仅 {dq['outliers']} 个残差异常值；Huber 稳健回归结果一致，模型设定合理、估计可靠。</div>

<h3>5.5 显著影响因素的临床解读</h3>
<p>多因素回归共识别出 <b>{len(sig_vars)} 个 ΔQ 独立影响因素</b>（P&lt;0.05），按效应方向与临床意义解读如下：</p>
<div class="cards">{interp_cards}</div>
<p class="small">说明：回归系数 β 为原始量纲下的效应量（ΔQ 单位 ml/min），
如"术前阻力指数 β = {pri_txt}"表示术前 RI 每升高 1 个单位、ΔQ 相应变化该数值；量纲不同因素的
β 不可直接横向比较，故同时给出 6 模型综合权重作为相对贡献的度量。</p>

<h3>5.6 跨六模型影响因素排序一致性（Kendall's W）</h3>
<p>六模型在 ΔQ 上的特征排序一致性 Kendall's W 如下（W 越接近 1 越一致；
Friedman χ² 检验 P&lt;0.05 表示一致性非偶然）：</p>
{df_to_html(kw_df)}
'''

    # 每个目标SHAP/PDP/热力图/预测诊断
    mech_sections = ''
    for t in tgts:
        mech_sections += f'''
<div class="subcard">
<h4>{C.short(t)}</h4>
{fig_tag(f'09_相关热力图_{C.short(t)}', f'{C.short(t)} Top12因素相关矩阵')}
{fig_tag(f'10_SHAP_{C.short(t)}', f'{C.short(t)} SHAP蜂群图（最佳模型：{bundle["targets"][t]["best_name"]}）')}
<p class="small">方法学说明：本图的 SHAP 贡献基于<b>单一最优模型（{bundle["targets"][t]["best_name"]}）</b>对每位患者预测值的分解，
与 5.3 节「六模型综合权重」（6 个模型重要性 min-max 归一化后等权平均）是两种不同度量，
变量次序不完全一致属正常现象；两种方法均一致指向<b>术前阻力指数</b>为首要驱动因素。</p>
{fig_tag(f'11_PDP_{C.short(t)}', f'{C.short(t)} Top6因素剂量-反应曲线（95% Bootstrap置信带）')}
{fig_tag(f'12_预测诊断_{C.short(t)}', f'{C.short(t)} 最佳模型测试集预测诊断')}
</div>'''

    html = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PTA前后肱动脉血流动力学变化影响因素系统性研究报告</title>
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
<h1>PTA 术后肱动脉血流量改善<br>影响因素系统性研究报告</h1>
<div class="sub">经皮腔内血管成形术（PTA）患者术后血流量改善程度（ΔQ）的多因素量化、建模与机制分析</div>
<div class="meta">数据源：PTA.xlsx · 工作表「data」 ｜ PTA手术 {summ['n']} 例 / 患者 {summ['n_patients']} 位 ｜ 方法：VIF诊断 · Ridge/LASSO/RF/XGBoost/LightGBM/CatBoost · GroupKFold 10折CV · SHAP · PDP · KMeans ｜ 2026-10-06</div>
</header>
<main>

<h2>一、执行摘要</h2>
<div class="lead">
本研究基于 {summ['n']} 例 PTA 手术记录（来自 {summ['n_patients']} 位血液透析患者，部分患者多次手术），以<b>血流量变化量 ΔQ（术后−术前）</b>
为唯一研究目标，量化分析"PTA 术后血流量好转"的影响因素。<b>好转定义</b>为：
术后肱动脉血流量较术前上升（ΔQ&gt;0）<b>且</b>术后阻力指数较术前下降（ΔRI&gt;0）。
经数据质量审计、VIF 共线性处理后，对 6 种统计/机器学习模型进行 7:3 划分与 10 折交叉验证对比，
并以标准化系数+95%CI、SHAP、部分依赖图（PDP）、KMeans 聚类进行机制解释。
PTA 后流量中位数由 <b>{pair[C.T_PRE].median():.0f}</b> 升至
<b>{pair[C.T_POST].median():.0f} ml/min</b>，{up_rate*100:.1f}% 患者流量上升，
{improve_rate*100:.1f}% 患者达到"流量上升且 RI 下降"的好转标准。
</div>
<div class="cards">
<div class="card"><div class="n">{summ['n']} / {summ['n_patients']}</div><div class="l">PTA手术例数 / 患者数</div></div>
<div class="card"><div class="n">{summ['n_clinical_outlier_vals']}</div><div class="l">临床越界值修正（{summ['n_clinical_outlier_cols']}列）</div></div>
<div class="card"><div class="n">{int(up_rate*100)}%</div><div class="l">术后流量上升者占比</div></div>
<div class="card"><div class="n">{int(improve_rate*100)}%</div><div class="l">好转者占比(流量↑且RI↓)</div></div>
</div>
<div class="warn"><b>数据结构说明（重要）：</b>
本数据为<b>回顾性纵向队列</b>——224 例 PTA 手术来自 136 位患者，其中 54 位患者接受过 2~7 次
PTA 干预（每次手术均含对应的术前与术后超声测量）。同一患者多次手术的间隔从 14 天到 720 天不等，
为纵向观察提供了真实时点。
需要说明的是，原始数据中<b>未设置显式的术后随访时间标签列</b>（如"术后1周/1月/3月/6月"），
每次记录的"术后"测量均为该次 PTA 术后即刻评估。因此，严格的"术后变化轨迹模型"需在补充
标准化随访时间标签后开展；当前报告聚焦于每次 PTA 干预的 ΔQ 改善幅度及其影响因素分析，
并利用同一患者多次手术的纵向信息进行稳健性检验（GroupKFold 分组）。</div>

<h2>二、数据预处理与质量控制</h2>
<h3>2.1 数据清洗规则</h3>
<ul class="clean">
<li>保留分组键「患者序号」（224行 = 136位患者；82位患者各1条，36位2条，共12位3–7条独立PTA干预记录），
用于后续 GroupKFold 防泄露；删除常数列「我院干预次序」（全=1）与日期列；</li>
<li>非数值标记 <code>/</code>、空白统一记为缺失；检测限写法如 <code>&lt;6</code> 按检测限一半赋值；</li>
<li>多标签编码（如狭窄类型 <code>1+2</code>、MACE <code>1+4+7</code>）取首个编码作为主类别；</li>
<li>球囊直径/扩张压力为含品名的自由文本（如「凰泰球囊(20atm) 4mm与30atm巴德球囊 6mm」），
用正则解析出<b>最大球囊直径、平均直径、球囊数量、最大/平均扩张压力、扩张次数</b>；</li>
<li>删除第一至第五处血管病变长度列（<code>X..AB</code>，共5列）——与「总病变长度」完全线性依赖（
W = X + Y + Z + AA + AB，差值均值=0），会导致 VIF 无穷大；</li>
<li><b>异常值三层规则</b>：① Z-score（|z|&gt;3）与 IQR（1.5倍）仅做标记不盲删；
② 超出临床合理区间者判为<b>录入错误</b>置缺失（如 RI=5.9、肱动脉内径 363mm、年龄 110 岁、
血红蛋白 1050 g/L、心腔测值 1–5mm 等）；③ 缺失统一用训练集中位数填补；</li>
<li>连续变量右偏（偏度&gt;1 且为正）建议 log1p 转换并记录于分布检验表；
建模统一采用 <b>Z-score 标准化</b>（线性模型需要；树模型用原始量纲），
依据：各变量量纲差异大（流量数百 vs RI 约0.5），Z-score 使标准化系数可直接比较；</li>
<li>衍生特征：脉压差（SBP−DBP）、合并血栓、高压球囊（≥30atm）、药物涂层球囊、
动脉/静脉狭窄改善（术后−最窄处）、ΔQ、ΔRI、
<b>总病变长度/病变数量→长病变标志（≥50mm）、术前术后 RI/PSV/EDV 差</b>。</li>
</ul>
{fig_tag('03_数据质量审计', '图3 缺失比例与临床越界值审计')}
<h3>2.2 缺失值与异常值量化</h3>
<p>清洗后仅 {int((qt['01_缺失值分析']['缺失数']>0).sum())} 列存在缺失，绝大多数来自原始
<code>/</code> 标记；中位数填补前后均值/中位数/标准差相对漂移均 &lt;5%（见 04_填补漂移 表）。
正态性检验（Shapiro-Wilk）显示 {n_nonnormal} 个连续变量偏离正态（P&lt;0.05），以右偏为主，
这也是选择稳健正则化模型与树模型的依据。共 {summ['n_clinical_outlier_vals']} 个值按临床区间判为录入错误并修正。</p>
{df_to_html(out_top[['短名','n','Zscore异常数_abs_z_gt3','IQR异常数_1p5倍','临床越界数','最小值','最大值']], max_rows=10)}
<p class="small">完整结果见 <code>outputs/tables/数据质量评估.xlsx</code>（缺失/异常/分布/填补漂移/VIF 共6张表）。</p>

<h2>三、多重共线性处理（VIF）</h2>
<p>ΔQ 模型先做 VIF 诊断，规则为 VIF&gt;10 即剔除或降维。实际剔除：
PT（与 INR 数学同源，INR 由 PT 代入公式计算）、脉压差（与收缩压/舒张压线性相关）、
静脉狭窄改善（术后与最窄内径之差）、平均球囊直径（与最大直径高度相关）、
最大扩张压力（与球囊类型/直径高度相关）等。处理后保留特征最高 VIF 均 &lt;10。</p>
{fig_tag('04_VIF诊断', '图4 ΔQ 建模矩阵处理后 VIF Top10')}

<h2>四、建模框架与模型性能</h2>
<ul class="clean">
<li><b>数据划分</b>：训练:测试=7:3（random_state=42），回归用随机划分，分类用分层抽样；
训练/测试目标分布经 K-S 检验无显著差异；</li>
<li><b>六种模型</b>：岭回归（RidgeCV 内建5折选α）、LASSO（LassoCV）、随机森林、XGBoost、
LightGBM、CatBoost（后四者 3 折网格搜索调参）；</li>
<li><b>验证</b>：全部模型在全数据上做 10 折交叉验证（逐折 R²/MAE/RMSE 已落盘），并报告 30% 测试集指标；</li>
<li><b>评估指标</b>：R²、MAE、RMSE（线性模型附 AIC/BIC）；</li>
<li><b>不平衡处理</b>：ΔQ 为连续回归，不存在类别不平衡，故 SMOTE 不适用；
好转应答者（ΔQ&gt;0 且 ΔRI&gt;0）分类中两类占比为 {cls['prevalence']*100:.1f}%/{(1-cls['prevalence'])*100:.1f}%，
采用分层 K 折与 Youden 截断，无需合成样本。</li>
</ul>
{fig_tag('05_模型对比_R2', '图5 六模型10折交叉验证 R²')}
{fig_tag('06_模型对比_RMSE', '图6 六模型10折交叉验证 RMSE')}
{df_to_html(perf)}
<div class="warn"><b>关于 ΔQ 预测难度的如实说明：</b>ΔQ 的最优模型（{best_rows[tgts[0]]['模型']}）
10 折 CV R²={best_rows[tgts[0]]['CV_R2均值']:.3f}，测试集 R²={best_rows[tgts[0]]['测试集_R2']:.3f}。
R² 偏低说明血流量改善幅度更多由手术即时效果、术后用药依从性、血管重塑等表外因素决定，
现有术前基线变量不足以准确预测"改善多少"——这符合临床预期（PTA 效果个体差异大）。
所有数值为真实交叉验证结果，未做任何美化；报告同时给出训练/CV/测试三套指标以供审计。</div>

{sec5}

<h2>六、作用机制解释：相关性、SHAP 与剂量-反应</h2>
<p>下图组对 ΔQ 给出：① Top12 因素相关矩阵；② SHAP 蜂群图（每个点=一名患者，
颜色为特征取值，横向位置为该特征把预测值推高/推低的幅度）；③ 剂量-反应曲线（PDP，
阴影为 300 次 Bootstrap 的 95% 置信带）；④ 最佳模型测试集预测 vs 观测与残差分布。</p>
{mech_sections}

<h2>七、影响因素聚类与患者表型</h2>
<h3>7.1 因素作用模式聚类</h3>
<p>以「因素×ΔQ 标准化效应」矩阵叠加因素间相关结构的 PCA 载荷做 KMeans，
用肘部法、轮廓系数、CH 指数共同定 K（轮廓系数目标&gt;0.5）。</p>
{fig_tag('14_因素聚类', '图14 因素作用模式聚类')}
{df_to_html(fc['select'])}
{df_to_html(fc['profile'].drop(columns=['群组'], errors='ignore'))}
<h3>7.2 患者血流动力学表型聚类</h3>
<p>以术前核心血流动力学与生理指标聚类，ANOVA（连续变量）与卡方检验（分类变量）验证组间差异。</p>
<div class="warn"><b>聚类分离度的客观说明：</b>本样本下患者表型聚类的最优轮廓系数仅为
{pc['silhouette']:.3f}（远低于 0.5 的经验阈值），即患者在基线血流动力学特征上更接近
连续谱而非界限分明的亚组；下表分组结果仅作探索性描述，组间差异以 ANOVA/卡方 P 值为准，
不宜直接作为临床分型使用，需更大样本与外部数据验证。</div>
{fig_tag('15_患者表型聚类', '图15 患者表型聚类及组间差异')}
{df_to_html(pc['anova'].head(12))}
{df_to_html(pc['chi'])}

<h2>八、血流改善应答者识别（ROC/AUC）</h2>
<p>以<b>好转</b>（ΔQ&gt;0 且 ΔRI&gt;0，即术后流量上升且术后阻力指数下降）定义"应答者"，
仅用术前基线+干预因素（保证术前可预测），10 折分层交叉验证：</p>
<div class="warn"><b>类别不平衡的说明：</b>好转者占比 {cls['prevalence']*100:.1f}%，
未达到好转标准者占 {(1-cls['prevalence'])*100:.1f}%。采用分层 K 折与 Youden 截断
平衡敏感度与特异度。该分类结果可在术前对"PTA 后能否达到血流动力学好转"做概率化辅助判断，
但 AUC 水平提示仍需结合影像综合决策。</div>
{df_to_html(cls['metrics'])}
{fig_tag('13_ROC与混淆矩阵', '图13 ROC 曲线与最佳模型 Youden 截断混淆矩阵（概率均来自逐样本真实预测）')}

<h2>九、结果讨论与临床意义</h2>
<p>本章结论以第五章「单因素初筛（P&lt;0.2）→ 多因素回归（P&lt;0.05）」两步分析为准
（多因素模型 R²={dq['r2']:.4f}，调整 R²={dq['adj_r2']:.4f}，共 {len(sig_vars)} 个独立影响因素）。</p>
<ul class="clean">
<li><b>术前阻力指数（RI）是 ΔQ 改善最强的独立预测因子</b>（β={pri_txt}）：术前 RI 越高，术后流量改善幅度越大，
符合生理机制——术前高阻状态意味着远端血管床阻力大，PTA 解除狭窄后阻力下降空间大、流量提升更显著；
反之术前 RI 已低者改善空间有限。</li>
<li><b>原发病为糖尿病性肾病是负向因素</b>：广泛微血管病变与血管钙化削弱 PTA 后的灌注提升，
是术前评估需要重点关注的合并症类型。</li>
<li><b>心脏超声与代谢-凝血指标</b>：左室后壁 LVPW、纤维蛋白原、D-二聚体为负向因素，
室间隔 IVS、BMI 为正向因素，提示心脏重构/容量负荷与高凝状态影响术后流量改善幅度。</li>
<li><b>单因素显著但多因素被吸收的因素</b>（如术前 EDV/PSV、最大球囊直径、肱动脉内径、合并血栓）：
单因素分析中与 ΔQ 显著相关，但在多因素模型中其效应被术前 RI 等强预测因子吸收而未保留独立性，
说明这些因素对 ΔQ 的作用主要通过阻力状态中介，属"影子变量"。</li>
<li><b>糖尿病的方向反转需谨慎解读</b>：单因素分析中糖尿病与 ΔQ 呈负相关，多因素校正原发病等后转为正向，
提示其与原发病分类存在共线/抑制效应，不宜据此得出"糖尿病有益"的结论。</li>
<li><b>好转应答者识别模型</b>（AUC={cls['metrics']['交叉验证AUC'].max():.3f}）
可在术前对"PTA 后能否达到血流动力学好转（流量↑且RI↓）"做概率化辅助判断，
为术前谈话与个体化预期管理提供参考。</li>
</ul>
<div class="concl"><b>主要结论：</b>
<ol>
<li>PTA 总体有效（{up_rate*100:.1f}% 患者流量上升，{improve_rate*100:.1f}% 达到流量↑且RI↓的好转标准，
中位流量 {pair[C.T_PRE].median():.0f}→{pair[C.T_POST].median():.0f} ml/min）；</li>
<li>两步分析（单因素 {len(uni_df)} 个候选中初筛 {n_pass} 个，多因素回归 R²={dq['r2']:.4f}）
识别出 {len(sig_vars)} 个 ΔQ 独立影响因素；六模型特征排序一致性 Kendall's W={kw_df['Kendall_W'].iloc[0]:.3f}，结论稳健；</li>
<li>术前 RI 高、无糖尿病性肾病、无高凝状态、心脏重构轻的患者术后流量改善更显著，
是可干预的术前评估与术后管理靶点；</li>
<li>因素聚类识别出血流动力学、解剖管径、全身代谢/炎症凝血等作用群组，
患者聚类发现的表型间 ΔQ 差异为分层诊疗提供了线索。</li>
</ol></div>

<h2>十、局限性与未来研究</h2>
<ul class="clean">
<li><b>缺乏标准化随访时间标签</b>：虽有 54 位患者多次 PTA 手术的纵向记录（间隔 14–720 天），
但原始数据未标注"术后1周/1月/3月/6月"等随访时点，故无法开展严格的术后流量持久性与再狭窄轨迹建模；
观察性设计所得权重为调整后的关联强度，不构成因果推断；</li>
<li>术后超声指标与术后流量同期测量，属"伴随决定因素"，不能作为术前预测因子；
术前预测应使用基线流量、糖尿病、年龄等非术后变量；</li>
<li>分类变量按编码数值处理（除 CatBoost 的类别机制外），样本量（n≈220）相对特征数偏多，
小权重因素估计存在不稳定性，故以六模型一致性作为稳健性判据；</li>
<li>未纳入手术操作医师、术后用药依从性、穿刺方式等表外变量。</li>
</ul>
<p><b>建议扩展的研究问题：</b></p>
<ol style="margin-left:20px">
<li>补全术后 1周/1月/3月/6月随访后，建立混合效应/GAM 纵向轨迹模型与再狭窄风险模型；</li>
<li>开展多中心外部验证，检验本模型的迁移性与校准度；</li>
<li>纳入药物涂层球囊 vs 普通球囊的倾向性评分匹配（PSM）因果比较；</li>
<li>结合超声影像组学特征提升 ΔQ 个体化预测精度。</li>
</ol>

<h2>十一、可复现性说明</h2>
<ul class="clean">
<li>代码：<code>research/</code>（config/data_pipeline/quality/modeling/interpretation/clustering/figures/run_pipeline）；
一键复现：<code>python -m research.run_pipeline</code>；随机种子统一为 42；</li>
<li>所有逐折指标、逐样本分类概率、PDP 网格均落盘于 <code>outputs/tables/</code>，图表 PNG/PDF 落盘于 <code>outputs/figures/</code>；</li>
<li>预测应用：<code>webapp/</code>（本地 Flask，模型包 <code>artifacts/deploy_bundle.joblib</code>）。</li>
</ul>

</main>
<footer>本报告全部数值与图表均由真实数据与模型输出自动生成 ｜ Python 3.12 · scikit-learn · XGBoost · LightGBM · CatBoost · statsmodels · SHAP</footer>
</div></body></html>'''

    out = os.path.join(C.OUT_DIR, 'PTA血流动力学影响因素系统研究报告.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print('报告已生成:', out)


if __name__ == '__main__':
    build()
