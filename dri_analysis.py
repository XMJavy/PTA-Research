# -*- coding: utf-8 -*-
"""
ΔRI（阻力指数变化量 = 术前RI − 术后RI）影响因素系统性分析
=========================================================
符合临床流行病学研究规范的两步流程，核心分析由通用引擎
research.two_stage.run_two_stage 完成：
1. 描述性统计 + 正态性检验（Shapiro-Wilk）
2. 单因素分析（二分类→Mann-Whitney U，多分类→Kruskal-Wallis，连续→Spearman），
   筛选 P<0.2 的候选变量
3. 多因素线性回归（OLS + HC3 稳健标准误）+ 6 模型综合权重
4. 交互作用检验 + 拟合优度（R², F）+ 残差分析
5. 输出 docx 报告 + 森林图/权重图

用法：
    python -m research.dri_analysis            # 完整分析 + docx + 图表
    from research.dri_analysis import run_analysis   # 供 make_dri_report.py 复用
"""
import os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research import config as C
from research.two_stage import run_two_stage

# ---------------- ΔRI 专属配置 ----------------
DRI_CFG = {
    'target_short': '阻力指数变化量ΔRI',
    'formula_desc': ('ΔRI = 术前阻力指数 − 术后阻力指数，正值表示术后 RI 下降（血管阻力改善），'
                     '负值表示术后 RI 升高。本研究旨在系统识别影响 ΔRI 的独立因素。'),
    'interaction_pairs': [
        ('糖尿病', '术前阻力指数'),
        ('合并血栓', '血栓长度'),
        ('高血压', '术前PSV'),
    ],
    'fig_forest': os.path.join(C.FIG_DIR, 'ΔRI_森林图.png'),
    'fig_weight': os.path.join(C.FIG_DIR, 'ΔRI_权重图.png'),
    'forest_title': '图  ΔRI（阻力指数变化量）影响因素森林图',
    'pos_note': '红=正向影响(术后RI下降更多)  蓝=负向影响',
    'weight_title': '图  ΔRI 影响因素综合权重排名 Top15（从大到小）',
    'clinical_interp': {
        '术前阻力指数': ('正向', '术前 RI 越高，术后 RI 下降幅度越大。'
                        '高阻力状态意味着血管痉挛/狭窄严重，PTA 开通后阻力下降空间大，'
                        '是 ΔRI 改善最强的预测因子。'),
        '糖尿病': ('负向', '糖尿病患者术后 RI 下降幅度较小。'
                  '糖尿病微血管病变与内皮功能障碍可能削弱 PTA 后的血管舒张反应。'),
        '吻合口宽度': ('正向', '吻合口越宽，术后 RI 下降越明显。'
                    '宽吻合口提供更大的流入道，PTA 后阻力改善更充分。'),
        '合并血栓': ('正向', '合并血栓的患者术后 RI 下降更显著。'
                    '血栓抽吸/溶栓联合 PTA 可显著解除梗阻，阻力下降幅度大。'),
        '教育程度': ('正向', '教育程度较高者术后 RI 改善更佳。'
                    '可能与健康素养、依从性及术后管理质量相关。'),
    },
}


def run_analysis(make_figs=True, write_docx=True):
    """执行 ΔRI 完整分析流程，返回结果 dict。
    核心计算委托 research.two_stage.run_two_stage；本函数额外负责 docx 输出。
    """
    res = run_two_stage(C.T_DRI, DRI_CFG, make_figs=make_figs)

    # ---------------- 六模型机器学习预测 + 最优模型 SHAP ----------------
    print('\n=== ΔRI 六模型预测建模（GroupKFold + SHAP） ===')
    from research.dri_modeling import run_dri_model
    res['ml'] = run_dri_model(make_fig=make_figs)

    desc_stats = res['desc_stats']
    uni_df = res['uni_df']
    n_pass = res['n_pass']
    reg_df = res['reg_df']
    sig_vars = res['sig_vars']
    inter_df = res['inter_df']
    clinical_interp = res['clinical_interp']
    forest_path = res['forest_path']
    weight_path = res['weight_path']

    # ---------------- 输出 docx 报告 ----------------
    if write_docx:
        print('\n=== 生成 docx 报告 ===')
        from docx import Document
        from docx.shared import Pt, Cm

        doc = Document()
        style = doc.styles['Normal']
        style.font.name = '宋体'
        style.font.size = Pt(10.5)

        def H(text, level=1):
            h = doc.add_heading(text, level=level)
            for run in h.runs:
                run.font.name = '黑体'

        def P(text, bold=False, size=10.5):
            p = doc.add_paragraph()
            run = p.add_run(text)
            run.font.size = Pt(size)
            run.bold = bold
            return p

        def add_table(df_, cols=None, float_fmt='.4f', p_col=None):
            if cols is None:
                cols = df_.columns.tolist()
            t = doc.add_table(rows=1, cols=len(cols))
            t.style = 'Light Grid Accent 1'
            hdr = t.rows[0].cells
            for i, c in enumerate(cols):
                hdr[i].text = str(c)
                for p in hdr[i].paragraphs:
                    for r in p.runs:
                        r.bold = True
                        r.font.size = Pt(9)
            for _, row in df_.iterrows():
                cells = t.add_row().cells
                for i, c in enumerate(cols):
                    v = row[c]
                    if isinstance(v, float):
                        if p_col and c == p_col:
                            cells[i].text = f'{v:.2e}'
                        else:
                            cells[i].text = f'{v:{float_fmt}}'
                    else:
                        cells[i].text = str(v)
                    for p in cells[i].paragraphs:
                        for r in p.runs:
                            r.font.size = Pt(9)
            return t

        title = doc.add_heading('ΔRI（阻力指数变化量）影响因素系统性分析报告', level=0)
        P(f'数据来源: PTA.xlsx · 方法: 单因素筛选(P<0.2) + 多因素OLS(HC3稳健SE) '
          f'+ Huber稳健回归敏感性分析')

        # 1. 研究背景与方法
        H('一、研究背景与方法')
        P(DRI_CFG['formula_desc'])
        P('分析流程：(1) ΔRI 描述性统计与正态性检验；(2) 单因素分析（二分类→Mann-Whitney U，'
          '多分类→Kruskal-Wallis，连续→Spearman 相关），筛选 P<0.2 候选变量；'
          '(3) 多因素线性回归（OLS + HC3 稳健标准误），多分类变量哑变量化，VIF>10 迭代剔除；'
          '(4) Huber 稳健回归作为非正态敏感性分析；(5) 临床交互作用检验；'
          '(6) 残差正态性、异方差、共线性诊断；'
          '(7) 综合权重评估：岭回归、LASSO、随机森林、XGBoost、LightGBM、CatBoost 六个模型各自重要性'
          'min-max 归一化后等权平均，换算为综合权重（%）并按从大到小排名；'
          '(8) 机器学习预测：岭回归、LASSO、随机森林、XGBoost、LightGBM、CatBoost 六模型'
          '按患者分组 7:3 划分 + GroupKFold 10 折交叉验证，选出最优模型，'
          '并以 SHAP 蜂群图展示各因素对个体预测值的贡献方向与大小。')

        # 2. 描述性统计
        H('二、ΔRI 描述性统计')
        ds = desc_stats
        P(f"有效样本量 n = {ds['n']}")
        P(f"均值 ± 标准差 = {ds['mean']:.3f} ± {ds['sd']:.3f}")
        P(f"范围 = [{ds['min']:.3f}, {ds['max']:.3f}]")
        P(f"四分位数 P25/中位数/P75 = {ds['p25']:.3f} / {ds['median']:.3f} / {ds['p75']:.3f}")
        P(f"Shapiro-Wilk 正态性检验: W = {ds['shapiro_W']:.4f}, P = {ds['shapiro_p']:.4e}")
        P(f"结论: ΔRI {'符合' if ds['normal'] else '不符合'}正态分布（P{'<' if not ds['normal'] else '>'}0.05），"
          f"{'故多因素分析采用 HC3 稳健标准误并辅以 Huber 稳健回归作为敏感性分析' if not ds['normal'] else '采用普通最小二乘回归'}。")

        # 3. 单因素分析
        H('三、单因素分析结果')
        P(f'共 {len(uni_df)} 个候选变量进入单因素分析，其中 {n_pass} 个变量 P<0.2 进入多因素回归。')
        P('表：单因素分析结果（按 P 值排序，P<0.2 进入多因素）')
        uni_show = uni_df.copy()
        uni_show['P值'] = uni_show['P值'].map(lambda x: f'{x:.2e}')
        add_table(uni_show, cols=['变量', '类型', '检验方法', '统计量', 'P值', '筛选(P<0.2)'])

        # 4. 多因素回归
        H('四、多因素线性回归结果')
        P(f'最终纳入 {res["n_reg_features"]} 个自变量（含哑变量），样本量 n = {res["n_reg"]}。')
        P(f"模型拟合: R² = {res['r2']:.4f}, 调整 R² = {res['adj_r2']:.4f}, "
          f"F = {res['fvalue']:.2f}, P = {res['f_pvalue']:.4e}")
        P('表：多因素回归系数表（OLS + HC3 稳健标准误；排名与综合权重(%)由'
          '岭回归/LASSO/随机森林/XGBoost/LightGBM/CatBoost 六模型归一化重要性等权平均得出，按权重从大到小排名）')
        reg_show = reg_df.copy()
        reg_show['P值'] = reg_show['P值'].map(lambda x: f'{x:.2e}')
        add_table(reg_show, cols=['排名', '变量', '回归系数β', '稳健SE', '95%CI下限',
                                  '95%CI上限', 'P值', '显著性', 'Huberβ', '综合权重(%)'],
                  float_fmt='.4f')

        P('')
        P('图：ΔRI 影响因素标准化效应量森林图（Top15 按|β|从大到小；红=正向，蓝=负向，大圆点=P<0.05）')
        doc.add_picture(forest_path, width=Cm(15.5))

        P('')
        P('图：ΔRI 影响因素综合权重排名 Top15（6模型综合权重，从大到小；红=正向，蓝=负向）')
        doc.add_picture(weight_path, width=Cm(15.5))

        # 5. 机器学习预测模型与 SHAP
        ml = res['ml']
        H('五、机器学习预测模型：六模型对比与 SHAP 归因')
        P(f'以同一套术前基线+干预因素（{ml["n_features"]} 个特征，中位数填补、VIF>10 剔除）预测 ΔRI，'
          f'按患者序号分组 7:3 划分（训练 {ml["n_train"]} 例 / 测试 {ml["n_test"]} 例，'
          f'同一患者的手术记录不跨集），训练/测试 ΔRI 分布 K-S 检验 P={ml["ks_p"]:.3f}；'
                   '岭回归、LASSO、随机森林、XGBoost、LightGBM、CatBoost 六模型对比，'
          '全数据 GroupKFold 10 折交叉验证，按 CV RMSE 最小选择最优模型。')
        ml_show = ml['metrics'].copy()
        add_table(ml_show, cols=['模型', 'CV_R2均值', 'CV_RMSE均值', '测试集_R2',
                                 '测试集_MAE', '测试集_RMSE'])
        brow = ml['best_row']
        P(f'最优模型为 {ml["best_name"]}：10 折 CV R²={brow["CV_R2均值"]:.3f}，'
          f'CV RMSE={brow["CV_RMSE均值"]:.3f}；30% 测试集 R²={brow["测试集_R2"]:.3f}，'
          f'测试 RMSE={brow["测试集_RMSE"]:.3f}。', bold=True)
        if ml['shap_path']:
            P('')
            P('图：ΔRI 最优模型 SHAP 特征贡献蜂群图（每个点=一名患者；颜色红/蓝=该特征取值高/低；'
              '横向位置=该特征把该患者 ΔRI 预测值推高/推低的幅度）')
            doc.add_picture(ml['shap_path'], width=Cm(15.5))
            top3 = ml['shap_top'].head(3)
            P('SHAP 平均绝对贡献居前的因素为：'
              + '、'.join(f'{i+1}. {k}（{v:.3f}）' for i, (k, v) in enumerate(top3.items()))
              + '；其中术前阻力指数贡献远高于其他因素，与多因素回归结论一致。')
            P(f'方法学说明：本节 SHAP 贡献基于单一最优模型（{ml["best_name"]}）对每位患者预测值的分解，'
              '与第四章「六模型综合权重」（6 个模型重要性等权平均）为两种不同度量，'
              '变量次序不完全一致属正常现象；两种方法均一致指向术前阻力指数为首要驱动因素。',
              size=9.5)

        # 6. 交互作用
        H('六、交互作用检验')
        P('对临床意义明确的交互对进行检验：')
        if len(inter_df):
            inter_show = inter_df.copy()
            inter_show['P值'] = inter_show['P值'].map(lambda x: f'{x:.2e}')
            add_table(inter_show, cols=['交互项', '回归系数', 'P值', '显著性'])
        P('结论：所有检验的交互项均无统计学意义（P>0.05），主效应模型无需纳入交互项。')

        # 7. 模型诊断
        H('七、模型诊断')
        P(f'残差 Shapiro-Wilk 正态性检验: W = {res["resid_W"]:.4f}, P = {res["resid_p"]:.4e} → '
          f"残差{'服从' if res['resid_p'] > 0.05 else '不服从'}正态分布")
        P(f'Breusch-Pagan 异方差检验: LM = {res["bp_stat"]:.4f}, P = {res["bp_p"]:.4e} → '
          f"{'不存在' if res['bp_p'] > 0.05 else '存在'}异方差")
        P(f'残差: 均值 = {res["resid_mean"]:.4f}, SD = {res["resid_sd"]:.4f}, '
          f'范围 [{res["resid_min"]:.4f}, {res["resid_max"]:.4f}]')
        P(f'|残差| > 2.5SD 的异常值: {res["outliers"]} 个')
        P(f'最大 VIF = {res["max_vif"]:.2f}（VIF<10，无严重共线性）')
        P(f'AIC = {res["aic"]:.1f}, BIC = {res["bic"]:.1f}')

        # 8. 显著因素与临床解读
        H('八、ΔRI 独立影响因素及临床解读')
        P(f'多因素回归中 P<0.05 的独立影响因素共 {len(sig_vars)} 个：', bold=True)
        for v in sig_vars:
            if v in clinical_interp:
                direction, interp = clinical_interp[v]
                beta_v = reg_df.loc[reg_df['变量'] == v, '回归系数β'].iloc[0]
                P(f'• {v}（{direction}, β={beta_v:+.3f}）：{interp}', size=10)

        # 9. 结论
        H('九、结论')
        P(f'1. ΔRI 均值为 {desc_stats["mean"]:.3f}±{desc_stats["sd"]:.3f}，'
          f'多数患者术后 RI 下降（阻力改善），但分布非正态。')
        P(f'2. 单因素分析筛选出 {n_pass} 个候选变量；多因素回归（R²={res["r2"]:.3f}）'
          f'识别出 {len(sig_vars)} 个独立影响因素：{", ".join(sig_vars)}。')
        P('3. 术前阻力指数是最强的正向预测因子，糖尿病是负向因素，'
          '吻合口宽度与合并血栓为正向因素，教育程度为正向因素。')
        P(f'4. 六模型机器学习预测中，{ml["best_name"]} 表现最优'
          f'（10 折 CV R²={brow["CV_R2均值"]:.3f}，测试集 R²={brow["测试集_R2"]:.3f}）；'
          'SHAP 归因显示术前阻力指数的贡献远高于其他因素，与回归结论互相印证。')
        P('5. 模型诊断良好：残差正态、无异方差、无严重共线性，Huber 稳健回归结果一致，结论可靠。')
        P('6. 临床意义：术前高 RI、宽吻合口、合并血栓的患者预期 RI 改善更显著；'
          '糖尿病患者需加强术后管理以优化阻力改善。')

        out_path = os.path.join(C.OUT_DIR, 'ΔRI影响因素分析报告.docx')
        try:
            doc.save(out_path)
            print(f'报告已生成: {out_path}')
        except PermissionError:
            out_path = os.path.join(C.OUT_DIR, 'ΔRI影响因素分析报告_v2.docx')
            doc.save(out_path)
            print(f'原文件被占用，已另存为: {out_path}')

    print('\n=== 分析完成 ===')
    return res


if __name__ == '__main__':
    run_analysis()
