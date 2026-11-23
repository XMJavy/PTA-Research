# -*- coding: utf-8 -*-
"""
断点续跑: 建模(model_bundle_raw.joblib)与解释(interpretation.joblib)已完成后,
依次执行 聚类 → 图表 → 部署包 → HTML报告
用法: python -m research.resume_after_modeling
"""
import os
import joblib

from research import config as C
from research.data_pipeline import build_clean_dataset
from research.clustering import run_clustering
from research import figures
from research.run_pipeline import build_deploy_bundle
from research.make_report import build as build_report


def main():
    print('[续跑] 载入模型与解释结果 ...', flush=True)
    bundle = joblib.load(os.path.join(C.TAB_DIR, 'model_bundle_raw.joblib'))
    interp = joblib.load(os.path.join(C.TAB_DIR, 'interpretation.joblib'))['interpretation']
    df, audit = build_clean_dataset()

    print('[1/4] 聚类分析 ...', flush=True)
    fc, pc = run_clustering(bundle, df)

    print('[2/4] 生成全部图表 ...', flush=True)
    # 质量表直接读已落盘文件
    import pandas as pd
    qpath = os.path.join(C.TAB_DIR, '数据质量评估.xlsx')
    q_tables = pd.read_excel(qpath, sheet_name=None)
    figures.render_all(df, q_tables, bundle, interp, fc, pc)

    print('[3/4] 生成网页应用部署模型包 ...', flush=True)
    build_deploy_bundle(df, bundle, interp)

    print('[4/4] 生成 HTML 研究报告 ...', flush=True)
    out_tab = q_tables['02_异常值检测']
    summary = dict(
        n=len(df),
        n_features_raw=83,
        n_clinical_outlier_vals=int(out_tab['临床越界数'].sum()),
        n_clinical_outlier_cols=int((out_tab['临床越界数'] > 0).sum()),
        target_missing={})
    joblib.dump(dict(df=df, q_tables=q_tables, bundle=bundle, interp=interp,
                     factor_cluster=fc, patient_cluster=pc, summary=summary),
                os.path.join(C.TAB_DIR, 'full_results.joblib'))
    build_report()
    print('续跑全部完成 ✓')


if __name__ == '__main__':
    main()
