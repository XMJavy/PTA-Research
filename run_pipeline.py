# -*- coding: utf-8 -*-
"""
一键运行完整研究流水线:
  数据质量 → 6模型建模 → 模型解释 → 聚类 → 全部图表 → 部署用模型包
用法: python -m research.run_pipeline
"""
import os, json
import numpy as np
import pandas as pd
import joblib

from research import config as C
from research.quality import assess
from research.modeling import run_all
from research.interpretation import run_interpretation
from research.clustering import run_clustering
from research import figures
from research import dq_analysis


def build_deploy_bundle(df, bundle, interp):
    """生成 webapp 使用的轻量模型包(每目标: 最佳模型+标准化器+特征+元数据)"""
    deploy = {'targets': {}}
    for tgt, r in bundle['targets'].items():
        best = r['best_name']
        scaled = r['inputs'][best] == 'scaled'
        # 中位数(供应用缺失填补)
        medians = df[r['features']].median(numeric_only=True)
        deploy['targets'][tgt] = {
            'target_name': C.short(tgt),
            'best_model': best,
            'needs_scaling': scaled,
            'features': r['features'],
            'feature_short': [C.short(f) for f in r['features']],
            'model': r['fitted'][best],
            'scaler': r['scaler'] if scaled else None,
            'medians': medians,
            'y_mean': float(r['y'].mean()), 'y_std': float(r['y'].std()),
            'y_median': float(r['y'].median()),
            'cv_r2': float(r['metrics'].set_index('模型').loc[best, 'CV_R2均值']),
            'cv_rmse': float(r['metrics'].set_index('模型').loc[best, 'CV_RMSE均值']),
            'test_r2': float(r['metrics'].set_index('模型').loc[best, '测试集_R2']),
            'top10': interp[tgt]['detail']['列名'].head(10).tolist(),
            'weights': interp[tgt]['detail'].set_index('列名')['综合权重(%)'].to_dict(),
            'pdp': {f: {k: v.tolist() for k, v in d.items()}
                    for f, d in interp[tgt]['pdp'].items()},
        }
    path = os.path.join(C.ARTIFACT_DIR, 'deploy_bundle.joblib')
    joblib.dump(deploy, path)
    print(f'部署模型包已保存: {path}')
    return deploy


def main():
    print('[1/6] 数据质量评估 ...', flush=True)
    df, q_tables, matrices, summary = assess()
    print('      ', summary)

    print('[2/6] 六模型对比建模 ...', flush=True)
    bundle = run_all(matrices, df=df)

    print('[3/6] 模型解释(权重/CI/SHAP/PDP) ...', flush=True)
    interp = run_interpretation(bundle)

    print('[4/6] 聚类分析 ...', flush=True)
    fc, pc = run_clustering(bundle, df)

    print('[5/6] 生成图表与部署包 ...', flush=True)
    figures.render_all(df, q_tables, bundle, interp, fc, pc)
    deploy = build_deploy_bundle(df, bundle, interp)

    print('[6/6] ΔQ 两步影响因素分析(单因素筛选+多因素回归) ...', flush=True)
    dq_two_stage = dq_analysis.run_analysis(make_figs=True)

    joblib.dump(dict(df=df, q_tables=q_tables, bundle=bundle, interp=interp,
                     factor_cluster=fc, patient_cluster=pc, summary=summary,
                     dq_two_stage=dq_two_stage),
                os.path.join(C.TAB_DIR, 'full_results.joblib'))
    print('流水线全部完成 ✓')


if __name__ == '__main__':
    main()
