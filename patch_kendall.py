# -*- coding: utf-8 -*-
"""修正 Kendall's W 计算(原实现误用 friedmanchisquare 导致恒≈0),
从已保存的 interpretation.joblib 重算 W/χ²/p/两两τ, 更新结果表与报告。"""
import os
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

from research import config as C

MODEL_ORDER = ['岭回归', 'LASSO', '随机森林', 'XGBoost', 'LightGBM', 'CatBoost']


def recompute_w(detail):
    feats = detail['列名'].tolist()
    n_it = len(feats)
    src = pd.DataFrame(index=feats)
    src['岭回归'] = detail['Ridge标准化β'].abs().values
    src['LASSO'] = detail['LASSO标准化β'].abs().values
    for name in ['随机森林', 'XGBoost', 'LightGBM', 'CatBoost']:
        src[name] = detail[f'{name}重要性'].values
    norm = src.apply(lambda v: (v - v.min()) / (v.max() - v.min())
                     if v.max() > v.min() else v * 0)
    k = 6
    ranks = np.column_stack(
        [rankdata(-norm[m].values, method='average') for m in MODEL_ORDER])
    R_i = ranks.sum(axis=1)
    S = np.sum((R_i - k * (n_it + 1) / 2) ** 2)
    T_total = 0.0
    for j in range(k):
        _, counts = np.unique(ranks[:, j], return_counts=True)
        tie = counts[counts > 1]
        T_total += np.sum(tie ** 3 - tie) / 12.0
    denom = (k ** 2 * (n_it ** 3 - n_it)) / 12.0 - k * T_total
    W = float(S / denom) if denom > 0 else 0.0
    chi2 = k * (n_it - 1) * W
    p = float(stats.chi2.sf(chi2, df=n_it - 1))
    taus = {}
    for i in range(k):
        for j in range(i + 1, k):
            tau, _ = stats.kendalltau(ranks[:, i], ranks[:, j])
            taus[f'{MODEL_ORDER[i]} vs {MODEL_ORDER[j]}'] = round(float(tau), 4)
    return W, float(chi2), p, taus


def main():
    path = os.path.join(C.TAB_DIR, 'interpretation.joblib')
    blob = joblib.load(path)
    interp = blob['interpretation']
    rows = []
    for t, out in interp.items():
        W, chi2, p, taus = recompute_w(out['detail'])
        out['kendall_w'], out['kendall_chi2'], out['kendall_p'], out['kendall_tau'] = W, chi2, p, taus
        out['detail'].to_excel(
            os.path.join(C.TAB_DIR, f'权重明细_{C.short(t)}.xlsx'), index=False)
        rows.append(dict(目标=C.short(t), Kendall_W=round(W, 4),
                         卡方=round(chi2, 2), P值=('%.2e' % p),
                         结论='排序一致性强' if W > 0.5
                         else ('中等一致' if W > 0.3 else '一致性弱')))
        print(f'{C.short(t)}: W={W:.4f}  χ²={chi2:.2f}  p={p:.3g}  '
              f'τ范围[{min(taus.values()):.3f},{max(taus.values()):.3f}]')

    # 聚合表(与 run_interpretation 输出格式一致)
    first = list(interp.keys())[0]
    tau_tab = pd.DataFrame([dict(模型对=k, Kendall_tau=v)
                            for k, v in interp[first]['kendall_tau'].items()])
    with pd.ExcelWriter(os.path.join(C.TAB_DIR, '排序一致性与权重汇总.xlsx'),
                        engine='openpyxl') as w:
        pd.DataFrame(rows).to_excel(w, sheet_name='Kendall_W', index=False)
        tau_tab.to_excel(w, sheet_name='两两Kendall_tau', index=False)
        for t, out in interp.items():
            out['detail'].head(25).to_excel(
                w, sheet_name=f'Top25_{C.short(t)}'[:28], index=False)
    joblib.dump(blob, path)

    # 同步 full_results.joblib 并重生成报告
    fr_path = os.path.join(C.TAB_DIR, 'full_results.joblib')
    if os.path.exists(fr_path):
        fr = joblib.load(fr_path)
        fr['interp'] = interp
        joblib.dump(fr, fr_path)
        from research.make_report import build as build_report
        build_report()
        print('报告已更新')


if __name__ == '__main__':
    main()
