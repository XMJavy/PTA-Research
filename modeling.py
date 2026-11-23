# -*- coding: utf-8 -*-
"""
6 模型对比建模(4 个目标变量)
  岭回归 / LASSO / 随机森林 / XGBoost / LightGBM / CatBoost
流程: 7:3 划分 → 网格搜索调参(3折) → 10折交叉验证 → 30%测试集评估
指标: R² / MAE / RMSE(+标准差) / 线性模型 AIC、BIC
另: 应答者(ΔQ>0)分类: ROC-AUC / 敏感度 / 特异度 / 混淆矩阵 (真实逐样本概率落盘)
"""
import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold,
                                     GroupKFold, GroupShuffleSplit,
                                     cross_validate, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, f1_score, accuracy_score)

from research import config as C

warnings.filterwarnings('ignore')
SEED = C.RANDOM_STATE


# ---------------- 模型工厂 ----------------
def _tree_grids():
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    return {
        '随机森林': (
            RandomForestRegressor(random_state=SEED, n_jobs=-1),
            {'n_estimators': [300, 600],
             'max_depth': [4, 8, None],
             'min_samples_leaf': [1, 2]},
            'raw'),
        'XGBoost': (
            xgb.XGBRegressor(random_state=SEED, n_jobs=-1,
                             objective='reg:squarederror', verbosity=0),
            {'n_estimators': [300, 600],
             'max_depth': [3, 5],
             'learning_rate': [0.03, 0.1]},
            'raw'),
        'LightGBM': (
            lgb.LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1),
            {'n_estimators': [300, 600],
             'max_depth': [3, 6, -1],
             'learning_rate': [0.03, 0.1],
             'num_leaves': [15, 31]},
            'raw'),
        'CatBoost': (
            CatBoostRegressor(random_seed=SEED, verbose=0,
                              allow_writing_files=False),
            {'iterations': [400, 800],
             'depth': [4, 6],
             'learning_rate': [0.03, 0.1]},
            'raw'),
    }


def _aic_bic(y_true, y_pred, k):
    n = len(y_true)
    sse = float(np.sum((y_true - y_pred) ** 2))
    if sse <= 0:
        return np.nan, np.nan
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    return round(aic, 2), round(bic, 2)


def bootstrap_test_r2_sd(y_test, pred_test, B=1000, seed=SEED):
    """测试集 R² 的 Bootstrap 标准差。

    测试集只评估一次，本身没有"折间标准差"；此处对测试集样本做 B 次有放回
    重抽样，每次重算 R²，取标准差作为该点估计的不确定性度量。
    所有模型共用同一套抽样索引，保证横向可比、结果可复现。
    """
    yt = np.asarray(y_test, dtype=float)
    n = len(yt)
    rng = np.random.default_rng(seed)
    idx_mat = rng.integers(0, n, size=(B, n))
    out = {}
    for name, p in pred_test.items():
        p = np.asarray(p, dtype=float)
        vals = [r2_score(yt[idx], p[idx]) for idx in idx_mat
                if np.std(yt[idx]) > 0]
        out[name] = round(float(np.std(vals, ddof=1)), 4)
    return out


def run_one_target(tgt, X, y, groups=None):
    """单个目标的完整训练/评估流程; groups=每组键(如患者序号), 则用 GroupKFold/GroupShuffleSplit 防泄露"""
    feats = X.columns.tolist()
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
        tr_idx, te_idx = next(gss.split(X.values, y.values, groups))
        X_train, X_test = X.values[tr_idx], X.values[te_idx]
        y_train, y_test = y.values[tr_idx], y.values[te_idx]
        groups_train = groups[tr_idx]
        groups_all = groups
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=0.30, random_state=SEED)
        groups_train = None
        groups_all = None

    # 训练/测试目标分布 K-S 检验
    ks = stats.ks_2samp(y_train, y_test)

    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

    result = dict(features=feats, target=tgt,
                  X=X, y=y, groups=groups,
                  X_train=X_train, X_test=X_test,
                  y_train=y_train, y_test=y_test,
                  scaler=scaler,
                  ks_stat=round(float(ks.statistic), 4), ks_p=round(float(ks.pvalue), 4),
                  models={}, fitted={}, inputs={}, best_params={},
                  metrics=[], cv_detail={}, pred_test={})

    # --- 线性模型(标准化输入) ---
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 60), cv=5)
    ridge.fit(Xtr_s, y_train)
    lasso = LassoCV(cv=5, max_iter=30000)
    lasso.fit(Xtr_s, y_train)
    linear = {'岭回归': ridge, 'LASSO': lasso}
    for name, m in linear.items():
        result['fitted'][name] = m
        result['inputs'][name] = 'scaled'

    # --- 树模型: 网格搜索(3折) ---
    for name, (est, grid, inp) in _tree_grids().items():
        gs = GridSearchCV(est, grid, cv=3, scoring='neg_root_mean_squared_error',
                          n_jobs=-1)
        gs.fit(X_train, y_train)
        result['fitted'][name] = gs.best_estimator_
        result['best_params'][name] = gs.best_params_
        result['inputs'][name] = inp

    # --- 10折交叉验证(全数据, 调参后模型; 分组→GroupKFold) ---
    if groups_all is not None:
        kf = GroupKFold(n_splits=10)
    else:
        kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    Xs_all = scaler.transform(X.values)
    for name, m in result['fitted'].items():
        scaled = result['inputs'][name] == 'scaled'
        Xin = Xs_all if scaled else X.values
        cvres = cross_validate(m, Xin, y.values, cv=kf,
                               groups=groups_all,
                               scoring={'r2': 'r2',
                                        'mae': 'neg_mean_absolute_error',
                                        'rmse': 'neg_root_mean_squared_error'},
                               n_jobs=-1)
        folds = pd.DataFrame({
            'fold': range(1, 11),
            'R2': cvres['test_r2'],
            'MAE': -cvres['test_mae'],
            'RMSE': -cvres['test_rmse'],
        })
        result['cv_detail'][name] = folds

    # --- 测试集评估 + 权重所需预测 ---
    for name, m in result['fitted'].items():
        scaled = result['inputs'][name] == 'scaled'
        Xin_tr = Xtr_s if scaled else X_train
        Xin_te = Xte_s if scaled else X_test
        p_tr = m.predict(Xin_tr)
        p_te = m.predict(Xin_te)
        result['pred_test'][name] = p_te
        row = dict(
            模型=name,
            CV_R2均值=round(float(np.mean(result['cv_detail'][name]['R2'])), 4),
            CV_R2标准差=round(float(np.std(result['cv_detail'][name]['R2'])), 4),
            CV_MAE均值=round(float(np.mean(result['cv_detail'][name]['MAE'])), 3),
            CV_RMSE均值=round(float(np.mean(result['cv_detail'][name]['RMSE'])), 3),
            CV_RMSE标准差=round(float(np.std(result['cv_detail'][name]['RMSE'])), 3),
            测试集_R2=round(r2_score(y_test, p_te), 4),
            测试集_MAE=round(mean_absolute_error(y_test, p_te), 3),
            测试集_RMSE=round(np.sqrt(mean_squared_error(y_test, p_te)), 3),
            参数数量=int(X.shape[1]),
        )
        if name in ('岭回归', 'LASSO'):
            aic, bic = _aic_bic(y_train, p_tr, X.shape[1] + 1)
            row['训练集_AIC'] = aic
            row['训练集_BIC'] = bic
        result['metrics'].append(row)

    result['metrics'] = pd.DataFrame(result['metrics'])
    boot_sd = bootstrap_test_r2_sd(y_test, result['pred_test'])
    result['metrics']['测试集_R2_SD'] = result['metrics']['模型'].map(boot_sd)
    best_row = result['metrics'].sort_values('CV_RMSE均值').iloc[0]
    result['best_name'] = best_row['模型']
    return result


# ---------------- 应答者分类(好转=ΔQ>0 且 ΔRI>0) ----------------
def run_classification(X, y_delta, y_dri=None, groups=None):
    """好转定义: 术后流量上升(ΔQ>0) 且 术后RI下降(ΔRI>0, 即术前-术后>0)"""
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    if y_dri is not None:
        responder = ((y_delta > 0) & (y_dri > 0)).astype(int).values
    else:
        responder = (y_delta > 0).astype(int).values
    d = pd.DataFrame({'x_idx': X.index, 'y': responder})
    y_cls = d['y'].values
    prevalence = float(y_cls.mean())

    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
        tr_idx, te_idx = next(gss.split(X.values, y_cls, groups))
        X_train, X_test = X.values[tr_idx], X.values[te_idx]
        y_train, y_test = y_cls[tr_idx], y_cls[te_idx]
        groups_all = groups
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_cls, test_size=0.30, random_state=SEED, stratify=y_cls)
        groups_all = None
    scaler = StandardScaler().fit(X_train)

    clfs = {
        '逻辑回归': (LogisticRegression(max_iter=2000, C=1.0), True),
        '随机森林': (RandomForestClassifier(n_estimators=500, max_depth=6,
                                       min_samples_leaf=2, random_state=SEED,
                                       n_jobs=-1), False),
        'XGBoost': (xgb.XGBClassifier(n_estimators=400, max_depth=3,
                                  learning_rate=0.05, random_state=SEED,
                                  n_jobs=-1, eval_metric='logloss',
                                  verbosity=0), False),
        'LightGBM': (lgb.LGBMClassifier(n_estimators=400, max_depth=4,
                                    learning_rate=0.05, num_leaves=15,
                                    random_state=SEED,
                                    verbose=-1, n_jobs=-1), False),
        'CatBoost': (CatBoostClassifier(iterations=500, depth=4,
                                    learning_rate=0.05, random_seed=SEED,
                                    verbose=0, allow_writing_files=False), False),
    }

    if groups_all is not None:
        skf = GroupKFold(n_splits=10)
        _group_arg = dict(groups=groups_all)
    else:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        _group_arg = {}
    rows, curves = [], {}
    for name, (clf, scaled) in clfs.items():
        Xin = X.values
        # OOF 概率
        oof = np.zeros(len(y_cls))
        for tr, va in skf.split(Xin, y_cls, **_group_arg):
            sc = StandardScaler().fit(Xin[tr])
            xtr = sc.transform(Xin[tr]) if scaled else Xin[tr]
            xva = sc.transform(Xin[va]) if scaled else Xin[va]
            clf.fit(xtr, y_cls[tr])
            oof[va] = clf.predict_proba(xva)[:, 1]
        auc = roc_auc_score(y_cls, oof)
        fpr, tpr, thr = roc_curve(y_cls, oof)
        j = np.argmax(tpr - fpr)
        cut = float(thr[j])
        pred = (oof >= cut).astype(int)
        cm = confusion_matrix(y_cls, pred)
        curves[name] = dict(fpr=fpr, tpr=tpr, auc=auc,
                            y_true=y_cls, y_score=oof)
        rows.append(dict(
            模型=name, 交叉验证AUC=round(auc, 4),
            Youden截断值=round(cut, 3),
            敏感度=round(recall_score(y_cls, pred), 4),
            特异度=round((cm[0, 0] / cm[0].sum()), 4),
            准确率=round(accuracy_score(y_cls, pred), 4),
            F1=round(f1_score(y_cls, pred), 4),
            阴性数=int((y_cls == 0).sum()), 阳性数=int((y_cls == 1).sum()),
        ))
    metrics = pd.DataFrame(rows).sort_values('交叉验证AUC', ascending=False)
    best = metrics.iloc[0]['模型']
    return dict(metrics=metrics, curves=curves, best=best,
                prevalence=prevalence,
                confusion_at_youden=confusion_matrix(
                    y_cls, (curves[best]['y_score'] >=
                            metrics.loc[metrics['模型'] == best,
                                        'Youden截断值'].iloc[0]).astype(int)))


# ---------------- 主入口 ----------------
def run_all(matrices, df=None):
    bundle = {'targets': {}}
    all_metric_tabs = {}
    for tgt, (X, y, g) in matrices.items():
        print(f'  建模: {C.short(tgt)}  (n={len(y)}, p={X.shape[1]}'
              + (f', groups={len(set(g))}' if g is not None else '') + ') ...',
              flush=True)
        r = run_one_target(tgt, X, y, groups=g)
        bundle['targets'][tgt] = r
        all_metric_tabs[C.short(tgt)] = r['metrics']
        print(f'    最优模型: {r["best_name"]}  '
              f'CV-R2={r["metrics"]["CV_R2均值"].max():.3f}  '
              f'测试RMSE={r["metrics"].set_index("模型").loc[r["best_name"],"测试集_RMSE"]:.1f}',
              flush=True)

    # 应答者分类(好转=ΔQ>0 且 ΔRI>0, 基于ΔQ同一特征矩阵)
    Xd, yd, gd = matrices[C.T_DFLOW]
    ydri = df.loc[Xd.index, C.T_DRI] if df is not None and C.T_DRI in df.columns else None
    print('  应答者分类(好转=ΔQ>0 且 ΔRI>0) ...', flush=True)
    cls_res = run_classification(Xd, yd, y_dri=ydri, groups=gd)
    bundle['classification'] = cls_res
    print(f'    最佳: {cls_res["best"]}  AUC={cls_res["metrics"]["交叉验证AUC"].max():.3f}  '
          f'好转率={cls_res["prevalence"]*100:.1f}%',
          flush=True)

    # 指标总表落盘
    path = os.path.join(C.TAB_DIR, '模型性能对比.xlsx')
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        for name, tab in all_metric_tabs.items():
            tab.to_excel(w, sheet_name=name[:28], index=False)
        cls_res['metrics'].to_excel(w, sheet_name='应答者分类', index=False)
        # 逐折记录
        for tgt, r in bundle['targets'].items():
            for mname, folds in r['cv_detail'].items():
                folds.to_excel(w, sheet_name=f'{C.short(tgt)[:10]}-{mname}'[:31],
                               index=False)
    joblib.dump(bundle, os.path.join(C.TAB_DIR, 'model_bundle_raw.joblib'))
    print('  模型性能表与中间结果已保存')
    return bundle


if __name__ == '__main__':
    from research.quality import assess
    df, tables, matrices, summary = assess()
    bundle = run_all(matrices, df=df)
    print('DONE')
