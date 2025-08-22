import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from src.data import load_data

TRAIN = "./data/train.csv"
TEST = "./data/test.csv"
TARGET = "target"
SEED = 42
N_SPLITS = 5

def train_lgb_cv(X, y, X_test):
    """
    Обучает модель LightGBM с кросс-валидацией и возвращает AUC по OOF-прогнозам.

    Аргументы:
        X (pd.DataFrame): тренировочные признаки
        y (pd.Series): целевая переменная
        X_test (pd.DataFrame): тестовые признаки

    Возвращает:
        auc (float): ROC AUC по out-of-fold предсказаниям
    """
    kf = StratifiedKFold(n_splits = N_SPLITS, shuffle = True, random_state = SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    params = dict(
        objective="binary", metric="auc", boosting_type="gbdt",
        learning_rate=0.03, num_leaves=31,
        feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5,
        seed=SEED, verbose=-1
    )

    for fold, (train_index, valid_index) in enumerate(kf.split(X,y), 1):
        print(f"[CV] Fold {fold}/{N_SPLITS}")
        X_tr, X_va = X.iloc[train_index], X.iloc[valid_index]
        y_tr, y_va = y.iloc[train_index], y.iloc[valid_index]

        ds_train = lgb.Dataset(X_tr, y_tr)
        ds_valid = lgb.Dataset(X_va, y_va, reference = ds_train)
    
        model = lgb.train(
            params, ds_train, valid_sets=[ds_train, ds_valid], valid_names=["train", "valid"],
            num_boost_round=2000, callbacks = [lgb.early_stopping(200), lgb.log_evaluation(200)]
        )

        oof[valid_index] = model.predict(X_va, num_iteration = model.best_iteration)
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / N_SPLITS


    auc = roc_auc_score(y, oof)
    print(f"OOF AUC: {auc:.5f}")
    return auc


if __name__ =="__main__":
    """
    Возвращает:
        X (pd.DataFrame): тренировочные признаки
        y (pd.Series): целевая переменная
        X_test (pd.DataFrame): тестовые признаки
    """
    X, y, Xtest = load_data(TRAIN, TEST, TARGET)
    """
    Возвращает:
        auc (float): ROC AUC по out-of-fold предсказаниям
    """
    _ = train_lgb_cv(X, y, Xtest)