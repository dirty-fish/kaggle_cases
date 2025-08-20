import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from src.data import load_data

TRAIN = "data/train.csv"
TEST = "data/test.csv"
TARGET = "target"
SEED = 42
N_SPLITS = 5

"""def _generate_synthetic(_train_path: str, _test_path: str, _target_col: str = "target"):
    rng = np.random.default_rng(42)
    n_train, n_test, n_features = 10000, 5000, 200
    _X_train = rng.normal(size = (n_train, n_features))
    _X_test = rng.normal(size = (n_test, n_features))

    _w = rng.normal(size = n_features)
    _logits = _X_train @ _w * 0.05
    _sigmoid = 1 / (1 + np.exp(-_logits))
    _y = (rng.uniform(size = n_train) < _sigmoid).astype(int)

    _df_train = pd.DataFrame(_X_train, columns = [f"var_{i}" for i in range(n_features)])
    _df_train[_target_col] = _y
    _df_test = pd.DataFrame(_X_test, columns = [f"var_{i}" for i in range(n_features)])

    Path(_train_path).parent.mkdir(parents = True, exist_ok = True)
    _df_train.to_csv(_train_path, index = False)
    _df_test.to_csv(_test_path, index = False)"""


"""def load_data(train_path: str, test_path: str, target_col):
    
    Загружает данные с диска или генерирует их, если файлов нет.

    Аргументы:
        train_path (str): путь к тренировочному датасету
        test_path (str): путь к тестовому датасету
        target_col (str): имя колонки с таргетом

    Возвращает:
        X (pd.DataFrame): тренировочные признаки
        y (pd.Series): целевая переменная
        X_test (pd.DataFrame): тестовые признаки
    
    if not Path(train_path).exists() or not Path(test_path).exists():
        _generate_synthetic(train_path, test_path, target_col)

    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)
    feats = [c for c in df_tr.columns if c != target_col]
    return df_tr[feats], df_tr[target_col], df_te[feats]"""


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
    X, y, Xtest = load_data(TRAIN, TARGET, TEST)
    """
    Возвращает:
        auc (float): ROC AUC по out-of-fold предсказаниям
    """
    _ = train_lgb_cv(X, y, Xtest)