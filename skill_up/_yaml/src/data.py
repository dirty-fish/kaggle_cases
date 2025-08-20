import pandas as pd
import numpy as np
from pathlib import Path

def _generate_synthetic(_train_path: str, _test_path: str, _target_col: str = "target"):
    rng = np.random.default_rng(42)
    n_train, n_test, n_features = 10000, 5000, 200
    _X_train = rng.normal(size = (n_train, n_features))
    _X_test = rng.normal(size = (n_test, n_features))

    _weight = rng.normal(size = n_features)
    _logits = _X_train @ _weight * 0.05
    _sigmoid = 1 / (1 + np.exp(-_logits))
    _y = (rng.uniform(size = n_train) < _sigmoid).astype(int)

    _df_train = pd.DataFrame(_X_train, columns = [f"var_{i}" for i in range(n_features)])
    _df_train[_target_col] = _y
    _df_test = pd.DataFrame(_X_test, columns = [f"var_{i}" for i in range(n_features)])

    Path(_train_path).parent.mkdir(parents = True, exist_ok = True)
    _df_train.to_csv(_train_path, index = False)
    _df_test.to_csv(_test_path, index = False)

def load_data(train_path: str, test_path: str, target_col):
    """
    Загружает данные с диска или генерирует их, если файлов нет.

    Аргументы:
        train_path (str): путь к тренировочному датасету
        test_path (str): путь к тестовому датасету
        target_col (str): имя колонки с таргетом

    Возвращает:
        X (pd.DataFrame): тренировочные признаки
        y (pd.Series): целевая переменная
        X_test (pd.DataFrame): тестовые признаки
    """
    if not Path(train_path).exists() or not Path(test_path).exists():
        _generate_synthetic(train_path, test_path, target_col)

    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)
    feats = [c for c in df_tr.columns if c != target_col]
    return df_tr[feats], df_tr[target_col], df_te[feats]