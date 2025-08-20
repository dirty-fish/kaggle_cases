import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

TRAIN = "data/train.csv"
TEST = "data/test.csv"
TARGET = "target"
SEED = 42
N_SPLITS = 5

def _generate_synthetic(_train_path: str, _test_path: str, _target_col: str = "target")
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
    _df_test = pd.DataFrame(_X_train, columns = [f"var_{i}" for i in range(n_features)])

    Path(_train_path).parent.mkdir(parents = True, exist_ok = True)
    _df_train.to_csv(_train_path, index = False)
    _df_test.to_csv(_train_path, index = False)


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