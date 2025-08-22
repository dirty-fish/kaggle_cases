import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RowStats(BaseEstimator, TransformerMixin):
    """
    Кастомный трансформер для scikit-learn, который вычисляет по каждой строке:
    - среднее значение (mean)
    - стандартное отклонение (std)

    Можно указать, какие именно колонки использовать. Если не указаны — берутся все.

    Пример использования:
        row_stats = RowStats(cols=["a", "b", "c"])
        X_new = row_stats.fit_transform(X)
    """


    def __init__(self, cols = None):
        """
        Инициализация трансформера.

        Аргументы:
            cols (list or None): Список колонок, по которым считать статистики.
                                 Если None — будут использованы все колонки из X.
        """
        self.cols = cols
    
    def fit (self, X, y = None):
        """
        Метод обучения (ничего не "учит", только запоминает список колонок).

        Аргументы:
            X (pd.DataFrame): Входной DataFrame с признаками.
            y (не используется): Поддержка совместимости с API sklearn.

        Возвращает:
            self: Объект трансформера (для возможности чейнинга).
        """
        if self.cols is None:
            self.cols = list(X.columns)
        return self
    
    def transform(self, X):
        """
        Преобразует DataFrame: считает статистики по строкам.

        Аргументы:
            X (pd.DataFrame): Входной DataFrame.

        Возвращает:
            pd.DataFrame: Новый DataFrame с двумя колонками:
                - "row_mean": среднее значение по строке
                - "row_std": стандартное отклонение по строке
        """
        sub = X[self.cols]
        return pd.DataFrame({
            "row_mean": sub.mean(axis = 1),
            "row_std": sub.std(axis = 1),
        }, index = X.index)