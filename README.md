# 🧠 Kaggle ML Задачи — Intermediate уровень

Этот репозиторий содержит мои решения прикладных ML-задач с Kaggle. Я подхожу к каждой задаче как **инженер** и **аналитик**, строго придерживаясь воспроизводимого и чистого пайплайна.

Задачи ориентированы на **практику для уровня Strong Junior / Intermediate ML Engineer** и охватывают весь ML-цикл:

---

## 📦 Структура подхода

Каждая задача решается по следующей схеме:

### 1. 📌 Постановка задачи
- Определение цели (регрессия, классификация, ранжирование)
- Выбор метрик качества
- Проверка утечек и временной зависимости

### 2. 📊 EDA (исследовательский анализ данных)
- Анализ пропусков, выбросов, типов признаков
- Построение распределений
- Поиск сильных и подозрительных фичей

### 3. 🏗️ Feature Engineering
- Импутация, Encoding, агрегаты, лаги
- Корреляции, мультиколлинеарность
- Категориальные флаги, binnings и ratios

### 4. ⚙️ ML Pipeline
- `sklearn.Pipeline`, `ColumnTransformer`
- Автоматизация препроцессинга и обучения
- Чистая структура: `features.py`, `config.yaml`, `train.py`

### 5. 📈 Обучение и валидация
- Модели: LightGBM / XGBoost / CatBoost
- Кросс-валидация: KFold, StratifiedKFold, TimeSeriesSplit
- Optuna для подбора гиперпараметров

### 6. 🔍 Интерпретация
- `.feature_importances_`, permutation importance, SHAP
- ROC AUC, confusion matrix, learning curves
- Интерпретация важности фичей с точки зрения бизнеса

### 7. 🧪 Финальное предсказание
- Финальное обучение на всем трейне
- Генерация сабмишнов (`submission.csv`)
- Подготовка README и `.py`/`.ipynb` файлов

---

## 📁 Структура проекта

```bash
kaggle-ml-intermediate/
├── competition_name/
│   ├── notebooks/
│   │   ├── EDA.ipynb
│   │   └── model_experiments.ipynb
│   ├── src/
│   │   ├── features.py
│   │   ├── pipeline.py
│   │   ├── train.py
│   │   └── utils.py
│   ├── config.yaml
│   ├── requirements.txt
│   ├── README.md
│   └── submission.csv
└── ...


⸻

🧰 Используемый стек
	•	Python, pandas, numpy
	•	scikit-learn, lightgbm, xgboost, catboost
	•	Optuna для tuning
	•	matplotlib, seaborn, plotly для визуализаций
	•	shap и eli5 для интерпретации моделей

⸻

💼 Цель

Цель этого репозитория —:
	•	собрать практическое ML-портфолио,
	•	показать владение продвинутыми приёмами фичеинжиниринга, валидации и упаковки моделей,
	•	отработать инженерную культуру и reproducibility.

⸻

🚀 Задачи

Задача	Тип	Метрика	Статус
Santander Customer Transaction	Классификация	AUC ROC	🟡 В процессе
IEEE-CIS Fraud Detection	Классификация	AUC ROC	🔜 Планируется
Instacart Market Basket	Ранжирование	F1 / MAP	🔜 Планируется

(список будет пополняться)

⸻

🧩 Контакты

TG: @skumbria1
