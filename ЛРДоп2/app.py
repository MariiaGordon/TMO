import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(
    page_title="Анализ данных",
    layout="wide"
)

st.title("Анализ данных с помощью машинного обучения")
st.write("Веб-приложение для обучения и сравнения моделей классификации.")

uploaded_file = st.file_uploader("Загрузите датасет", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Исходные данные")
    st.dataframe(df.head())

    st.write("Количество наблюдений:", df.shape[0])
    st.write("Количество столбцов:", df.shape[1])

    st.subheader("Настройки обучения")

    target = st.selectbox(
        "Выберите целевой признак",
        df.columns
    )

    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = X.select_dtypes(include="number").columns
    categorical_features = X.select_dtypes(exclude="number").columns

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    models = {
        "Метод k ближайших соседей": KNeighborsClassifier(n_neighbors=5),
        "Логистическая регрессия": LogisticRegression(max_iter=1000),
        "Дерево решений": DecisionTreeClassifier(random_state=42),
        "Случайный лес": RandomForestClassifier(random_state=42)
    }

    selected_models = st.multiselect(
        "Выберите модели для обучения",
        options=list(models.keys())
    )

    train_button = st.button("Обучить модели")

    if train_button:
        if not selected_models:
            st.warning("Выберите хотя бы одну модель.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            predictions = {}
            trained_models = {}

            with st.spinner("Выполняется обучение моделей..."):
                for name in selected_models:
                    model = Pipeline([
                        ("preprocessor", preprocessor),
                        ("classifier", models[name])
                    ])

                    model.fit(X_train, y_train)
                    trained_models[name] = model
                    predictions[name] = model.predict(X_test)

            st.success("Обучение моделей завершено!")

            results = []

            for name, y_pred in predictions.items():
                results.append({
                    "Модель": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
                })

            results = pd.DataFrame(results)
            results = results.sort_values("Accuracy", ascending=False)

            st.subheader("Результаты обучения")
            st.dataframe(
                results.style.format({
                    "Accuracy": "{:.5f}",
                    "Precision": "{:.5f}",
                    "Recall": "{:.5f}",
                    "F1-score": "{:.5f}"
                }),
                hide_index=True
            )
            st.subheader("Визуализация моделей")

            if "Метод k ближайших соседей" in selected_models:
                X_processed = preprocessor.transform(X_test)

                if hasattr(X_processed, "toarray"):
                    X_processed = X_processed.toarray()

                X_pca = PCA(n_components=2).fit_transform(X_processed)
                labels = predictions["Метод k ближайших соседей"]

                fig, ax = plt.subplots()

                for label in np.unique(labels):
                    mask = labels == label
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=str(label))

                ax.legend(title="Предсказанный класс")
                ax.set_title("Результаты классификации KNN")
                ax.set_xlabel("Главная компонента 1")
                ax.set_ylabel("Главная компонента 2")
                st.pyplot(fig, width=700)
                plt.close(fig)

            if "Логистическая регрессия" in selected_models:
                model = trained_models["Логистическая регрессия"]
                features = model.named_steps["preprocessor"].get_feature_names_out()
                coefficients = model.named_steps["classifier"].coef_

                importance = pd.Series(
                    np.abs(coefficients).mean(axis=0),
                    index=features
                ).nlargest(10)

                fig, ax = plt.subplots()
                importance.sort_values().plot.barh(ax=ax)
                ax.set_title("Коэффициенты логистической регрессии")
                ax.set_xlabel("Средний абсолютный коэффициент")
                ax.set_ylabel("Признак")
                st.pyplot(fig, width=700)
                plt.close(fig)

            if "Дерево решений" in selected_models:
                model = trained_models["Дерево решений"]
                features = model.named_steps["preprocessor"].get_feature_names_out()
                tree = model.named_steps["classifier"]

                fig, ax = plt.subplots(figsize=(16, 8))
                plot_tree(
                    tree,
                    feature_names=features,
                    class_names=[str(c) for c in tree.classes_],
                    filled=True,
                    max_depth=3,
                    ax=ax
                )
                ax.set_title("Структура дерева решений")
                st.pyplot(fig, width=700)
                plt.close(fig)

            if "Случайный лес" in selected_models:
                model = trained_models["Случайный лес"]
                features = model.named_steps["preprocessor"].get_feature_names_out()
                forest = model.named_steps["classifier"]

                importance = pd.Series(
                    forest.feature_importances_,
                    index=features
                ).nlargest(10)

                fig, ax = plt.subplots()
                importance.sort_values().plot.barh(ax=ax)
                ax.set_title("Важность признаков случайного леса")
                ax.set_xlabel("Важность признака")
                ax.set_ylabel("Признак")
                plt.tight_layout()
                st.pyplot(fig, width=700)
                plt.close(fig)