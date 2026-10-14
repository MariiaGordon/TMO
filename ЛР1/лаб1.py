import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Загрузка датасета
iris = load_iris()

# Создание DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Добавляем целевую переменную
df["target"] = iris.target

# Добавляем название вида ириса
df["species"] = df["target"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

print("Датасет Iris")
print()
print("Первые 5 строк:")
print(df.head())
print()
print("Размер датасета:")
print(df.shape)
print()
print("Типы данных:")
print(df.dtypes)
print()
print("Количество пропусков:")
print(df.isnull().sum())
print()
print("Статистические характеристики:")
print(df.describe())

# Гистограммы распределения признаков
df.hist(
    figsize=(12, 8),
    bins=15
)
plt.suptitle("Распределение признаков датасета Iris")
plt.tight_layout()
plt.savefig("result/histograms.png")
plt.show()

# Диаграмма рассеяния:
# длина чашелистика и длина лепестка
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="species"
)
plt.title("Зависимость длины лепестка от длины чашелистика")
plt.xlabel("Длина чашелистика (см)")
plt.ylabel("Длина лепестка (см)")
plt.tight_layout()
plt.savefig("result/scatter_sepal_petal.png")
plt.show()

# Диаграмма рассеяния:
# длина лепестка и ширина лепестка
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species"
)
plt.title("Зависимость ширины лепестка от его длины")
plt.xlabel("Длина лепестка (см)")
plt.ylabel("Ширина лепестка (см)")
plt.tight_layout()
plt.savefig("result/scatter_petal.png")
plt.show()

# Ящик с усами для сравнения длины лепестка
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="species",
    y="petal length (cm)"
)
plt.title("Распределение длины лепестка по видам ириса")
plt.xlabel("Вид ириса")
plt.ylabel("Длина лепестка (см)")
plt.tight_layout()
plt.savefig("result/boxplot_petal_length.png")
plt.show()

# Корреляция признаков
# Выбираем только числовые признаки
numeric_df = df[
    [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
]

# Вычисляем корреляционную матрицу
correlation_matrix = numeric_df.corr()

print()
print("Корреляционная матрица:")
print(correlation_matrix)

# Тепловая карта корреляций
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Корреляция признаков датасета Iris")
plt.tight_layout()
plt.savefig("result/correlation_heatmap.png")
plt.show()