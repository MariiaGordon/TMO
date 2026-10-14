import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Загрузка данных
df = pd.read_csv("train.csv")

# Сохраняем количество пропусков до обработки
missing_before = df.isnull().sum()

print("Первые 5 строк:")
print(df.head())
print()
print("Размер датасета:")
print(df.shape)
print()
print("Типы данных:")
print(df.dtypes)
print()
print("Количество пропусков до обработки:")
print(df.isnull().sum())

# График пропусков до обработки
plt.figure(figsize=(10, 5))
missing_before[missing_before > 0].plot(kind="bar")
plt.title("Количество пропусков до обработки")
plt.xlabel("Признак")
plt.ylabel("Количество пропусков")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Обработка пропусков
# Заменяем пропуски в Age на медиану
df["Age"] = df["Age"].fillna(df["Age"].median())

# Заменяем пропуски в Embarked на самое частое значение
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Удаляем Cabin из-за большого количества пропусков
df = df.drop(columns=["Cabin"])

print()
print("Пропуски после обработки:")
print(df.isnull().sum())

# Кодирование категориальных признаков
df = pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    dtype=int
)

print()
print("Данные после кодирования категориальных признаков:")
print(df.head())

# Масштабирование данных
numeric_columns = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

# Сохраняем данные до масштабирования
before_scaling = df[numeric_columns].copy()

# Создаём StandardScaler
scaler = StandardScaler()

# Выполняем масштабирование
df[numeric_columns] = scaler.fit_transform(
    df[numeric_columns]
)

# Сохраняем данные после масштабирования
after_scaling = df[numeric_columns].copy()

print()
print("Данные после масштабирования:")
print(df[numeric_columns].head())

# График до и после масштабирования
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=before_scaling)
plt.title("До масштабирования")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.boxplot(data=after_scaling)
plt.title("После масштабирования")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Проверка результатов
print()
print("Итоговое количество пропусков:")
print(df.isnull().sum().sum())

print()
print("Средние значения масштабированных признаков:")
print(df[numeric_columns].mean())

print()
print("Стандартные отклонения масштабированных признаков:")
print(df[numeric_columns].std())