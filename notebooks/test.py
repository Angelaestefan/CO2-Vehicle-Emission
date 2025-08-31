# exploratory_analysis.ipynb

# ================================
# 🚗 Exploratory Data Analysis - CO2 Emissions Dataset
# ================================

# 📦 Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración para visualizaciones
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# ================================
# 1. Cargar los datos
# ================================
data_path = "../data/co2.csv"  # ajusta ruta si es necesario
df = pd.read_csv(data_path)

# ================================
# 2. Inspección inicial
# ================================
print("🔍 Dimensiones del dataset:", df.shape)
print("\n📋 Columnas y tipos de datos:\n", df.dtypes)
print("\n👀 Primeras filas:\n", df.head())

# ================================
# 3. Valores nulos y duplicados
# ================================
print("\n❓ Valores nulos por columna:\n", df.isnull().sum())
print("\n🔁 Duplicados:", df.duplicated().sum())

# ================================
# 4. Estadísticas descriptivas
# ================================
print("\n📊 Estadísticas básicas:\n", df.describe(include="all"))

# ================================
# 5. Distribución de la variable objetivo
# ================================
target = "CO2 Emissions(g/km)"  # revisa el nombre exacto de la columna
plt.figure(figsize=(8, 5))
sns.histplot(df[target], bins=30, kde=True, color="green")
plt.title("Distribución de emisiones de CO₂")
plt.xlabel("CO₂ g/km")
plt.ylabel("Frecuencia")
plt.show()

# ================================
# 6. Correlaciones numéricas
# ================================
plt.figure(figsize=(10, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="Greens", fmt=".2f")
plt.title("Matriz de correlación")
plt.show()

# ================================
# 7. Relación entre variables clave
# ================================
# Motor size vs Emisiones
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Engine Size(L)", y=target, data=df, alpha=0.7)
plt.title("Tamaño del motor vs Emisiones de CO₂")
plt.show()

# Fuel consumption vs Emisiones
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Fuel Consumption Comb (L/100 km)", y=target, hue="Fuel Type", data=df, alpha=0.7)
plt.title("Consumo de combustible vs Emisiones de CO₂")
plt.show()

# ================================
# 8. Comparación por variables categóricas
# ================================
plt.figure(figsize=(10, 5))
sns.boxplot(x="Fuel Type", y=target, data=df)
plt.title("Distribución de CO₂ según tipo de combustible")
plt.show()

plt.figure(figsize=(12, 5))
sns.boxplot(x="Transmission", y=target, data=df)
plt.title("Distribución de CO₂ según transmisión")
plt.xticks(rotation=45)
plt.show()
