import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

#Funciones
def agrupar_categorias(df, threshold=0.01):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        freqs = df[col].value_counts(normalize=True)
        rare_cats = freqs[freqs < threshold].index
        df[col] = df[col].apply(lambda x: 'otras' if x in rare_cats else x)
    return df

def one_hot_encode(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
    # Concatenar las columnas vectorizadas sin eliminar las originales
    df = pd.concat([df, encoded_df], axis=1)
    return df

# Cargar datos
DATA_PATH = './data/co2.csv'
df = pd.read_csv(DATA_PATH)


categorical_cols = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']

#Eliminar duplicados y nulos
df = df.drop_duplicates().dropna()

#Agrupar categorias 
df = agrupar_categorias(df, threshold=0.01)

# One-hot encoding
df = one_hot_encode(df)

# Ajusta estos nombres de columna según tu archivo CSV


# Selecciona solo columnas numéricas para X
y_column = 'CO2 Emissions(g/km)'
categorical_cols = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
X = df.drop(columns=[y_column] + categorical_cols)
y = df[y_column]



# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
expl_var = explained_variance_score(y_test, y_pred)

print('Coeficientes:', model.coef_)
print('Intercepto:', model.intercept_)
print('MSE (Error cuadrático medio):', mse)
print('MAE (Error absoluto medio):', mae)
print('R2 (Coeficiente de determinación):', r2)
print('Explained Variance (Varianza explicada):', expl_var)

# Visualización: Predicciones vs Valores reales
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores reales')
plt.show()

# Visualización: Importancia de coeficientes
import numpy as np
feature_names = X.columns
coef = model.coef_
indices = np.argsort(np.abs(coef))[::-1][:10]  # Top 10
plt.figure(figsize=(10,6))
plt.barh(np.array(feature_names)[indices], coef[indices])
plt.xlabel('Coeficiente')
plt.title('Importancia de variables (Top 10)')
plt.gca().invert_yaxis()
plt.show()
