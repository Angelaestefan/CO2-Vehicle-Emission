import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

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

def print_metrics(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    expl_var = explained_variance_score(y_true, y_pred)
    print(f"\n--- {label} ---")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")
    print(f"Explained Variance: {expl_var:.4f}")

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



# Selecciona solo columnas numéricas para X

y_column = 'CO2 Emissions(g/km)'
categorical_cols = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
X = df.drop(columns=[y_column] + categorical_cols)
y = df[y_column]

# Separar datos en train, validation y test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42) # 0.25*0.8=0.2

models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=10.0)   
}

for name, model in models.items():
    model.fit(X_train, y_train)

    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    print(f"\n===== {name.upper()} =====")
    print_metrics(y_train, y_pred_train, "Train")
    print_metrics(y_val, y_pred_val, "Validation")
    print_metrics(y_test, y_pred_test, "Test")

    # Importancia de coeficientes (Top 10)
    coef = model.coef_
    feature_names = X.columns
    indices = np.argsort(np.abs(coef))[::-1][:10]

    plt.figure(figsize=(10,6))
    plt.barh(np.array(feature_names)[indices], coef[indices])
    plt.xlabel('Coeficiente')
    plt.title(f'Importancia de variables (Top 10) - {name}')
    plt.gca().invert_yaxis()
    plt.show()
    
    # Visualización: Predicciones vs Valores reales
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, y_pred_test, alpha=0.5, label='Test')
    plt.scatter(y_val, y_pred_val, alpha=0.5, label='Validation', color='orange')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs Valores reales')
    plt.legend()
    plt.show()
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='r2', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)

    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Entrenamiento')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='orange', label='Validación')
    plt.xlabel('Tamaño del conjunto de entrenamiento')
    plt.ylabel('R2 Score')
    plt.title('Curva de validación')
    plt.legend()
    plt.grid(True)
    plt.show()













