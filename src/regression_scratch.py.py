import pandas as pd
import numpy as np


# Linear Regression Gradiente descendiente

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.theta = None
        self.mean = None
        self.std = None

    def fit(self, X, y):
        # Save normalization parameters
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        X_norm = (X - self.mean) / self.std

        m, n = X_norm.shape
        X_b = np.c_[np.ones((m, 1)), X_norm]  # add bias

        # Inicializar pesos
        self.theta = np.zeros(n + 1)

        # Gradiente Descendente
        for _ in range(self.n_iter):
            gradients = (1/m) * X_b.T.dot(X_b.dot(self.theta) - y)
            self.theta -= self.learning_rate * gradients

    def predict(self, X):
        X_norm = (X - self.mean) / self.std
        m, n = X_norm.shape
        X_b = np.c_[np.ones((m, 1)), X_norm]
        return X_b.dot(self.theta)

def agrupar_categorias(df, threshold=0.01):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        freqs = df[col].value_counts(normalize=True)
        rare_cats = freqs[freqs < threshold].index
        df[col] = df[col].apply(lambda x: 'otras' if x in rare_cats else x)
    return df


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    # leer dataset
    df = pd.read_csv("./data/co2.csv")
    
    df = df.drop_duplicates()
    df = df.dropna()
    
    df = agrupar_categorias(df, threshold=0.01)

    df = df.drop(columns=['Model', 'Make', 'Vehicle Class' , 'Transmission', 'Fuel Type'])

    # Features and target 
    X = df[["Engine Size(L)", "Cylinders","Fuel Consumption City (L/100 km)", 'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)' , 'Fuel Consumption Comb (mpg)']].values
    y = df["CO2 Emissions(g/km)"].values

    # Entrenar modelo
    model = LinearRegressionGD(learning_rate=0.01, n_iter=2000)
    model.fit(X, y)


    y_pred = model.predict(X)

    # Metricsa

    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - y_pred))


    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_manual = 1 - (ss_res / ss_tot)

    print("=== Model Evaluation ===")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R2 (manual): {r2_manual:.4f}")

    
    print("\nReal vs Predicted (first 10):")
    for real, pred in list(zip(y[:10], y_pred[:10])):
        print(f"Real: {real:.2f} | Pred: {pred:.2f} | Error: {real - pred:.2f}")


    X_new = np.array([[3.5, 6, 12, 8, 10, 30]]) 
    pred_new = model.predict(X_new)
    print("\nExample Prediction for Engine=3.5L, Cylinders=6, City=12:")
    print("Predicted CO2:", pred_new[0])
    
    # Visualización: Predicciones vs Valores reales
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs Valores reales')
    plt.show()
