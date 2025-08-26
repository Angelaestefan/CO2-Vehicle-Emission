# 🚗🌱 Predicción de Emisiones de CO₂ en Vehículos

Este proyecto implementa un modelo de **regresión** para predecir las emisiones de CO₂ de automóviles utilizando un dataset de características técnicas de vehículos.  

El trabajo se divide en **dos enfoques**:
1. Una implementación **desde cero** usando únicamente `numpy` y `pandas` (sin librerías de machine learning).  
2. Una implementación usando librerías de ML como `scikit-learn` para comparar resultados.  

---

## 📊 Dataset

- **Fuente**: [CO2 Emissions Dataset en Kaggle](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)  
- **Descripción**: El dataset contiene información sobre distintos automóviles, incluyendo:
  - Tamaño del motor  
  - Número de cilindros  
  - Consumo de combustible (ciudad, carretera, combinado)  
  - Tipo de combustible  
  - Transmisión  
  - Emisiones de CO₂ (variable objetivo)  

---

## 🎯 Objetivo

Entrenar un modelo de regresión que prediga las emisiones de CO₂ de un vehículo en función de sus características técnicas.  

---

## 🛠️ Metodología

1. **Preprocesamiento de datos**  
   - Limpieza de datos faltantes  
   - Transformación de variables categóricas (One-Hot Encoding o manual)  
   - Normalización/estandarización de variables numéricas  

2. **Modelo 1: Regresión desde cero (sin librerías de ML)**  
   - Implementación de regresión lineal usando el método de **gradiente descendente**  
   - Función de costo: Error Cuadrático Medio (MSE)  
   - Actualización de parámetros hasta convergencia  

3. **Modelo 2: Regresión con librerías**  
   - Implementación con `scikit-learn` (`LinearRegression`, `Ridge`, etc.)  
   - Comparación de resultados con la implementación manual  

4. **Evaluación**  
   - Métricas: RMSE, MAE, R²  
   - Comparación de desempeño entre los dos enfoques  

---

## ▶️ Ejecución

1. **Clonar repositorio y entrar al proyecto**:
 ```bash
 git clone <repo-url>
 cd co2-prediction
  ```

2. Instalar dependencias:
  ```bash
  pip install -r requirements.txt
  ```

3. Correr implementación desde cero:
  ```bash
  python src/regression_scratch.py
  ```

4. Correr implementación con librerías:
 ```bash
  python src/regression_sklearn.py
```

