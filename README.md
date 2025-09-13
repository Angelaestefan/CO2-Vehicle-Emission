# 🚗🌱 Predicción de Emisiones de CO₂ en Vehículos

Este proyecto implementa un modelo de **regresión** para predecir las emisiones de CO₂ de automóviles utilizando un dataset de características técnicas de vehículos.  

El trabajo se divide en **dos enfoques**:
1. Una implementación **desde cero** usando únicamente `numpy` y `pandas` (sin librerías de machine learning).  
2. Una implementación usando librerías de ML `scikit-learn` para comparar resultados.  

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
   - Eliminación de duplicados y valores nulos.  
   - Filtrado de valores extremos no lógicos (ej. cilindros negativos).  
   - Normalización de variables numéricas para mejorar la convergencia del gradiente.  
   - Transformación opcional de variables categóricas mediante One-Hot Encoding.  

2. **Modelo 1: Regresión Lineal desde cero (Gradiente Descendiente)**  
   En este enfoque **no se utilizan librerías de machine learning**. La implementación se hace paso a paso con `numpy`:
   - **Inicialización**:  
     Los parámetros (θ) se inicializan en cero.  
   - **Normalización**:  
     Cada feature se estandariza con media y desviación estándar para evitar escalas distintas.  
   - **Función de costo (MSE)**:  
     Se mide el error cuadrático medio entre las predicciones y los valores reales.  
     
   - **Gradiente Descendiente**:  
     En cada iteración se actualizan los parámetros según la derivada parcial del costo:  
    
     Donde **α** es la tasa de aprendizaje.  
   - **Convergencia**:  
     El proceso se repite hasta que se alcanza un número fijo de iteraciones o los cambios en el costo son mínimos.  

   Este modelo está implementado en `src/regression_scratch.py`.

3. **Modelo 2: Regresión con librerías**  
   Para comparar, se utiliza `scikit-learn` con modelos como:
   - `LinearRegression` (regresión lineal estándar)  
 

4. **Evaluación**  
   - **Métricas utilizadas**:  
     - **MSE**: Error cuadrático medio  
     - **RMSE**: Raíz del error cuadrático medio  
     - **MAE**: Error absoluto medio  
     - **R²**: Coeficiente de determinación  
   - Se comparan los resultados de la implementación manual contra los de `scikit-learn`.  

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

