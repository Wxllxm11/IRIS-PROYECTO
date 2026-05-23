# Iris Species Classifier

Universidad de la Costa — Mineria de Datos

Proyecto final 
## Integrantes

| Nombre         | Programa               |
|----------------|------------------------|
| William Angulo | Ingenieria de Sistemas |
| Martin Torres  | Ingenieria de Sistemas |

---

Universidad de la Costa · Mineria de Datos


---

## Objetivo

El objetivo es predecir a que especie pertenece una flor Iris segun cuatro medidas: el largo y ancho del sepalo, y el largo y ancho del petalo. El dataset tiene 150 flores repartidas en tres especies: setosa, versicolor y virginica.

Para eso entrenamos un modelo de Random Forest y montamos todo en un dashboard interactivo con Streamlit, donde se pueden ver las metricas del modelo, hacer predicciones en tiempo real y explorar los datos con graficas.

---

## Como correr el proyecto

### 1. Clona el repositorio

```
git clone https://github.com/Wxllxm11/IRIS-PROYECTO.git
cd IRIS-PROYECTO
```

### 2. Crea un entorno virtual (opcional pero recomendado)

```
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Instala las dependencias

```
pip install -r requirements.txt
```

### 4. Corre la app

```
streamlit run Proyect.py
```

https://iris-proyecto.streamlit.app/

---

## Que tiene el dashboard

El dashboard tiene tres secciones que se navegan desde el panel lateral:

**Metricas del Modelo**
Muestra Accuracy, Precision, Recall y F1-Score del modelo entrenado. Tambien incluye la matriz de confusion, la importancia de cada variable y un reporte completo de clasificacion por especie.

**Prediccion**
El usuario mueve cuatro sliders con las medidas de la flor y el modelo predice la especie al instante. Aparecen las probabilidades para cada clase y un grafico de dispersion 3D que muestra donde cae la nueva muestra dentro del dataset real.

**Exploracion de Datos**
Cuatro pestanas para explorar el dataset: tabla completa con estadisticas descriptivas, histogramas por variable, scatter matrix de todas las combinaciones y boxplots por especie.

---

## Como funciona el pipeline

1. Se carga el archivo Iris.csv y se limpian los nombres de columnas y especies.
2. Se hace una exploracion inicial con graficas para entender los datos.
3. Se estandarizan las variables con StandardScaler.
4. Se divide el dataset en 75% entrenamiento y 25% prueba de forma estratificada.
5. Se entrena un RandomForestClassifier con 100 arboles.
6. Se evalua el modelo con las cuatro metricas principales.
7. Todo se presenta en el dashboard de Streamlit.

---

## Por que Random Forest

Es robusto ante valores atipicos, no asume ninguna distribucion especifica de los datos, funciona bien con datasets pequenos como este y da de forma nativa la importancia de cada variable, lo que hace el modelo mas interpretable.

---

## Estructura del repositorio

```
IRIS-PROYECTO/
├── Proyect.py          # App principal de Streamlit
├── Iris.csv            # Dataset
├── requirements.txt    # Dependencias
└── README.md           # Este archivo
```

---
