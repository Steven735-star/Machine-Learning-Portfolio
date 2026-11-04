<div align="center">

# Machine Learning Portfolio

### From data exploration to intelligent, explainable applications

**A growing collection of machine learning projects, experiments, and practical implementations.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-16877E.svg)](LICENSE)

</div>

---

## Sobre este repositorio

Este repositorio reúne proyectos académicos, experimentos e implementaciones prácticas desarrollados durante mi formación en **Ciencias de la Computación**. Su objetivo es documentar el proceso completo de construcción de soluciones de aprendizaje automático: desde el análisis y preparación de datos hasta el entrenamiento, la evaluación, la interpretabilidad y, cuando corresponde, el despliegue.

El portafolio **crecerá progresivamente** con nuevos proyectos. Cada trabajo tendrá su propia documentación, dependencias, metodología y resultados para facilitar su reproducción y revisión independiente.

### Áreas de interés

- **Aprendizaje supervisado:** clasificación, regresión y comparación de algoritmos.
- **Optimización de modelos:** validación cruzada, búsqueda de hiperparámetros y optimización multiobjetivo.
- **Explainable AI (XAI):** interpretación de predicciones mediante SHAP, LIME y técnicas de importancia de variables.
- **Deep Learning:** redes neuronales y experimentación con distintas arquitecturas.
- **Aprendizaje federado:** entrenamiento distribuido, heterogeneidad de datos y evaluación de sistemas colaborativos.
- **Aplicaciones de ML:** integración de modelos entrenados en interfaces y herramientas interactivas.

> Estas son líneas de trabajo del portafolio; no todas corresponden todavía a proyectos publicados en este repositorio.

## Proyectos

| Proyecto | Descripción | Componentes principales |
|---|---|---|
| **[ChurnGuard AI — Customer Churn Prediction](<Optimización%20de%20Machine%20Learning%2C%20Explainable%20AI%20y%20Business%20Pitch/>)** | Predicción del abandono de clientes de telecomunicaciones y apoyo a campañas de retención. | Clasificación, Grid/Random/TPE, NSGA-II, SHAP, LIME y Streamlit. |

**Próximos proyectos:** se incorporarán a esta tabla a medida que estén documentados y disponibles en el repositorio.

### Proyecto destacado: ChurnGuard AI

ChurnGuard AI utiliza el dataset *Telco Customer Churn* para explorar patrones de abandono, comparar algoritmos de clasificación y desarrollar una aplicación que permita consultar predicciones y escenarios comerciales.

**Aspectos técnicos:**

- Comparación inicial de seis enfoques: Logistic Regression, Random Forest, CatBoost, XGBoost, LightGBM y Stacking.
- Optimización de CatBoost con Grid Search, Random Search y TPE; experimento adicional con algoritmo genético.
- Optimización multiobjetivo con NSGA-II, considerando F1-score y latencia de inferencia.
- Interpretabilidad mediante SHAP y LIME; referencia adicional de Random Forest Feature Importance.
- Dashboard en Streamlit con EDA, evaluación de modelos, predicción individual y simulación de campañas.

| Métrica del modelo final (test) | Resultado |
|---|---:|
| Modelo seleccionado | CatBoost + Grid Search |
| ROC-AUC | 0.8462 |
| Recall | 0.7193 |
| F1-score | 0.6329 |
| Accuracy | 0.7786 |

Los resultados pertenecen a la evaluación experimental del dataset utilizado. Las proyecciones económicas del simulador son **escenarios**, no resultados causales comprobados de campañas reales.

**Explorar el proyecto:** [código y notebook](<Optimización%20de%20Machine%20Learning%2C%20Explainable%20AI%20y%20Business%20Pitch/>) · [presentación](Slides/) · [README específico](<Optimización%20de%20Machine%20Learning%2C%20Explainable%20AI%20y%20Business%20Pitch/README.md>)

## Tecnologías y herramientas

Las herramientas varían según el proyecto. Entre las utilizadas en los trabajos publicados y las contempladas para nuevas implementaciones se encuentran:

| Categoría | Herramientas |
|---|---|
| Lenguaje y entorno | Python, Jupyter Notebook, Git y GitHub |
| Datos y análisis | NumPy, pandas, Matplotlib, Seaborn, Plotly |
| Machine Learning | scikit-learn, CatBoost, XGBoost, LightGBM |
| Optimización | Optuna, Grid Search, Random Search, NSGA-II |
| Explicabilidad | SHAP, LIME |
| Aplicaciones interactivas | Streamlit |

## Organización del repositorio

```text
Machine-Learning-Portfolio/
├── README.md                         # Presentación general del portafolio
├── LICENSE                           # Licencia MIT
├── .gitignore
├── Optimización de Machine Learning,
│   Explainable AI y Business Pitch/
│   ├── README.md                     # Guía específica de ChurnGuard AI
│   ├── ChurnGuard_AI_Profesional.ipynb
│   ├── app.py                        # Aplicación principal
│   ├── app2.py                       # Implementación alternativa
│   ├── requirements.txt
│   ├── archive/                      # Dataset
│   └── artifacts/                    # Modelos, reportes y figuras
└── Slides/                            # Material de presentación
```

Conforme se incorporen nuevos proyectos, cada uno tendrá su propia carpeta y, de ser necesario, un `README.md` específico.

## Cómo ejecutar un proyecto

1. Clona el repositorio y entra a la carpeta del proyecto que quieras explorar.
2. Consulta su `README.md` para identificar requisitos, dataset, archivos necesarios y comandos.
3. Crea un entorno virtual e instala las dependencias indicadas por ese proyecto.
4. Ejecuta el notebook, los scripts o la aplicación según corresponda.

Ejemplo general para Ubuntu/Linux:

```bash
git clone git@github.com:Steven735-star/Machine-Learning-Portfolio.git
cd Machine-Learning-Portfolio

# Accede a la carpeta del proyecto elegido antes de instalar dependencias.
# Consulta su README.md para ver las instrucciones particulares.
```

> **Reproducibilidad:** los modelos, datasets, semillas y dependencias pueden variar entre proyectos. Evita asumir que todas las carpetas utilizan el mismo entorno o los mismos comandos. Algunos artefactos serializados con `joblib` pueden requerir versiones compatibles de las bibliotecas utilizadas al entrenarlos; carga únicamente archivos de confianza.

## Metodología de trabajo

En los proyectos se procura documentar una secuencia reproducible:

**Datos → Exploración → Preprocesamiento → Entrenamiento → Validación → Interpretación → Aplicación**

Se presta especial atención a evitar la fuga de datos (*data leakage*), mantener separados los conjuntos de evaluación, utilizar métricas acordes al problema y reconocer las limitaciones de cada experimento.

## Autor

**Steven Rodríguez**  
Ciencias de la Computación  
GitHub: [@Steven735-star](https://github.com/Steven735-star)

## Licencia

Este repositorio se distribuye bajo la [licencia MIT](LICENSE), salvo que algún dataset, modelo de terceros o recurso incluido indique condiciones de uso específicas. Consulta las licencias de las fuentes originales antes de redistribuir sus materiales.

---

<div align="center">

*Learning, experimenting, and building — one project at a time.*

</div>
