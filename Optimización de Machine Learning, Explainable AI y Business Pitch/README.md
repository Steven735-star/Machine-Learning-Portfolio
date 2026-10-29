# ChurnGuard AI — Tarea 6

## Inicio local (Ubuntu)
1. Copia el archivo `WA_Fn-UseC_-Telco-Customer-Churn.csv` en `archive/`.
2. Desde esta carpeta ejecuta `python -m venv .venv && source .venv/bin/activate`.
3. `pip install -r requirements.txt`
4. `jupyter notebook ChurnGuard_AI_Profesional.ipynb` y ejecuta las celdas en orden.
5. `streamlit run app.py` para abrir la aplicación.

El notebook usa `Path.cwd()`; inicia Jupyter dentro de la carpeta del proyecto. Los artefactos se exportan a `artifacts/`. Para una comparación extensa cambia `PROFILE='deep'` antes de ejecutarlo. No consultes test para afinar modelos.

## Despliegue
Sube el repositorio a GitHub y publícalo en Streamlit Community Cloud seleccionando `app.py`. Incluye los artefactos generados (si respetan el tamaño admitido y la política de publicación) y el archivo `requirements.txt`. Las métricas y el dataset deben corresponder al entrenamiento efectivamente ejecutado. La URL de despliegue debe comprobarse manualmente; no está creada automáticamente.

## Limitaciones
El dataset Telco no mide el resultado causal de campañas. Las cifras monetarias son escenarios hipotéticos. SHAP y LIME explican predicciones, no relaciones causales. Las métricas no se han calculado aquí: ejecutar el notebook para obtenerlas. La sección de LIME/SHAP en Streamlit puede ser lenta o no soportar todos los ensambles; el notebook guarda versiones completas de las explicaciones en HTML y figuras.
