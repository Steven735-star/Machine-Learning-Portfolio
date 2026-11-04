"""ChurnGuard AI | Customer Retention Intelligence.

Dashboard Streamlit (solo inferencia): carga el pipeline exportado, NO reentrena.
Estructura esperada:
    artifacts/models/churn_model.joblib, metadata.json
    artifacts/reports/*.csv  (baseline_cv, optimizer_comparison, final_metrics, pareto, ...)
    artifacts/figures/*.png
    archive/WA_Fn-UseC_-Telco-Customer-Churn.csv  (o artifacts/data/*.csv)
"""
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
REPORTS = ART / 'reports'
FIGS = ART / 'figures'

st.set_page_config(page_title='ChurnGuard AI', layout='wide',
                   initial_sidebar_state='expanded')

# --------------------------------------------------------------------------- #
# Estilo
# --------------------------------------------------------------------------- #
PRIMARY, DANGER, OK, WARN = '#3b82c4', '#d62728', '#2a9d8f', '#e9a820'
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {background: linear-gradient(90deg, #14365a 0%, #2e75b6 100%);
       padding: 1.2rem 1.6rem; border-radius: 10px; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 1.8rem; color: #ffffff !important;}
.hero p {margin: .25rem 0 0 0; color: #e6eef7 !important; font-size: .95rem;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.25);
       padding: .7rem .9rem; border-radius: 8px;}
.stTabs [data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
.risk-box {padding: 1rem 1.2rem; border-radius: 8px; border-left: 6px solid; margin: .5rem 0;}
</style>
<div class="hero"><h1>ChurnGuard AI</h1>
<p>Inteligencia de retención de clientes · Predicción, validación y explicabilidad.
Las asociaciones del modelo son predictivas, no causales.</p></div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def plot(fig, **kw):
    """plotly_chart compatible con versiones nuevas y antiguas de Streamlit."""
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    try:
        st.plotly_chart(fig, width='stretch', **kw)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, **kw)


def table(df, **kw):
    try:
        st.dataframe(df, width='stretch', hide_index=True, **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def read_csv(name):
    p = REPORTS / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def clean_name(s):
    """Quita prefijos del ColumnTransformer: 'cat__Contract_Month-to-month' -> 'Contract = Month-to-month'."""
    s = re.sub(r'\b(num|cat|remainder)__', '', str(s))
    return re.sub(r'^([A-Za-z]+)_(.+)$', r'\1 = \2', s) if '=' not in s and '<' not in s and '>' not in s else s


METRIC_KEYS = [('roc_auc', 'ROC-AUC'), ('auc', 'ROC-AUC'), ('accuracy', 'Accuracy'),
               ('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1')]


def metric_cols(df):
    """Detecta columnas de métricas {columna: etiqueta} (ignora desviaciones estándar)."""
    found, used = {}, set()
    for c in df.columns:
        lc = str(c).lower()
        if 'std' in lc or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        for key, label in METRIC_KEYS:
            if key in lc and label not in used:
                found[c] = label
                used.add(label)
                break
    return found


def model_col(df):
    for c in ('modelo', 'model', 'nombre', 'name'):
        if c in df.columns:
            return c
    obj = [c for c in df.columns if df[c].dtype == object]
    return obj[0] if obj else df.columns[0]


def find_dataset():
    for p in (ART / 'data' / 'telco_clean.csv', ROOT / 'archive' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'):
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_dataset(path):
    df = pd.read_csv(path)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].replace({1: 'Yes', 0: 'No', '1': 'Yes', '0': 'No'})
        df['ChurnBinary'] = (df['Churn'] == 'Yes').astype(int)
    return df


# --------------------------------------------------------------------------- #
# Carga del modelo (sin reentrenar)
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model():
    return joblib.load(ART / 'models' / 'churn_model.joblib')


if not (ART / 'models' / 'churn_model.joblib').exists():
    st.warning('Ejecuta primero el notebook desde el directorio del proyecto. '
               'Deben generarse artifacts/models y artifacts/reports.')
    st.stop()

model = load_model()
meta = json.loads((ART / 'models' / 'metadata.json').read_text(encoding='utf-8'))
THRESHOLD = float(meta['threshold'])
DATA_PATH = find_dataset()

# Valores finales del notebook (CatBoost + Grid Search, umbral 0.35, test aislado).
# Se usan solo como respaldo si final_metrics.csv / confusion_matrix.csv no tienen un formato reconocible.
FALLBACK_METRICS = {'Accuracy': 0.7786, 'Precision': 0.5651, 'Recall': 0.7193, 'F1': 0.6329, 'ROC-AUC': 0.8462}
FALLBACK_CM = np.array([[828, 207], [105, 269]])


def final_metrics():
    df = read_csv('final_metrics.csv')
    if not df.empty:
        mc = metric_cols(df)
        if mc:
            mcol = model_col(df)
            row = df[df[mcol].astype(str).str.contains('cat', case=False)] if mcol in df.columns else df
            row = (row if not row.empty else df).iloc[-1]
            vals = {lab: float(row[c]) for c, lab in mc.items()}
            return {k: (v / 100 if v > 1 else v) for k, v in vals.items()}
    return FALLBACK_METRICS


def confusion():
    df = read_csv('confusion_matrix.csv')
    if not df.empty:
        num = df.select_dtypes('number')
        if num.shape == (2, 2):
            return num.to_numpy()
    return FALLBACK_CM


with st.sidebar:
    st.header('Modelo en producción')
    st.markdown(f"**{meta.get('model_name', 'CatBoost')}**")
    st.metric('Umbral de decisión', f'{THRESHOLD:.2f}')
    fm = final_metrics()
    st.caption('Evaluación sobre conjunto de prueba aislado')
    for k in ('ROC-AUC', 'Recall', 'Precision', 'F1'):
        if k in fm:
            st.write(f'**{k}:** {fm[k]:.1%}')
    st.divider()
    st.caption('El umbral bajo (<0.5) prioriza detectar a quienes se van (Recall) '
               'aun contactando a algunos clientes de más.')
    st.caption('La app solo carga modelos exportados; no reentrena.')

@st.cache_resource
def rf_importance(top=15):
    """Importancia de variables del Random Forest de referencia (archivo exportado, sin reentrenar)."""
    p = ART / 'models' / 'random_forest_reference.joblib'
    if not p.exists():
        return None
    try:
        m = joblib.load(p)
        steps = list(m.named_steps.values()) if hasattr(m, 'named_steps') else [m]
        est = [x for x in steps if hasattr(x, 'feature_importances_')][-1]
        prep = next((x for x in steps if hasattr(x, 'get_feature_names_out')), model.named_steps['prep'])
        names = [clean_name(n) for n in prep.get_feature_names_out()]
        d = pd.DataFrame({'variable': names, 'importancia': est.feature_importances_})
        return d.sort_values('importancia', ascending=False).head(top)
    except Exception:
        return None


tabs = st.tabs(['EDA', 'Model Performance', 'Prediction & Explainability', 'Business Application'])

# --------------------------------------------------------------------------- #
# A. EDA
# --------------------------------------------------------------------------- #
with tabs[0]:
    st.subheader('Exploración del dataset')
    if DATA_PATH is None:
        st.info('Coloca el CSV en archive/ (o artifacts/data/) para activar el EDA interactivo.')
    else:
        df = load_dataset(DATA_PATH)
        with st.container(border=True):
            st.markdown('**Filtros de segmento**')
            f1, f2, f3 = st.columns(3)
            contratos = sorted(df['Contract'].unique())
            sel_c = f1.multiselect('Contrato', contratos, default=contratos)
            internet = sorted(df['InternetService'].unique())
            sel_i = f2.multiselect('Servicio de internet', internet, default=internet)
            pagos = sorted(df['PaymentMethod'].unique())
            sel_p = f3.multiselect('Método de pago', pagos, default=pagos)
            tmin, tmax = int(df.tenure.min()), int(df.tenure.max())
            rng = st.slider('Antigüedad (meses)', tmin, tmax, (tmin, tmax))
        frame = df[df.Contract.isin(sel_c) & df.InternetService.isin(sel_i) & df.PaymentMethod.isin(sel_p)
                   & df.tenure.between(*rng)]
        if frame.empty:
            st.warning('Los filtros no devuelven clientes.')
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric('Clientes', f'{len(frame):,}')
            k2.metric('Tasa de abandono', f'{frame.ChurnBinary.mean():.1%}',
                      delta=f'{frame.ChurnBinary.mean() - df.ChurnBinary.mean():+.1%} vs total', delta_color='inverse')
            k3.metric('Cargo mensual promedio', f'${frame.MonthlyCharges.mean():.2f}')
            k4.metric('Ingreso mensual perdido', f'${frame.loc[frame.ChurnBinary == 1, "MonthlyCharges"].sum():,.0f}')

            c1, c2 = st.columns(2)
            cats = [c for c in df.select_dtypes('object').columns if c not in ('customerID', 'Churn', 'TotalCharges')]
            var = c1.selectbox('Variable categórica a cruzar con abandono', cats,
                               index=cats.index('Contract') if 'Contract' in cats else 0)
            rate = frame.groupby(var).ChurnBinary.agg(['mean', 'size']).reset_index()
            rate.columns = [var, 'tasa_abandono', 'clientes']
            fig = px.bar(rate.sort_values('tasa_abandono'), x='tasa_abandono', y=var, orientation='h',
                         text=rate.sort_values('tasa_abandono').tasa_abandono.map('{:.1%}'.format),
                         hover_data=['clientes'], color='tasa_abandono', color_continuous_scale='Reds',
                         title=f'Tasa de abandono por {var}')
            fig.update_xaxes(tickformat='.0%')
            fig.update_layout(coloraxis_showscale=False)
            with c1:
                plot(fig)
            fig = px.histogram(frame, x='tenure', color='Churn', nbins=36, barmode='overlay', opacity=.7,
                               color_discrete_map={'Yes': DANGER, 'No': PRIMARY},
                               title='Distribución de antigüedad por abandono')
            with c2:
                plot(fig)

            c3, c4 = st.columns(2)
            fig = px.box(frame, x='Contract', y='MonthlyCharges', color='Churn',
                         color_discrete_map={'Yes': DANGER, 'No': PRIMARY}, title='Cargo mensual por contrato y abandono')
            with c3:
                plot(fig)
            samp = frame.sample(min(2500, len(frame)), random_state=1)
            fig = px.scatter(samp, x='tenure', y='MonthlyCharges', color='Churn', opacity=.55,
                             color_discrete_map={'Yes': DANGER, 'No': PRIMARY}, title='Antigüedad vs cargo mensual')
            with c4:
                plot(fig)
            st.caption('Los patrones son asociaciones descriptivas; no demuestran causalidad.')

# --------------------------------------------------------------------------- #
# B. Model Performance
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader('Desempeño del modelo final')
    st.caption('Métricas sobre el conjunto de prueba, aislado durante entrenamiento, optimización y selección.')
    cols = st.columns(5)
    for col, k in zip(cols, ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1']):
        if k in fm:
            col.metric(k, f'{fm[k]:.1%}')

    sec = st.radio('Sección', ['Modelos', 'Optimización de hiperparámetros', 'Frente de Pareto',
                               'Matriz de confusión', 'Explicabilidad global'],
                   horizontal=True, label_visibility='collapsed')

    # ---- Comparación de modelos ------------------------------------------------
    if sec == 'Modelos':
        dat = read_csv('baseline_cv.csv')
        if dat.empty:
            st.info('No se encontró baseline_cv.csv')
        else:
            mc, mcol = metric_cols(dat), model_col(dat)
            st.markdown('**Comparación de modelos (validación cruzada)**')
            table(dat)
            if len(mc) >= 3:
                norm = dat.copy()
                for c in mc:
                    if norm[c].max() > 1:
                        norm[c] = norm[c] / 100
                ca, cb = st.columns(2)
                fig = go.Figure()
                for _, r in norm.iterrows():
                    labels = list(mc.values())
                    vals = [r[c] for c in mc]
                    fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1],
                                                  name=str(r[mcol]), fill='toself', opacity=.35))
                lo = max(0, float(norm[list(mc)].min().min()) - 0.05)
                fig.update_layout(title='Perfil multimétrica (radar)', polar=dict(radialaxis=dict(range=[lo, 1])))
                with ca:
                    plot(fig)
                heat = norm.set_index(mcol)[list(mc)].rename(columns=mc)
                fig = px.imshow(heat, text_auto='.3f', aspect='auto', color_continuous_scale='Blues',
                                title='Mapa de calor de métricas')
                with cb:
                    plot(fig)
                gap = (heat - heat.max()) * 100
                fig = px.imshow(gap, text_auto='.2f', aspect='auto', color_continuous_scale='RdYlGn',
                                title='Brecha vs el mejor modelo en cada métrica (puntos porcentuales)')
                plot(fig)
            else:
                ycol = next(iter(mc), dat.select_dtypes('number').columns[0])
                fig = px.bar(dat.sort_values(ycol), x=ycol, y=mcol, orientation='h', text_auto='.3f')
                fig.update_xaxes(range=[dat[ycol].min() * 0.95, dat[ycol].max() * 1.01])
                plot(fig)
            st.caption('Se usa un radar, un mapa de calor y la brecha vs el mejor modelo, porque las diferencias '
                       'de ROC-AUC entre modelos son pequeñas. Recall y F1 son más relevantes que Accuracy '
                       'en un problema desbalanceado (~26% de abandono).')

    # ---- Optimizadores ---------------------------------------------------------
    elif sec == 'Optimización de hiperparámetros':
        dat = read_csv('optimizer_comparison.csv')
        sp = read_csv('search_spaces.csv')
        if dat.empty:
            st.info('No se encontró optimizer_comparison.csv')
        else:
            st.markdown('**Comparación de estrategias (mismo modelo base, particiones y validación cruzada)**')
            table(dat)
            if {'metodo', 'segundos', 'roc_auc_cv'}.issubset(dat.columns):
                ca, cb = st.columns(2)
                fig = px.bar(dat.sort_values('segundos'), x='metodo', y='segundos', text_auto='.1f',
                             color='metodo', title='Tiempo de optimización (s)')
                fig.update_layout(showlegend=False)
                with ca:
                    plot(fig)
                if 'configuraciones' in dat.columns:
                    fig = px.bar(dat, x='metodo', y='configuraciones', text_auto=True, color='metodo',
                                 title='Configuraciones evaluadas')
                    fig.update_layout(showlegend=False)
                    with cb:
                        plot(fig)
                fig = px.scatter(dat, x='segundos', y='roc_auc_cv', color='metodo', text='metodo',
                                 size='configuraciones' if 'configuraciones' in dat.columns else None,
                                 title='Eficiencia: tiempo vs ROC-AUC CV (eje Y recortado para ver diferencias)')
                fig.update_traces(textposition='top center')
                pad = max((dat.roc_auc_cv.max() - dat.roc_auc_cv.min()) * 0.5, 0.002)
                fig.update_yaxes(range=[dat.roc_auc_cv.min() - pad, dat.roc_auc_cv.max() + pad])
                plot(fig)
        if not sp.empty:
            st.markdown('**Hiperparámetros y espacios de búsqueda**')
            table(sp)
        else:
            st.caption('Opcional: guarda artifacts/reports/search_spaces.csv desde el notebook para mostrar aquí '
                       'la tabla de espacios de búsqueda que pide el deber.')

    # ---- Pareto ----------------------------------------------------------------
    elif sec == 'Frente de Pareto':
        dat = read_csv('pareto.csv')
        if dat.empty or not {'ms_cliente', 'f1_cv'}.issubset(dat.columns):
            st.info('No se encontró pareto.csv con columnas ms_cliente y f1_cv')
        else:
            dat = dat.reset_index(drop=True)
            f, t = dat.f1_cv.to_numpy(), dat.ms_cliente.to_numpy()
            dominated = [bool(np.any((f >= f[i]) & (t <= t[i]) & ((f > f[i]) | (t < t[i])))) for i in range(len(dat))]
            dat['estado'] = np.where(dominated, 'Dominada', 'No dominada')
            nd = dat[dat.estado == 'No dominada']
            fn = (nd.f1_cv - nd.f1_cv.min()) / max(np.ptp(nd.f1_cv), 1e-12)
            tn = (nd.ms_cliente - nd.ms_cliente.min()) / max(np.ptp(nd.ms_cliente), 1e-12)
            knee = (nd.index[np.argmin(np.sqrt((1 - fn) ** 2 + tn ** 2))]) if len(nd) else dat.index[0]
            label = 'trial' if 'trial' in dat.columns else None
            opts = list(dat.index)
            choice = st.selectbox('Resaltar solución', opts, index=opts.index(knee),
                                  format_func=lambda i: f'#{dat.loc[i, label] if label else i}'
                                  f'{" (punto de compromiso)" if i == knee else ""}')
            fig = px.scatter(dat, x='ms_cliente', y='f1_cv', color='estado',
                             color_discrete_map={'No dominada': PRIMARY, 'Dominada': '#b0b7c3'},
                             hover_data=[label] if label else None,
                             title='Frente de Pareto: maximizar F1 (CV) vs minimizar tiempo de inferencia')
            allp = read_csv('nsga_all.csv')
            if {'ms_cliente', 'f1_cv'}.issubset(allp.columns):
                fig.add_trace(go.Scatter(x=allp.ms_cliente, y=allp.f1_cv, mode='markers', name='Todas las evaluaciones',
                                         marker=dict(size=5, color='rgba(150,150,150,.35)')))
            line = nd.sort_values('ms_cliente')
            fig.add_trace(go.Scatter(x=line.ms_cliente, y=line.f1_cv, mode='lines', name='Frente',
                                     line=dict(color=PRIMARY, dash='dot')))
            sel = dat.loc[[choice]]
            fig.add_trace(go.Scatter(x=sel.ms_cliente, y=sel.f1_cv, mode='markers', name='Seleccionada',
                                     marker=dict(size=18, color=WARN, symbol='star', line=dict(width=1, color='black'))))
            fig.update_xaxes(title='Tiempo de inferencia (ms/cliente)')
            fig.update_yaxes(title='F1 (CV)')
            plot(fig)
            r = dat.loc[choice]
            st.info(f'Solución #{r[label] if label else choice}: F1 CV = {r.f1_cv:.4f}, '
                    f'{r.ms_cliente:.4f} ms/cliente. Una solución más lenta solo conviene si la ganancia en F1 '
                    'compensa el costo de cómputo; en retención por lotes la latencia pesa poco, '
                    'mientras que en predicción en línea pesa más.')
            table(dat)

    # ---- Matriz de confusión ---------------------------------------------------
    elif sec == 'Matriz de confusión':
        cm = confusion()
        ca, cb = st.columns(2)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                        x=['Pred: No abandona', 'Pred: Abandona'], y=['Real: No abandona', 'Real: Abandona'],
                        title='Matriz de confusión (conteos)')
        with ca:
            plot(fig)
        pct = cm / cm.sum(axis=1, keepdims=True)
        fig = px.imshow(pct, text_auto='.1%', color_continuous_scale='Greens',
                        x=['Pred: No abandona', 'Pred: Abandona'], y=['Real: No abandona', 'Real: Abandona'],
                        title='Normalizada por clase real (Recall en la diagonal)')
        with cb:
            plot(fig)
        tn, fp, fn_, tp = cm.ravel()
        st.markdown(f'- **{tp}** clientes que se iban fueron detectados (Recall {tp / (tp + fn_):.1%}).\n'
                    f'- **{fn_}** clientes que se iban no fueron detectados: es el error más costoso.\n'
                    f'- **{fp}** clientes que se quedaban fueron contactados de más (Precision {tp / (tp + fp):.1%}).')
        thr = read_csv('threshold_oof.csv')
        if not thr.empty:
            xcol = next((c for c in thr.columns if re.search(r'thr|umbral|thresh', str(c), re.I)), thr.columns[0])
            ycols = [c for c in thr.select_dtypes('number').columns if c != xcol]
            if ycols:
                fig = px.line(thr, x=xcol, y=ycols, title='Selección del umbral con predicciones fuera de muestra (CV)')
                fig.add_vline(x=THRESHOLD, line_dash='dot', line_color=WARN,
                              annotation_text=f'Umbral elegido {THRESHOLD:.2f}')
                fig.update_layout(legend_title_text='')
                plot(fig)
        pic = FIGS / 'final_evaluation.png'
        if pic.exists():
            with st.expander('Figura original del notebook (matriz y calibración)'):
                st.image(str(pic))

    # ---- XAI global ------------------------------------------------------------
    else:
        c1, c2 = st.columns(2)
        rf = rf_importance()
        if rf is not None:
            with c1:
                plot(px.bar(rf.iloc[::-1], x='importancia', y='variable', orientation='h',
                            title='Importancia de variables: Random Forest (referencia)'))
        else:
            c1.info('No se pudo calcular la importancia de Random Forest desde random_forest_reference.joblib.')
        gpic = FIGS / 'shap_global.png'
        if gpic.exists():
            with c2:
                st.markdown('**SHAP global (CatBoost)**')
                st.image(str(gpic))
        lpic = FIGS / 'shap_local.png'
        if lpic.exists():
            with st.expander('SHAP local: ejemplos del notebook'):
                st.image(str(lpic))
        lime_files = sorted(REPORTS.glob('lime_cliente_*.html'))
        if lime_files:
            st.markdown('**LIME: clientes explicados en el notebook**')
            ltabs = st.tabs([f'Cliente {i + 1}' for i in range(len(lime_files))])
            for t, fpath in zip(ltabs, lime_files):
                with t:
                    components.html(fpath.read_text(encoding='utf-8'), height=480, scrolling=True)
        st.caption('Importancia predictiva no implica causalidad. SHAP (CatBoost) está en escala log-odds.')

# --------------------------------------------------------------------------- #
# C. Prediction & Explainability
# --------------------------------------------------------------------------- #
LABELS = {'gender': 'Género', 'SeniorCitizen': 'Adulto mayor', 'Partner': 'Tiene pareja', 'Dependents': 'Dependientes',
          'tenure': 'Antigüedad (meses)', 'PhoneService': 'Servicio telefónico', 'MultipleLines': 'Líneas múltiples',
          'InternetService': 'Tipo de internet', 'OnlineSecurity': 'Seguridad en línea', 'OnlineBackup': 'Respaldo en línea',
          'DeviceProtection': 'Protección de equipo', 'TechSupport': 'Soporte técnico', 'StreamingTV': 'TV en streaming',
          'StreamingMovies': 'Películas en streaming', 'Contract': 'Tipo de contrato',
          'PaperlessBilling': 'Facturación electrónica', 'PaymentMethod': 'Método de pago',
          'MonthlyCharges': 'Cargo mensual (USD)', 'TotalCharges': 'Cargos totales (USD)'}
GROUPS = [('Perfil', ['gender', 'SeniorCitizen', 'Partner', 'Dependents']),
          ('Contrato y facturación', ['tenure', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                                         'MonthlyCharges', 'TotalCharges']),
          ('Servicios', ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'])]


@st.cache_resource(show_spinner='Preparando explicadores…')
def get_explainers():
    import shap
    prep, clf = model.named_steps['prep'], model.named_steps['model']
    names = list(prep.get_feature_names_out())
    shap_exp = shap.TreeExplainer(clf)
    lime_exp = None
    try:
        from lime.lime_tabular import LimeTabularExplainer
        bg_path = REPORTS / 'background.csv'
        if bg_path.exists():
            raw = pd.read_csv(bg_path)
        elif DATA_PATH is not None:
            raw = load_dataset(DATA_PATH).sample(300, random_state=42)
        else:
            raw = None
        if raw is not None:
            bg = np.asarray(prep.transform(raw[meta['columns']]), dtype=float)
            cat_idx = [i for i, n in enumerate(names) if n.startswith('cat__')]
            lime_exp = LimeTabularExplainer(bg, feature_names=names, categorical_features=cat_idx,
                                            class_names=['No abandona', 'Abandona'], mode='classification',
                                            random_state=42)
    except Exception:
        lime_exp = None
    return shap_exp, lime_exp, names


def bar_contrib(names, vals, title):
    d = pd.DataFrame({'variable': [clean_name(n) for n in names], 'efecto': np.asarray(vals, dtype=float)})
    d = d.reindex(d.efecto.abs().sort_values(ascending=False).index).head(10)
    d['sentido'] = np.where(d.efecto > 0, 'Aumenta el riesgo', 'Reduce el riesgo')
    fig = px.bar(d.iloc[::-1], x='efecto', y='variable', color='sentido', orientation='h', title=title,
                 color_discrete_map={'Aumenta el riesgo': DANGER, 'Reduce el riesgo': PRIMARY})
    fig.update_layout(legend_title_text='', yaxis_title='')
    return fig


def risk_level(p):
    if p >= THRESHOLD:
        return 'ALTO', DANGER
    if p >= THRESHOLD * 0.6:
        return 'MEDIO', WARN
    return 'BAJO', OK


def gauge(p):
    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=p * 100, number={'suffix': '%', 'valueformat': '.1f'},
        title={'text': 'Probabilidad de abandono'},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#333'},
               'steps': [{'range': [0, THRESHOLD * 60], 'color': '#c9ece8'},
                         {'range': [THRESHOLD * 60, THRESHOLD * 100], 'color': '#f8e3b0'},
                         {'range': [THRESHOLD * 100, 100], 'color': '#f4c1c1'}],
               'threshold': {'line': {'color': 'black', 'width': 4}, 'value': THRESHOLD * 100}}))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def friendly(feature):
    """Convierte una variable transformada en frase legible para no técnicos."""
    s = clean_name(feature)
    return s.replace(' = ', ': ')


with tabs[2]:
    st.subheader('Predicción individual')
    st.caption(f'Modelo: {meta.get("model_name", "CatBoost")} · umbral de decisión {THRESHOLD:.2f} '
               '(seleccionado por validación cruzada)')

    numeric = set(meta['numerical_features'])
    values, columns = {}, list(meta['columns'])

    def field(c, box):
        label = LABELS.get(c, c)
        if c in numeric:
            d = float(meta['numeric_defaults'].get(c, 0))
            if c == 'tenure':
                return box.slider(label, 0, 72, int(min(max(d, 0), 72)), key=f'in_{c}')
            if c == 'MonthlyCharges':
                return box.slider(label, 15.0, 125.0, float(min(max(d, 15), 125)), 0.5, key=f'in_{c}')
            if c == 'SeniorCitizen':
                return box.selectbox(label, [0, 1], index=int(d >= 0.5), format_func=lambda v: 'Sí' if v else 'No',
                                     key=f'in_{c}')
            return box.number_input(label, value=d, key=f'in_{c}')
        opts = meta['categorical_options'][c]
        d = meta['category_defaults'].get(c)
        return box.selectbox(label, opts, index=opts.index(d) if d in opts else 0, key=f'in_{c}')

    placed = set()
    gcols = st.columns(3)
    for gcol, (title, fields) in zip(gcols, GROUPS):
        with gcol.container(border=True):
            st.markdown(f'**{title}**')
            for c in fields:
                if c not in columns:
                    continue
                placed.add(c)
                if c == 'TotalCharges':
                    auto = st.checkbox('Calcular cargos totales = antigüedad × cargo mensual', value=True, key='auto_tc')
                    if auto:
                        values[c] = float(values.get('tenure', 0)) * float(values.get('MonthlyCharges', 0))
                        st.caption(f'Cargos totales: ${values[c]:,.2f}')
                    else:
                        values[c] = field(c, st)
                else:
                    values[c] = field(c, st)
    rest = [c for c in columns if c not in placed]
    for c in rest:   # columnas no previstas (p. ej. customerID) se rellenan con un valor neutro
        if c.lower() == 'customerid':
            values[c] = 'APP-0000'
        else:
            values[c] = field(c, st)

    if st.button('Analizar riesgo del cliente', type='primary'):
        x = pd.DataFrame([values], columns=columns)
        prep, clf = model.named_steps['prep'], model.named_steps['model']
        transformed = np.asarray(prep.transform(x), dtype=float)   # fuera de los try (corrige el NameError)
        p = float(model.predict_proba(x)[0, 1])
        res = {'p': p, 'shap': None, 'lime': None, 'names': None, 'err': []}
        try:
            shap_exp, lime_exp, names = get_explainers()
            res['names'] = names
            try:
                sv = shap_exp.shap_values(transformed)
                if isinstance(sv, list):
                    v = sv[1][0]
                else:
                    sv = np.asarray(sv)
                    v = sv[0, :, 1] if sv.ndim == 3 else sv[0]
                res['shap'] = np.asarray(v, dtype=float)
            except Exception as ex:
                res['err'].append(f'SHAP no disponible: {str(ex)[:160]}')
            if lime_exp is None:
                res['err'].append('LIME desactivado: falta artifacts/reports/background.csv o el CSV del dataset.')
            else:
                try:
                    d = lime_exp.explain_instance(transformed[0], clf.predict_proba, num_features=10, num_samples=1000)
                    res['lime'] = d.as_list()
                except Exception as ex:
                    res['err'].append(f'LIME no disponible: {str(ex)[:160]}')
        except Exception as ex:
            res['err'].append(f'No se pudieron cargar los explicadores: {str(ex)[:160]}')
        st.session_state['pred'] = res

    res = st.session_state.get('pred')
    if res:
        p = res['p']
        lvl, color = risk_level(p)
        st.divider()
        g1, g2 = st.columns([1, 1.3])
        with g1:
            plot(gauge(p))
        with g2:
            st.markdown(f'<div class="risk-box" style="border-color:{color};background:{color}1a">'
                        f'<h3 style="margin:0;color:{color}">Riesgo {lvl}</h3>'
                        f'Probabilidad estimada de abandono: <b>{p:.1%}</b> '
                        f'(umbral de acción: {THRESHOLD:.0%}).</div>', unsafe_allow_html=True)
            if res['shap'] is not None:
                order = np.argsort(res['shap'])
                up = [friendly(res['names'][i]) for i in order[::-1][:3] if res['shap'][i] > 0]
                down = [friendly(res['names'][i]) for i in order[:3] if res['shap'][i] < 0]
                st.markdown('**En palabras simples**')
                txt = (f'El modelo estima que este cliente tiene un riesgo **{lvl.lower()}** de abandonar. ')
                if up:
                    txt += 'Lo que más **aumenta** ese riesgo: ' + '; '.join(up) + '. '
                if down:
                    txt += 'Lo que más lo **reduce**: ' + '; '.join(down) + '. '
                st.write(txt)
                if lvl == 'ALTO':
                    st.success('Acción sugerida: incluir al cliente en una campaña de retención.')
            st.caption('Esta probabilidad es predictiva; una característica influyente no es una causa comprobada.')

        c1, c2 = st.columns(2)
        with c1:
            if res['shap'] is not None:
                plot(bar_contrib(res['names'], res['shap'], 'SHAP: qué empuja esta predicción'))
                st.caption('Escala log-odds: importan el signo y el orden, no el valor como porcentaje.')
        with c2:
            if res['lime']:
                plot(bar_contrib([a for a, _ in res['lime']], [b for _, b in res['lime']], 'LIME: explicación local'))
                st.caption('LIME usa variables numéricas estandarizadas: un umbral como "tenure <= -0.95" '
                           'es antigüedad baja, no meses reales.')
        for e in res['err']:
            st.caption(e)

# --------------------------------------------------------------------------- #
# D. Business Application
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner='Calculando riesgo por cliente…')
def scored_base(_model, path_str, mtime):
    """Probabilidades por cliente. Prefiere test_predictions.csv (test aislado); si no, puntúa el dataset."""
    tp = REPORTS / 'test_predictions.csv'
    if tp.exists():
        t = pd.read_csv(tp)
        pc = next((c for c in t.columns if c.lower() in ('proba', 'prob', 'probabilidad', 'y_proba', 'y_prob')), None)
        yc = next((c for c in t.columns if c.lower() in ('y_true', 'churn', 'real', 'y', 'target')), None)
        if pc and yc:
            y = t[yc].replace({'Yes': 1, 'No': 0}).astype(int)
            return pd.DataFrame({'proba': t[pc].astype(float), 'real': y}), 'test_predictions.csv (test aislado)'
    if path_str:
        d = load_dataset(Path(path_str))
        proba = _model.predict_proba(d[meta['columns']])[:, 1]
        out = pd.DataFrame({'proba': proba, 'real': d['ChurnBinary']})
        if 'customerID' in d.columns:
            out.insert(0, 'customerID', d['customerID'])
        return out, 'dataset completo (incluye datos de entrenamiento: orientativo)'
    return pd.DataFrame(), ''


with tabs[3]:
    st.subheader('Simulador de campañas de retención')
    st.info('**Simulación de impacto económico.** Se combinan las probabilidades estimadas por '
            'ChurnGuard AI con parámetros comerciales configurables. Los beneficios son proyecciones '
            'hipotéticas, no ingresos obtenidos ni efectos causales demostrados.')

    base, origen = scored_base(model, str(DATA_PATH) if DATA_PATH else '',
                               DATA_PATH.stat().st_mtime if DATA_PATH else 0)
    if base.empty:
        st.info('Se necesita artifacts/reports/test_predictions.csv (proba, y_true) o el CSV del dataset.')
    else:
        st.caption(f'Base para la simulación: {origen} · {len(base):,} registros. '
                   'Para una estimación de desempeño independiente, utilizar predicciones del conjunto de prueba.')
        with st.container(border=True):
            a, b, c, d = st.columns(4)
            costo = a.slider('Costo por contacto (USD)', 1, 50, 10)
            exito = b.slider('Eficacia incremental hipotética de la campaña', 0.05, 0.60, 0.20, 0.01,
                             help='Proporción supuesta de abandonos esperados que se evitarían con la campaña. '
                                  'No es una métrica medida por CatBoost.')
            margen = c.slider('Margen mensual por cliente retenido (USD)', 10, 120, 45)
            meses = d.slider('Horizonte de valor retenido (meses)', 3, 36, 12)
            e, f = st.columns(2)
            pct = e.slider('% de clientes a contactar (mayor riesgo primero)', 1, 100, 30)
            cartera = f.number_input('Tamaño de cartera del comprador (clientes)', 1000, 5_000_000, 50_000, 1000)
            precio = st.number_input('Precio anual propuesto de ChurnGuard AI (USD)', 0, 1_000_000, 24_000, 1000)

        s = base.sort_values('proba', ascending=False).reset_index(drop=True).copy()
        s['proba'] = pd.to_numeric(s['proba'], errors='coerce').clip(0, 1)
        s = s.dropna(subset=['proba']).reset_index(drop=True)
        if s.empty:
            st.error('No existen probabilidades válidas para simular la campaña.')
            st.stop()
        scale = cartera / len(s)
        prob_cum = np.cumsum(s['proba'].to_numpy())

        def economic_projection(n):
            # Proyección comercial: no usa las etiquetas reales de abandono.
            expected_churn = float(prob_cum[n - 1]) * scale
            retained = expected_churn * exito
            contacts = n * scale
            gross = retained * margen * meses
            contact_cost = contacts * costo
            net_before_license = gross - contact_cost
            total_cost = contact_cost + precio
            net_total = gross - total_cost
            roi_total = net_total / total_cost if total_cost > 0 else float('nan')
            return {'expected_churn': expected_churn, 'retained': retained,
                    'contacts': contacts, 'gross': gross, 'contact_cost': contact_cost,
                    'net_before_license': net_before_license, 'net_total': net_total,
                    'roi_total': roi_total}

        n = max(1, int(len(s) * pct / 100))
        proj = economic_projection(n)
        # Evaluación retrospectiva, separada del modelo comercial y solo si existen etiquetas.
        real = pd.to_numeric(s['real'], errors='coerce') if 'real' in s else None
        has_real = real is not None and real.notna().all() and real.sum() > 0
        capture = float(real.iloc[:n].sum() / real.sum()) if has_real else None

        k = st.columns(5)
        k[0].metric('Clientes a contactar', f'{proj["contacts"]:,.0f}')
        k[1].metric('Abandonos esperados entre contactados', f'{proj["expected_churn"]:,.0f}')
        k[2].metric('Retenciones hipotéticas', f'{proj["retained"]:,.0f}')
        k[3].metric('Beneficio neto tras todos los costos', f'${proj["net_total"]:,.0f}')
        k[4].metric('ROI total hipotético', f'{proj["roi_total"]:,.0%}' if np.isfinite(proj['roi_total']) else 'n/d')
        st.caption('ROI total = (valor bruto proyectado − costos de contacto − licencia) / '
                   '(costos de contacto + licencia). El beneficio supone que las probabilidades '
                   'son aplicables a la cartera y que la eficacia incremental configurada se cumple.')

        grid = np.unique(np.r_[np.linspace(1, len(s), 60).astype(int), n, len(s)])
        curve = pd.DataFrame({'pct_contactado': grid / len(s) * 100,
                              'beneficio': [economic_projection(int(i))['net_total'] for i in grid],
                              'captura_estimada': [float(prob_cum[i - 1] / prob_cum[-1]) * 100
                                                   if prob_cum[-1] > 0 else 0 for i in grid]})
        best = curve.loc[curve.beneficio.idxmax()]
        c1, c2 = st.columns(2)
        fig = px.line(curve, x='pct_contactado', y='beneficio',
                      title='Beneficio neto proyectado vs % de clientes contactados')
        fig.add_vline(x=pct, line_dash='dot', line_color=WARN, annotation_text='Tu selección')
        fig.add_trace(go.Scatter(x=[best.pct_contactado], y=[best.beneficio], mode='markers',
                                 name='Mejor escenario evaluado',
                                 marker=dict(size=13, color=DANGER, symbol='star')))
        fig.update_xaxes(title='% de clientes contactados')
        fig.update_yaxes(title='Beneficio neto hipotético (USD)')
        with c1:
            plot(fig)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve.pct_contactado, y=curve.captura_estimada,
                                 name='Concentración de riesgo estimado', line=dict(color=PRIMARY)))
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name='Selección aleatoria (referencia)',
                                 line=dict(dash='dash', color='gray')))
        fig.update_layout(title='Ganancia acumulada estimada según el modelo')
        fig.update_xaxes(title='% de clientes contactados')
        fig.update_yaxes(title='% del riesgo esperado acumulado')
        with c2:
            plot(fig)
        st.caption(f'Entre los escenarios evaluados, el mayor beneficio neto proyectado se obtiene '
                   f'contactando aproximadamente {best.pct_contactado:.0f}% de la cartera '
                   f'(${best.beneficio:,.0f}). Depende de los supuestos configurados.')

        # Mostrar claramente la parte histórica sin emplearla para los beneficios estimados.
        s['decil'] = pd.qcut(s.index, min(10, len(s)),
                             labels=[f'D{i}' for i in range(1, min(10, len(s)) + 1)])
        if has_real:
            hist = s.copy()
            hist['real'] = real
            dec = hist.groupby('decil', observed=True).agg(
                clientes=('proba', 'size'), prob_media=('proba', 'mean'),
                tasa_real=('real', 'mean')).reset_index()
            dec['lift'] = dec.tasa_real / hist.real.mean() if hist.real.mean() > 0 else np.nan
            st.markdown('**Evaluación retrospectiva por deciles de riesgo (D1 = mayor riesgo)**')
            st.caption('Esta sección utiliza etiquetas históricas conocidas para describir la '
                       'concentración del abandono. No interviene en la proyección económica anterior.'
                       + (' Incluye registros de entrenamiento; no debe presentarse como evaluación independiente.'
                          if 'dataset completo' in origen else ''))
            st.metric('Abandonos históricos capturados en el segmento elegido', f'{capture:.1%}')
            cc1, cc2 = st.columns([1.2, 1])
            with cc1:
                fig = px.bar(dec, x='decil', y='tasa_real', color='prob_media',
                             color_continuous_scale='Reds',
                             title='Tasa histórica de abandono por decil',
                             text=dec.tasa_real.map('{:.0%}'.format))
                fig.update_yaxes(tickformat='.0%')
                plot(fig)
            with cc2:
                table(dec.assign(prob_media=dec.prob_media.map('{:.1%}'.format),
                                 tasa_real=dec.tasa_real.map('{:.1%}'.format),
                                 lift=dec.lift.map('{:.2f}x'.format)))
        else:
            st.markdown('**Segmentación por deciles de riesgo**')
            dec = s.groupby('decil', observed=True).agg(clientes=('proba', 'size'),
                                                        prob_media=('proba', 'mean')).reset_index()
            table(dec)

        out = s.head(n)[['customerID', 'proba']].copy() if 'customerID' in s else s.head(n)[['proba']].copy()
        out = out.rename(columns={'proba': 'probabilidad_abandono'})
        out['prioridad'] = np.arange(1, len(out) + 1)
        st.download_button('Descargar lista priorizada por riesgo (CSV)',
                           out.to_csv(index=False).encode('utf-8'),
                           file_name='clientes_priorizados.csv', mime='text/csv')
        st.caption('El archivo contiene probabilidades estimadas, no resultados futuros confirmados. '
                   'No distribuir identificadores de clientes reales sin autorización.')

    hyp = read_csv('business_scenarios_HYPOTHETICAL.csv')
    if not hyp.empty:
        with st.expander('Escenarios hipotéticos precalculados en el notebook'):
            table(hyp)
            if {'costo_contacto_USD', 'beneficio_neto_USD', 'eficacia_incremental_hipotetica'}.issubset(hyp.columns):
                plot(px.line(hyp, x='costo_contacto_USD', y='beneficio_neto_USD', color='eficacia_incremental_hipotetica',
                             markers=True, title='Sensibilidad de beneficios'))
    st.markdown('**Validación para implementación empresarial:** confirmar costos de intervención, margen retenido, '
                'consentimiento y protección de datos; evaluar el efecto incremental de las campañas mediante '
                'un experimento controlado antes de afirmar beneficios reales.')
