from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
st.set_page_config(page_title='ChurnGuard AI',page_icon='🛡️',layout='wide')
st.title('ChurnGuard AI | Customer Retention Intelligence')
st.caption('Predicción de abandono, validación y explicabilidad · No constituye una estimación causal de retención.')

@st.cache_resource
def load_model():
    return joblib.load(ART/'models/churn_model.joblib')

def read_csv(name):
    p=ART/'reports'/name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

if not (ART/'models/churn_model.joblib').exists():
    st.warning('Ejecuta primero el notebook desde el directorio del proyecto. Deben generarse artifacts/models y artifacts/reports.')
    st.stop()

model=load_model()
meta=json.loads((ART/'models/metadata.json').read_text(encoding='utf-8'))
tabs=st.tabs(['EDA','Model Performance','Prediction & Explainability','Business Application'])

with tabs[0]:
    st.subheader('Exploración del dataset')
    path=ROOT/'archive/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    if not path.exists():st.info('Coloca el CSV en archive/ para activar el EDA interactivo.')
    else:
        df=pd.read_csv(path);df['TotalCharges']=pd.to_numeric(df.TotalCharges,errors='coerce')
        df['ChurnBinary']=(df.Churn=='Yes').astype(int)
        sel=st.multiselect('Segmentos por contrato',sorted(df.Contract.unique()),default=sorted(df.Contract.unique()))
        frame=df[df.Contract.isin(sel)]
        col1,col2,col3=st.columns(3)
        col1.metric('Clientes',f'{len(frame):,}');col2.metric('Tasa de abandono',f'{frame.ChurnBinary.mean():.1%}');col3.metric('Ingreso mensual promedio',f'${frame.MonthlyCharges.mean():.2f}')
        st.plotly_chart(px.histogram(frame,x='Churn',color='Contract',barmode='group',title='Abandono por contrato'),use_container_width=True)
        st.plotly_chart(px.box(frame,x='Churn',y='tenure',color='Contract',title='Antigüedad y abandono'),use_container_width=True)
        st.plotly_chart(px.scatter(frame,x='tenure',y='MonthlyCharges',color='Churn',title='Antigüedad vs pago mensual'),use_container_width=True)

with tabs[1]:
    st.subheader('Validación y optimización')
    for filename,title in [('baseline_cv.csv','Modelos en validación cruzada'),('optimizer_comparison.csv','Optimización monoobjetivo'),('final_metrics.csv','Evaluación final'),('pareto.csv','Soluciones no dominadas')]:
        dat=read_csv(filename)
        if dat.empty:continue
        st.markdown('**'+title+'**');st.dataframe(dat,use_container_width=True,hide_index=True)
        if filename=='baseline_cv.csv':st.plotly_chart(px.bar(dat,x='modelo',y='roc_auc',title='ROC-AUC CV'),use_container_width=True)
        if filename=='optimizer_comparison.csv':st.plotly_chart(px.scatter(dat,x='segundos',y='roc_auc_cv',color='metodo',size='configuraciones',title='Tiempo vs ROC-AUC CV'),use_container_width=True)
        if filename=='pareto.csv':st.plotly_chart(px.scatter(dat,x='ms_cliente',y='f1_cv',hover_data=['trial'],title='Frente de Pareto'),use_container_width=True)
    pic=ART/'figures/final_evaluation.png'
    if pic.exists():st.image(str(pic),caption='Matriz de confusión y calibración')

with tabs[2]:
    st.subheader('Predicción individual')
    st.caption(f'Modelo: {meta["model_name"]} | Umbral obtenido mediante CV: {meta["threshold"]:.3f}')
    values={}
    with st.form('cliente'):
        numeric=set(meta['numerical_features'])
        for c in meta['columns']:
            if c in numeric:
                default=meta['numeric_defaults'][c]
                values[c]=st.number_input(c,value=float(default),step=1.0 if c=='tenure' else 0.5)
            else:
                opts=meta['categorical_options'][c]
                default=meta['category_defaults'][c]
                values[c]=st.selectbox(c,opts,index=opts.index(default) if default in opts else 0)
        submitted=st.form_submit_button('Analizar riesgo')
    if submitted:
        x=pd.DataFrame([values],columns=meta['columns']);p=float(model.predict_proba(x)[0,1]);label=int(p>=meta['threshold'])
        st.metric('Probabilidad estimada de abandono',f'{p:.1%}',delta='Riesgo priorizado' if label else 'Riesgo no priorizado',delta_color='off')
        st.info('Esta probabilidad es predictiva; una característica influyente no constituye una causa comprobada.')
        try:
            import shap
            transformed=np.asarray(model.named_steps['prep'].transform(x),dtype=float)
            clf=model.named_steps['model'];names=model.named_steps['prep'].get_feature_names_out()
            exp=shap.TreeExplainer(clf)
            sv=exp.shap_values(transformed)
            if isinstance(sv,list):v=sv[1][0]
            else:
                v=np.asarray(sv)
                v=v[0,:,1] if v.ndim==3 else v[0]
            st.markdown('**SHAP: factores principales en esta predicción**')
            st.dataframe(pd.DataFrame({'variable':names,'contribucion_shap':v}).assign(magnitud=lambda d:d.contribucion_shap.abs()).sort_values('magnitud',ascending=False).head(12),hide_index=True)
        except Exception as ex:st.caption(f'SHAP local requiere un modelo árbol compatible: {str(ex)[:160]}')
        try:
            from lime.lime_tabular import LimeTabularExplainer
            path=ROOT/'archive/WA_Fn-UseC_-Telco-Customer-Churn.csv'
            if path.exists():
                df=pd.read_csv(path)
                df['TotalCharges']=pd.to_numeric(df.TotalCharges,errors='coerce')
                sampled=df[meta['columns']].sample(n=min(250,len(df)),random_state=42)
                bg=np.asarray(model.named_steps['prep'].transform(sampled),dtype=float)
                explain=LimeTabularExplainer(bg,feature_names=list(model.named_steps['prep'].get_feature_names_out()),class_names=['No churn','Churn'],mode='classification',random_state=42)
                detail=explain.explain_instance(transformed[0],clf.predict_proba,num_features=10,num_samples=500)
                st.markdown('**LIME: explicación local**');st.dataframe(pd.DataFrame(detail.as_list(),columns=['condición','peso']),hide_index=True)
        except Exception as ex:st.caption(f'LIME no disponible en esta sesión: {str(ex)[:160]}')

with tabs[3]:
    st.subheader('Simulación comercial: escenarios hipotéticos')
    dat=read_csv('business_scenarios_HYPOTHETICAL.csv')
    st.warning('Los resultados de negocio son simulaciones hipotéticas, NO ingresos realizados ni evidencia de efectividad causal.')
    if not dat.empty:
        st.dataframe(dat,use_container_width=True,hide_index=True)
        st.plotly_chart(px.line(dat,x='costo_contacto_USD',y='beneficio_neto_USD',color='eficacia_incremental_hipotetica',markers=True,title='Sensibilidad de beneficios'),use_container_width=True)
    st.markdown('El comprador debe validar margen retenido, costo de intervención, consentimiento, privacidad y éxito incremental con una campaña aleatorizada.')
