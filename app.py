import streamlit as st
import joblib
import numpy as np
import pandas as pd  # Importar Pandas
import shap
import matplotlib.pyplot as plt
from traditional_method import calculate_traditional_fs
from datetime import datetime
import time

# --- 1. Configuración de la Página ---
st.set_page_config(
    page_title="GeoLiquefAI - Evaluador de Licuefacción",
    page_icon="🌎",
    layout="wide",
)

# --- 2. Funciones de Carga y Clasificación ---
@st.cache_resource
def load_artifacts():
    """
    Carga el modelo de ML y el scaler desde los archivos.
    Usa cache para que solo se carguen una vez.
    """
    try:
        model = joblib.load("modelo_rf.joblib")
        scaler = joblib.load("scaler.joblib")
        return model, scaler
    except FileNotFoundError:
        st.error(
            "⚠️ No se encontraron los archivos del modelo ('modelo_rf.joblib' o 'scaler.joblib'). "
            "Por favor, ejecuta `python save_model.py` para generarlos."
        )
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar artefactos: {e}")
        st.stop()

def classify_risk(prob):
    """
    Clasifica la probabilidad de licuefacción (0.0 a 1.0) en etiquetas de riesgo
    basadas en la escala provista.
    """
    # La probabilidad (prob) debe estar entre 0.0 y 1.0

    if prob >= 0.80:
        return "Riesgo Muy Alto" 
    elif prob >= 0.50:
        return "Riesgo Alto" 
    elif prob >= 0.20:
        return "Riesgo Moderado"
    else: # Esto cubre el rango de 0.00 a 0.199... (0-20%)
        return "Riesgo Bajo"

def classify_fs(fs):
    """Clasifica el Factor de Seguridad (FS) en etiquetas de riesgo."""
    if fs is None:
        return "Error"
    if fs < 1.0:
        return "Licuefactible"
    elif fs < 1.3:
        return "Licuefacción Marginal"
    else:
        return "No Licuefactible"

# --- 3. Función Principal de la App ---
def main():
    # --- CSS para Arreglar Impresión (v19) ---
    st.markdown(
        """
        <style>
        @media print {
            /* Ocultar elementos de la UI de Streamlit al imprimir */
            .stApp > header, .stApp .e10yg2by1, .stApp .e10yg2by3, .stTabs .st-emotion-cache-1gpf04l {
                display: none !important;
            }
            /* Forzar fondo blanco y texto negro */
            .stApp, .main .block-container {
                background-color: white !important;
                color: black !important;
            }
            /* Asegurar que el contenido principal sea visible */
            .main .block-container {
                display: block !important;
                width: 100% !important;
                padding: 0 !important;
            }
            /* Estilos de texto explícitos */
            body, h1, h2, h3, h4, h5, h6, p, div, span, .stMetric, .stMarkdown {
                color: black !important;
                background-color: white !important;
            }
            /* Ocultar botones y elementos interactivos */
            .stButton, .stNumberInput, .stForm {
                display: none !important;
            }
            /* Mostrar el gráfico SHAP (si es una imagen) */
            .stImage, .stPlotlyChart {
                display: block !important;
            }
        }
        
        /* Reducir tamaño de fuentes de sub-cabeceras */
        h4 {
            font-size: 1.5rem !important;
            font-weight: 600 !important;
        }
        h5 {
            font-size: 1.25rem !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Cargar modelo y scaler
    model, scaler = load_artifacts()
    if model is None or scaler is None:
        return

    # --- Barra Lateral (Sidebar) ---
    st.sidebar.header("Acerca de GeoLiquefAI")
    st.sidebar.info(
        "**GeoLiquefAI** es una herramienta de **evaluación preliminar** que combina " # CAMBIO: "cribado" -> "evaluación preliminar"
        "la ingeniería geotécnica tradicional con la inteligencia artificial "
        "para evaluar el potencial de licuefacción del suelo."
    )
    st.sidebar.markdown("### Metodología")
    st.sidebar.markdown(
        """
        1.  **Método Tradicional:** Calcula el Factor de Seguridad (FS) 
            basado en el método simplificado (Seed & Idriss).
        2.  **Inteligencia Artificial:** Un modelo **Random Forest** entrenado 
            en un historial de casos predice la *probabilidad* de licuefacción.
        """
    )
    st.sidebar.warning(
        "Esta herramienta no reemplaza un análisis geotécnico detallado "
        "realizado por un ingeniero calificado. Los resultados son "
        "referenciales."
    )

    # --- Título Principal ---
    st.title("🌎 GeoLiquefAI: Evaluador de Riesgo de Licuefacción")
    st.markdown(
        "Plataforma dual para evaluar el potencial de licuefacción usando **Inteligencia Artificial** (Random Forest) y el **Método Tradicional** (Seed & Idriss)."
    )

    # --- CAMBIO: Diseño de Pestañas (v20) ---
    tab1, tab2 = st.tabs(["📝 Ingreso de Datos", "📊 Resultados del Análisis"])

    # --- PESTAÑA 1: INGRESO DE DATOS ---
    with tab1:
        st.markdown("---")
        st.markdown("<h4>Ingrese los parámetros del sitio</h4>", unsafe_allow_html=True) # CAMBIO: Tamaño de fuente
        st.info(
            "Complete los 6 parámetros para el modelo de IA y los 2 parámetros adicionales (en gris) para el método tradicional."
        )

        with st.form(key="liquefaction_form"):
            input_dict = {}

            st.markdown("<h5>1. Parámetros del Suelo</h5>", unsafe_allow_html=True) # CAMBIO: Tamaño de fuente
            col1, col2, col3 = st.columns(3)
            with col1:
                input_dict["N1_60_cs"] = st.number_input(
                    "Golpes (N1)60cs", min_value=1.0, max_value=60.0, value=15.0, step=0.5,
                    help="Número de golpes SPT corregido."
                )
            with col2:
                input_dict["FC"] = st.number_input(
                    "Contenido de Finos (FC) [%]", min_value=0.0, max_value=100.0, value=10.0, step=0.5,
                    help="Porcentaje de suelo que pasa la malla #200."
                )
            with col3:
                input_dict["D50"] = st.number_input(
                    "Diámetro medio (D50) [mm]", min_value=0.01, max_value=5.0, value=0.25, step=0.01,
                    help="Diámetro medio de las partículas."
                )
            
            st.markdown("<h5>2. Parámetros Sísmicos y de Esfuerzo</h5>", unsafe_allow_html=True) # CAMBIO: Tamaño de fuente
            col4, col5 = st.columns(2)
            with col4:
                input_dict["a_max"] = st.number_input(
                    "Aceleración Máxima (a_max) [g]", min_value=0.01, max_value=2.0, value=0.4, step=0.01,
                    help="Aceleración horizontal máxima en la superficie."
                )
                input_dict["Mw"] = st.number_input(
                    "Magnitud del Sismo (Mw)", min_value=4.0, max_value=10.0, value=7.5, step=0.1,
                    help="Magnitud de momento del sismo."
                )

            with col5:
                input_dict["estres_v_ef"] = st.number_input(
                    "Esfuerzo Efectivo (σ'v) [kPa]", min_value=1.0, max_value=1000.0, value=100.0, step=1.0,
                    help="Esfuerzo vertical efectivo en el punto de análisis. USADO POR AMBOS MÉTODOS."
                )
            
            st.markdown("<h5>3. Parámetros Adicionales (Solo para Método Tradicional)</h5>", unsafe_allow_html=True) # CAMBIO: Tamaño de fuente
            col6, col7 = st.columns(2)
            with col6:
                input_dict["z_m"] = st.number_input(
                    "Profundidad (z) [m]", min_value=0.5, max_value=50.0, value=10.0, step=0.5,
                    help="Profundidad del estrato a analizar."
                )
            with col7:
                input_dict["estres_v_total"] = st.number_input(
                    "Esfuerzo Total (σv) [kPa]", min_value=1.0, max_value=2000.0, value=180.0, step=1.0,
                    help="Esfuerzo vertical total en el punto de análisis."
                )

            st.markdown("---")
            submit_button = st.form_submit_button(
                label="Analizar Riesgo de Licuefacción",
                use_container_width=True
            )

        # --- Lógica de cálculo (dentro de la Pestaña 1) ---
        if submit_button:
            with st.spinner("Realizando cálculos... 🤖"):
                # Conversión de kPa a PSF para el modelo de IA
                KPA_TO_PSF = 20.8854
                estres_v_ef_psf_para_ia = input_dict["estres_v_ef"] * KPA_TO_PSF
                
                # --- 1. Predicción del Modelo de IA ---
                feature_order_ia = ["N1_60_cs", "FC", "D50", "a_max", "estres_v_ef", "Mw"]
                input_dict_ia = input_dict.copy()
                input_dict_ia["estres_v_ef"] = estres_v_ef_psf_para_ia
                
                proba_ia = None
                risk_label_ia = "Error"
                x_scaled = None
                
                try:
                    x_ia = np.array([[input_dict_ia[k] for k in feature_order_ia]])
                    x_scaled = scaler.transform(x_ia)
                    proba_ia = float(model.predict_proba(x_scaled)[0][1])
                    risk_label_ia = classify_risk(proba_ia)
                except Exception as e:
                    st.error(f"Error en la predicción de IA: {e}")

                # --- 2. Cálculo del Método Tradicional ---
                fs_trad = None
                risk_label_trad = "Error"
                trad_results = {}
                try:
                    trad_results = calculate_traditional_fs(input_dict)
                    fs_trad = trad_results.get("FS_trad")
                    risk_label_trad = classify_fs(fs_trad)
                except Exception as e:
                    st.error(f"Error en el cálculo tradicional: {e}")

                # --- 3. Gráfico SHAP ---
                shap_fig = None
                if x_scaled is not None:
                    try:
                        explainer = shap.TreeExplainer(model)
                        x_scaled_df = pd.DataFrame(x_scaled, columns=feature_order_ia)
                        explanation = explainer(x_scaled_df)
                        explanation_class_1 = explanation[0, :, 1]
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        shap.waterfall_plot(
                            explanation_class_1,
                            max_display=len(feature_order_ia),
                            show=False
                        )
                        shap_fig = fig
                    except Exception as e:
                        st.error(f"Error en la generación del gráfico SHAP: {e}")
                
                # --- Guardar TODO en st.session_state ---
                st.session_state['analysis_complete'] = True
                st.session_state['results'] = {
                    "input_dict": input_dict,
                    "proba_ia": proba_ia,
                    "risk_label_ia": risk_label_ia,
                    "fs_trad": fs_trad,
                    "risk_label_trad": risk_label_trad,
                    "trad_results": trad_results,
                    "shap_fig": shap_fig
                }
            
            st.success("¡Análisis completado! Revise la pestaña 'Resultados del Análisis'.")


    # --- PESTAÑA 2: RESULTADOS DEL ANÁLISIS ---
    with tab2:
        st.markdown("---")
        # Verificar si hay resultados en la sesión
        if not st.session_state.get('analysis_complete', False):
            st.info("Presione 'Analizar' en la pestaña 'Ingreso de Datos' para ver los resultados.")
        else:
            # Si hay resultados, cargarlos desde la sesión
            results = st.session_state['results']
            input_dict = results["input_dict"]
            proba_ia = results["proba_ia"]
            risk_label_ia = results["risk_label_ia"]
            fs_trad = results["fs_trad"]
            risk_label_trad = results["risk_label_trad"]
            trad_results = results["trad_results"]
            shap_fig = results["shap_fig"]

            st.markdown("<h4>Resultados del Análisis</h4>", unsafe_allow_html=True) # CAMBIO: Tamaño de fuente

            # --- Resumen de Datos de Entrada (para Impresión) ---
            st.markdown("<h5>Resumen de Datos de Entrada</h5>", unsafe_allow_html=True) # CAMBIO: Tamaño de fuente
            col_in_1, col_in_2, col_in_3 = st.columns(3)
            with col_in_1:
                # CAMBIO: Quitado los `backticks` para arreglar fondo de impresión
                st.markdown(f"**Golpes (N1)60cs:** {input_dict.get('N1_60_cs', 'N/A')}")
                st.markdown(f"**Contenido de Finos (FC):** {input_dict.get('FC', 'N/A')} %")
                st.markdown(f"**Diámetro medio (D50):** {input_dict.get('D50', 'N/A')} mm")
            with col_in_2:
                st.markdown(f"**Aceleración Máxima (a_max):** {input_dict.get('a_max', 'N/A')} g")
                st.markdown(f"**Magnitud del Sismo (Mw):** {input_dict.get('Mw', 'N/A')}")
                st.markdown(f"**Esfuerzo Efectivo (σ'v):** {input_dict.get('estres_v_ef', 'N/A')} kPa")
            with col_in_3:
                st.markdown(f"**Profundidad (z):** {input_dict.get('z_m', 'N/A')} m")
                st.markdown(f"**Esfuerzo Total (σv):** {input_dict.get('estres_v_total', 'N/A')} kPa")

            st.markdown("---") 

            # --- Mostrar Resultados en Métricas ---
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown("<h5 style='text-align: center;'>Predicción IA (Random Forest)</h5>", unsafe_allow_html=True)
                if proba_ia is not None:
                    st.metric(
                        label="Probabilidad de Licuefacción",
                        value=f"{proba_ia * 100:.2f} %",
                        delta=risk_label_ia,
                    )
                    st.progress(proba_ia)
                else:
                    st.error("No se pudo calcular la predicción de IA.")
            
            with res_col2:
                st.markdown("<h5 style='text-align: center;'>Método Tradicional (Seed & Idriss)</h5>", unsafe_allow_html=True)
                if fs_trad is not None:
                    st.metric(
                        label="Factor de Seguridad (FS)",
                        value=f"{fs_trad:.3f}",
                        delta=risk_label_trad
                    )
                    st.write(f"**CSR (Solicitación):** {trad_results.get('CSR', 0):.3f}")
                    st.write(f"**CRR (Resistencia):** {trad_results.get('CRR_adj', 0):.3f}")
                else:
                    st.error("No se pudo calcular el FS tradicional.")

            # --- Sección de Interpretación ---
            st.markdown("---")
            st.markdown("<h4>Interpretación y Recomendaciones</h4>", unsafe_allow_html=True) # CAMBIO: Tamaño de fuente
            
            inter_col1, inter_col2 = st.columns(2)
            
            with inter_col1:
                st.markdown("<h5>Método Tradicional (FS)</h5>", unsafe_allow_html=True) # CAMBIO: Tamaño de fuente
                if fs_trad is not None:
                    if risk_label_trad == "Licuefactible":
                        st.error(f"**Resultado: {risk_label_trad} (FS = {fs_trad:.3f})**\n\nEl FS es menor a 1.0. El método tradicional indica que el suelo fallará.")
                    elif risk_label_trad == "Licuefacción Marginal":
                        st.warning(f"**Resultado: {risk_label_trad} (FS = {fs_trad:.3f})**\n\nEl FS está en una zona de incertidumbre (entre 1.0 y 1.3). Se requiere precaución y análisis adicional.")
                    else:
                        st.success(f"**Resultado: {risk_label_trad} (FS = {fs_trad:.3f})**\n\nEl FS es mayor a 1.3. El método tradicional indica que el suelo es estable.")
                else:
                    st.error("No se pudo calcular el FS.")

            with inter_col2:
                st.markdown("<h5>Inteligencia Artificial (IA)</h5>", unsafe_allow_html=True) # CAMBIO: Tamaño de fuente
                if proba_ia is not None:
                    if risk_label_ia == "Alto Riesgo":
                        st.error(f"**Resultado: {risk_label_ia} ({proba_ia*100:.1f}%)**\n\nLa IA tiene alta confianza de que este escenario es peligroso, basándose en patrones de casos históricos de falla.")
                    elif risk_label_ia == "Riesgo Moderado":
                        st.warning(f"**Resultado: {risk_label_ia} ({proba_ia*100:.1f}%)**\n\nLa IA no está segura. Los parámetros coinciden tanto con casos de falla como de no-falla. Se recomienda precaución.")
                    else:
                        st.success(f"**Resultado: {risk_label_ia} ({proba_ia*100:.1f}%)**\n\nLa IA tiene alta confianza de que este escenario es seguro.")
                else:
                    st.error("No se pudo calcular la predicción de IA.")

            # --- Mostrar Gráfico SHAP ---
            st.markdown("---")
            st.markdown("<h4>Explicación de la Predicción de IA (Análisis SHAP)</h4>", unsafe_allow_html=True) # CAMBIO: Tamaño de fuente
            st.write("Este gráfico de 'cascada' (waterfall) muestra cómo cada factor 'empujó' la predicción de la IA, "
                     "desde el valor base (riesgo promedio) hasta la predicción final para este caso.")
            
            if shap_fig is not None:
                st.pyplot(shap_fig, use_container_width=True)
                plt.close(shap_fig) # Cerrar la figura después de mostrarla
            else:
                st.warning("No se pudo generar el gráfico SHAP (posiblemente debido a un error en la predicción de IA).")


if __name__ == "__main__":
    main()
