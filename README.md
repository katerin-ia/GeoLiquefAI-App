GeoLiquefAI: Evaluador de Riesgo de Licuefacción

Autora

Este proyecto fue desarrollado por Katerin De La Cruz Alvarez.

Resumen

GeoLiquefAI es una aplicación web en Streamlit para la evaluación preliminar del riesgo de licuefacción de suelos.

La herramienta compara dos métodos de análisis:

Factor de Seguridad (FS): Calculado con el método tradicional (Seed & Idriss).

Probabilidad de Licuefacción (%): Predicha por un modelo de Inteligencia Artificial (Random Forest).

La aplicación utiliza SHAP (SHapley Additive exPlanations) para interpretar la predicción de la IA, mostrando la influencia de cada parámetro en el resultado.

Características

Análisis Dual: Compara el FS tradicional contra la probabilidad de la IA.

Interfaz de Pestañas: Separa el ingreso de datos de la visualización de resultados.

Explicabilidad (XAI): Gráfico de cascada (waterfall) de SHAP para interpretar el modelo.

Interpretación de Resultados: Texto de recomendación basado en los hallazgos.

Optimizado para Impresión: La pestaña de resultados se puede imprimir (Ctrl+P) para informes.

Contexto Metodológico: Barra lateral con información sobre el modelo y sus limitaciones.

Stack Tecnológico

Lenguaje: Python

Framework Web: Streamlit

Machine Learning: Scikit-learn (RandomForestClassifier)

Explicabilidad (XAI): SHAP

Análisis de Datos: Pandas y NumPy

Visualización: Matplotlib

Fuente de Datos (Dataset)

El modelo de Inteligencia Artificial fue entrenado utilizando el conjunto de datos mmc2.csv, que es un compendio de estudios de caso de licuefacción (SPT-based). Este conjunto de datos proviene de la siguiente publicación de investigación:

Cetin, K. O., Seed, R. B., Kayen, R. E., Moss, R. E. S., Bilge, H. T., Ilgac, M., & Chowdhury, K. (2018). The use of the SPT-based seismic soil liquefaction triggering evaluation methodology in engineering hazard assessments. MethodsX, 5, 1556-1575. https://doi.org/10.1016/j.mex.2018.11.016

Estructura del Proyecto

GeoLiquefAI/
│
├── app.py                  # Aplicación principal de Streamlit
├── save_model.py            # Script para entrenar y guardar el modelo
├── traditional_method.py    # Lógica para el cálculo del FS tradicional
│
├── mmc2.csv                  # Dataset de entrenamiento (Cetin et al., 2018)
├── requirements.txt          # Lista de dependencias de Python
│
├── modelo_rf.joblib          # Archivo del modelo de IA (generado)
└── scaler.joblib             # Archivo del scaler (generado)


Instalación y Ejecución

Siga estos 3 pasos para ejecutar la aplicación en su máquina local.

Paso 1: Configurar el Entorno

Se recomienda crear un entorno virtual para este proyecto.

# Clone este repositorio (si aplica)
# ...

# Cree un entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows use: venv\Scripts\activate

# Instale todas las dependencias
pip install -r requirements.txt


Paso 2: Entrenar el Modelo

Antes de ejecutar la aplicación, debe generar los archivos modelo_rf.joblib y scaler.joblib. El script save_model.py lo hace automáticamente usando los datos de mmc2.csv.

Nota: Solo necesita hacer esto una vez.

python save_model.py


Debería ver un mensaje de "¡Entrenamiento y guardado completados!" en su terminal.

Paso 3: Ejecutar la Aplicación

Una vez que los modelos estén generados, inicie la aplicación de Streamlit:

streamlit run app.py


Streamlit abrirá automáticamente la aplicación en su navegador.

Modo de Uso

Abra la aplicación en su navegador.

En la pestaña "Ingreso de Datos", complete el formulario con los 6 parámetros de la IA y los 2 adicionales para el método tradicional.

Haga clic en el botón "Analizar Riesgo de Licuefacción".

La aplicación procesará los datos y le indicará que el análisis está listo.

Vaya a la pestaña "Resultados del Análisis" para revisar el reporte completo, el cual incluye:

Resumen de Datos de Entrada (para impresión).

Resultados en Métricas (FS y Probabilidad).

Interpretación y Recomendaciones.

Gráfico SHAP de explicabilidad.

Para imprimir el reporte, use Control + P (o Cmd + P) en la pestaña de resultados.

Limitaciones

Esta herramienta es un software educativo y de evaluación preliminar. No reemplaza un análisis geotécnico detallado realizado por un ingeniero calificado ni debe usarse como la única base para el diseño de ingeniería.
