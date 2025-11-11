GeoLiquefAI: Evaluador de Riesgo de Licuefacción

GeoLiquefAI es una aplicación web interactiva desarrollada en Streamlit que sirve como herramienta de evaluación preliminar (screening) para el potencial de licuefacción de suelos. La aplicación fusiona un método geotécnico tradicional (Seed & Idriss) con un modelo de Inteligencia Artificial (Random Forest) para ofrecer un diagnóstico dual.

La herramienta permite a los ingenieros y estudiantes ingresar parámetros geotécnicos y sísmicos de un sitio y recibir instantáneamente dos métricas clave:

Factor de Seguridad (FS): Calculado con el método tradicional.

Probabilidad de Licuefacción (%): Predicha por el modelo de IA.

Además, la aplicación utiliza SHAP (SHapley Additive exPlanations) para explicar la predicción de la IA, mostrando qué factores (SPT, a_max, etc.) influyeron más en el resultado.

Características Principales

Análisis Dual: Compara directamente el FS tradicional con la probabilidad de la IA.

Interfaz Interactiva: Una aplicación de una sola página, fácil de usar y rápida.

Explicabilidad (XAI): Un gráfico de cascada (waterfall) de SHAP muestra por qué la IA ha tomado su decisión.

Interpretación de Resultados: La aplicación proporciona recomendaciones claras basadas en la combinación de ambos resultados.

Listo para Imprimir: La página de resultados está optimizada para imprimirse (Ctrl+P) y adjuntarse a informes.

Barra Lateral Informativa: Provee contexto sobre la metodología y las limitaciones de la herramienta.

Stack Tecnológico

Lenguaje: Python

Framework Web: Streamlit

Machine Learning: Scikit-learn (RandomForestClassifier)

Explicabilidad (XAI): SHAP

Análisis de Datos: Pandas y NumPy

Visualización: Matplotlib

Estructura del Proyecto

GeoLiquefAI/
│
├── app.py                  # Aplicación principal de Streamlit
├── save_model.py            # Script para entrenar y guardar el modelo
├── traditional_method.py    # Lógica para el cálculo del FS tradicional
│
├── mmc2.csv                  # Dataset de entrenamiento
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

Use el formulario principal para ingresar los 6 parámetros de la IA y los 2 adicionales para el método tradicional.

Haga clic en el botón "Analizar Riesgo de Licuefacción".

Revise los resultados, que aparecerán debajo del formulario:

Resumen de Datos de Entrada: Los valores que ingresó (se imprimen).

Resultados en Métricas: Los indicadores con el FS y la Probabilidad.

Interpretación: Texto que explica qué significa la combinación de resultados.

Gráfico SHAP: El análisis de explicabilidad de la IA.

Para imprimir, use Control + P (o Cmd + P).

Limitaciones

Esta herramienta es un software educativo y de evaluación preliminar. No reemplaza un análisis geotécnico detallado realizado por un ingeniero calificado ni debe usarse como la única base para el diseño de ingeniería.