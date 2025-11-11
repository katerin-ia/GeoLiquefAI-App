import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# --- CORRECCIÓN: Apuntar al archivo CSV y usar pd.read_csv ---
DATA_FILE = "mmc2.csv"  # El nombre de tu archivo CSV

def load_and_prepare_data(path: str):
    # --- CORRECCIÓN: Usar pd.read_csv ---
    try:
        df_raw = pd.read_csv(path, sep=';')
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de datos en la ruta: {path}")
        print("Asegúrate de que 'mmc2.xlsx - CETIN_2018.csv' esté en el mismo directorio.")
        return None, None
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return None, None
        
    df_raw.columns = df_raw.columns.str.strip()

    column_mapping = {
        "(N1) 60": "N1_60_cs",
        "FC %": "FC",
        "D50 (mm)": "D50",
        "a max (g)": "a_max",
        "sv' (psf)": "estres_v_ef",
        "Magnitude for KMw": "Mw",
        "Liquefied?": "liquefaction",
    }
    
    # Verificar si todas las columnas necesarias existen
    missing_cols = [col for col in column_mapping if col not in df_raw.columns]
    if missing_cols:
        print(f"Error: Faltan columnas en el CSV: {missing_cols}")
        return None, None

    df = df_raw.rename(columns=column_mapping)

    if "liquefaction" not in df.columns:
        raise ValueError("No se encontró la columna 'Liquefied?' / 'liquefaction' en el archivo de datos.")

    df["liquefaction"] = df["liquefaction"].apply(
        lambda x: 1 if str(x).strip().lower() == "yes" else 0
    )

    features = ["N1_60_cs", "FC", "D50", "a_max", "estres_v_ef", "Mw"]
    
    # --- Limpieza de datos robusta ---
    # Eliminar filas donde falta alguna característica esencial
    df_clean = df[features + ["liquefaction"]].copy()
    
    # Convertir a numérico, forzando errores a NaN
    for col in features:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Contar filas antes y después de eliminar NaN
    rows_before = df_clean.shape[0]
    df_clean = df_clean.dropna()
    rows_after = df_clean.shape[0]
    
    print(f"Filas leídas: {rows_before}. Filas usadas (después de limpiar NaN): {rows_after}.")

    if rows_after == 0:
        print("Error: No quedaron datos después de la limpieza. Revisa tu archivo CSV.")
        return None, None

    X = df_clean[features]
    y = df_clean["liquefaction"]
    
    return X, y

def main():
    X, y = load_and_prepare_data(DATA_FILE)
    if X is None or y is None:
        print("Falló la carga de datos. Abortando.")
        return

    features = ["N1_60_cs", "FC", "D50", "a_max", "estres_v_ef", "Mw"]
    X = X[features] # Asegurar el orden de las columnas

    print(f"Entrenando con {X.shape[0]} muestras. Features: {features}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    print("Entrenando Random Forest...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy en test: {acc * 100:.2f}%")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nGuardando modelo en 'modelo_rf.joblib'")
    joblib.dump(model, "modelo_rf.joblib")

    print("Guardando scaler en 'scaler.joblib'")
    joblib.dump(scaler, "scaler.joblib")

    print("\n¡Entrenamiento y guardado completados!")

if __name__ == "__main__":
    main()