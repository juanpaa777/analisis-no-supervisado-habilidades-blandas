"""
entrenar_modelo.py
-------------------
Script de pre-entrenamiento. Se ejecuta ANTES de la exposicion.

1. Genera el dataset sintetico y lo guarda en data/
2. Entrena los 11 modelos (StandardScaler -> KMeans) y los guarda en modelos/

Uso:
    python entrenar_modelo.py
"""
import os
import pandas as pd
import joblib

from core.config import ARCHIVO_DATASET, ARCHIVO_MODELOS, DATA_DIR, MODELOS_DIR
from core.generador_datos import generar_dataset
from core.entrenamiento import entrenar_todos_los_modelos


def main():
    # Crear directorios si no existen
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELOS_DIR, exist_ok=True)

    # Paso 1: Generar dataset
    print("=" * 60)
    print("PASO 1: Generando dataset sintetico...")
    print("=" * 60)
    df, info_perfiles = generar_dataset(n_empleados=500, seed=42)
    df.to_csv(ARCHIVO_DATASET, index=False)
    print(f"Dataset guardado en: {ARCHIVO_DATASET}")
    print(f"Registros: {len(df)}")
    print("Perfiles ocultos (no se guardan en el CSV):")
    nombres = {'A': 'Lideres Integrales', 'B': 'Tecnicos Reservados', 'C': 'En Desarrollo'}
    for p, c in info_perfiles.items():
        print(f"  Perfil {p} ({nombres.get(p, p)}): {c} empleados ({c/len(df)*100:.0f}%)")

    # Paso 2: Entrenar modelos
    print("\n" + "=" * 60)
    print("PASO 2: Entrenando 11 modelos...")
    print("=" * 60)
    modelos, log_text = entrenar_todos_los_modelos(df)
    joblib.dump(modelos, ARCHIVO_MODELOS)

    print(log_text)
    print(f"\nModelos guardados en: {ARCHIVO_MODELOS}")



if __name__ == '__main__':
    main()
