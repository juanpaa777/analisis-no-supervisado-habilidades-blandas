"""
core/generador_datos.py
Genera un dataset sintetico con perfiles ocultos y ruido gaussiano.
"""
import numpy as np
import pandas as pd
from core.config import TODAS_LAS_COLUMNAS, PARAMS_PERFILES


def generar_dataset(n_empleados=500, seed=42):
    """
    Genera un DataFrame con evaluaciones sinteticas de habilidades blandas.

    Se crean 3 perfiles ocultos:
      A - Lideres integrales (altos en todo)
      B - Tecnicos reservados (altos en adaptacion, medios/bajos en comunicacion)
      C - En desarrollo (medios en comunicacion/colaboracion, bajos en resolucion)

    Retorna: (DataFrame, dict con conteo de perfiles)
    """
    np.random.seed(seed)
    proporciones = [0.30, 0.40, 0.30]
    perfiles = np.random.choice(['A', 'B', 'C'], size=n_empleados, p=proporciones)

    datos = {'id_empleado': list(range(1, n_empleados + 1))}

    for col in TODAS_LAS_COLUMNAS:
        valores = []
        for perfil in perfiles:
            media, std = PARAMS_PERFILES[perfil][col]
            valor = np.random.normal(loc=media, scale=std)
            valor = round(np.clip(valor, 1.0, 10.0), 1)
            valores.append(valor)
        datos[col] = valores

    df = pd.DataFrame(datos)

    unique, counts = np.unique(perfiles, return_counts=True)
    info_perfiles = {p: int(c) for p, c in zip(unique, counts)}

    return df, info_perfiles
