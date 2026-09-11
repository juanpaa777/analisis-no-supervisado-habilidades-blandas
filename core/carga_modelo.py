"""
core/carga_modelo.py
Funciones para CARGAR y APLICAR modelos pre-entrenados.
Este modulo NUNCA entrena — solo lee el .pkl y aplica predict.
"""
import joblib


def cargar_modelos(ruta_pkl):
    """Carga el diccionario de modelos desde un archivo .pkl."""
    return joblib.load(ruta_pkl)


def buscar_modelo(modelos, habilidades_seleccionadas):
    """
    Busca el modelo correspondiente a una combinacion de habilidades.
    Compara por conjunto (set) para ser independiente del orden de seleccion.

    Retorna: dict con info del modelo, o None si no se encuentra.
    """
    for clave in modelos:
        if set(clave) == set(habilidades_seleccionadas):
            return modelos[clave]
    return None


def aplicar_modelo(pipeline, X):
    """
    Aplica un pipeline pre-entrenado a datos nuevos.
    Retorna las etiquetas de cluster asignadas.
    """
    return pipeline.predict(X)
