"""
core/entrenamiento.py
Entrena un pipeline (StandardScaler -> KMeans) para cada una de las 11
combinaciones posibles de habilidades.
Guarda historial completo de silhouette por cada k probado.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from core.config import (
    HABILIDADES_MAP, TODAS_COMBINACIONES, K_MIN, K_MAX
)


def entrenar_todos_los_modelos(df):
    """
    Entrena 11 modelos (uno por combinacion de 2-4 habilidades).
    Pipeline: StandardScaler -> KMeans.
    Elige k optimo con silhouette_score sobre datos escalados.

    Retorna:
        modelos: dict con clave=tuple(habilidades), valor=dict con:
            - pipeline: Pipeline entrenado con k optimo
            - k_optimo: int
            - silhouette_score: float
            - columnas: list de columnas usadas
            - scores_por_k: dict {k: silhouette_score} para TODOS los k probados
            - n_registros: int (cantidad de registros de entrenamiento)
        log_text: str con el log del entrenamiento
    """
    modelos = {}
    log_lines = []
    log_lines.append(f"Total de combinaciones a entrenar: {len(TODAS_COMBINACIONES)}")
    log_lines.append("Pipeline: StandardScaler -> KMeans")
    log_lines.append(f"Rango de k evaluado: {K_MIN} a {K_MAX}")
    log_lines.append("=" * 60)

    for combo in TODAS_COMBINACIONES:
        columnas = []
        for hab in combo:
            columnas.extend(HABILIDADES_MAP[hab])

        X = df[columnas].values
        mejor_k = K_MIN
        mejor_score = -1
        resultados_k = {}

        for k in range(K_MIN, K_MAX + 1):
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('kmeans', KMeans(n_clusters=k, random_state=42, n_init=10))
            ])
            pipeline.fit(X)

            X_escalado = pipeline.named_steps['scaler'].transform(X)
            etiquetas = pipeline.named_steps['kmeans'].labels_
            score = silhouette_score(X_escalado, etiquetas)
            resultados_k[k] = round(score, 4)

            if score > mejor_score:
                mejor_score = score
                mejor_k = k

        # Pipeline final con k optimo
        pipeline_final = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=mejor_k, random_state=42, n_init=10))
        ])
        pipeline_final.fit(X)

        clave = tuple(combo)
        modelos[clave] = {
            'pipeline': pipeline_final,
            'k_optimo': mejor_k,
            'silhouette_score': round(mejor_score, 4),
            'columnas': columnas,
            'scores_por_k': resultados_k,
            'n_registros': len(X),
        }

        habs_str = " + ".join(combo)
        log_lines.append(f"\n{habs_str}")
        log_lines.append(f"  Columnas: {columnas}")
        for k_val, s_val in resultados_k.items():
            marca = " << OPTIMO" if k_val == mejor_k else ""
            log_lines.append(f"  k={k_val}: silhouette={s_val}{marca}")
        log_lines.append(f"  -> k optimo = {mejor_k}, silhouette = {mejor_score:.4f}")

    log_lines.append("\n" + "=" * 60)
    log_lines.append(f"Total de modelos entrenados: {len(modelos)}")

    return modelos, "\n".join(log_lines)
