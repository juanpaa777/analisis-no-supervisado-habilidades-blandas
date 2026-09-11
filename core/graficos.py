"""
core/graficos.py
Todas las funciones de graficas (matplotlib).
Cada funcion retorna un objeto Figure listo para mostrar o guardar.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def grafico_tamano_clusters(conteo_clusters, k_opt):
    """
    Grafico de barras con el tamano (numero de empleados) de cada cluster.
    conteo_clusters: pd.Series con index=cluster_id, values=conteo.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    colores = plt.cm.viridis(np.linspace(0.2, 0.8, len(conteo_clusters)))
    barras = ax.bar(
        [f"Cluster {i}" for i in conteo_clusters.index],
        conteo_clusters.values,
        color=colores, edgecolor='black', linewidth=0.5
    )
    for barra, valor in zip(barras, conteo_clusters.values):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 2,
            str(valor), ha='center', va='bottom', fontweight='bold'
        )
    ax.set_title(f"Tamano de cada Cluster (k={k_opt})", fontsize=12)
    ax.set_ylabel("Numero de empleados")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def grafico_perfil_clusters(perfil_cluster, columnas_sel, k_opt):
    """
    Grafico de barras agrupadas con el perfil promedio de cada cluster
    por cada indicador seleccionado.
    """
    fig, ax = plt.subplots(figsize=(max(10, len(columnas_sel) * 1.2), 5))
    n_clusters = k_opt
    n_indicadores = len(columnas_sel)
    x_pos = np.arange(n_indicadores)
    ancho = 0.8 / n_clusters

    colores = plt.cm.viridis(np.linspace(0.2, 0.8, n_clusters))

    for i in range(n_clusters):
        if i in perfil_cluster.index:
            valores = perfil_cluster.loc[i, columnas_sel].values
        else:
            valores = np.zeros(n_indicadores)
        offset = (i - n_clusters / 2 + 0.5) * ancho
        ax.bar(
            x_pos + offset, valores, ancho,
            label=f"Cluster {i}", color=colores[i],
            edgecolor='black', linewidth=0.3
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(columnas_sel, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Puntuacion promedio (1-10)")
    ax.set_title("Perfil Promedio por Cluster e Indicador", fontsize=12)
    ax.legend()
    ax.set_ylim(0, 10.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def grafico_silhouette_vs_k(scores_por_k, k_optimo):
    """
    Grafico de linea: silhouette score vs. k probado.
    Marca con un punto rojo el k optimo seleccionado.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = list(scores_por_k.keys())
    scores = list(scores_por_k.values())

    ax.plot(ks, scores, 'o-', color='#2c3e50', linewidth=2, markersize=8)
    # Resaltar el k optimo
    idx_opt = ks.index(k_optimo)
    ax.plot(k_optimo, scores[idx_opt], 'o', color='red', markersize=14,
            zorder=5, label=f'k optimo = {k_optimo}')
    ax.annotate(
        f'k={k_optimo}\nsil={scores[idx_opt]:.4f}',
        xy=(k_optimo, scores[idx_opt]),
        xytext=(k_optimo + 0.3, scores[idx_opt] + 0.02),
        fontsize=9, color='red', fontweight='bold'
    )

    ax.set_xlabel("Numero de clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs. k", fontsize=12)
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def grafico_comparacion_combinaciones(modelos):
    """
    Grafico de barras horizontales: silhouette score (mejor k) de cada
    una de las 11 combinaciones entrenadas, ordenadas de mayor a menor.
    """
    datos = []
    for clave, info in modelos.items():
        datos.append({
            'combinacion': ' + '.join(clave),
            'silhouette': info['silhouette_score'],
            'k': info['k_optimo'],
        })

    # Ordenar de mayor a menor silhouette
    datos.sort(key=lambda x: x['silhouette'], reverse=True)

    etiquetas = [d['combinacion'] for d in datos]
    scores = [d['silhouette'] for d in datos]
    ks = [d['k'] for d in datos]

    fig, ax = plt.subplots(figsize=(10, 6))
    colores = plt.cm.viridis(np.linspace(0.2, 0.8, len(datos)))
    barras = ax.barh(range(len(datos)), scores, color=colores,
                     edgecolor='black', linewidth=0.3)

    ax.set_yticks(range(len(datos)))
    ax.set_yticklabels(etiquetas, fontsize=8)
    ax.set_xlabel("Silhouette Score (mejor k)")
    ax.set_title("Comparacion de Silhouette entre las 11 Combinaciones", fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Texto con score y k en cada barra
    for i, (barra, score, k) in enumerate(zip(barras, scores, ks)):
        ax.text(
            score + 0.005, i,
            f"{score:.4f} (k={k})",
            va='center', fontsize=8
        )

    plt.tight_layout()
    return fig
