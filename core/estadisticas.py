"""
core/estadisticas.py
Funciones de estadistica:
  - Estadistica base (calculo manual propio, sin df.describe() ni ML)
  - Perfil promedio por cluster (resultado del modelo)
"""
import pandas as pd


def calcular_estadistica_base(df_datos):
    """
    Calcula estadisticas descriptivas de forma MANUAL.
    Cada operacion (media, desviacion, min, max, mediana) esta escrita
    con operaciones aritmeticas explicitas, sin usar df.describe()
    ni funciones de machine learning.

    Retorna: DataFrame con filas=indicadores, columnas=metricas.
    """
    resultados = {}

    for col in df_datos.columns:
        valores = df_datos[col].values
        n = len(valores)

        # Media: suma de todos los valores dividida entre n
        suma = 0.0
        for v in valores:
            suma += v
        media = suma / n

        # Desviacion estandar poblacional: raiz de la media de cuadrados de diferencias
        suma_cuadrados = 0.0
        for v in valores:
            suma_cuadrados += (v - media) ** 2
        desviacion = (suma_cuadrados / n) ** 0.5

        # Minimo: recorrer todos los valores y quedarse con el menor
        minimo = valores[0]
        for v in valores[1:]:
            if v < minimo:
                minimo = v

        # Maximo: recorrer todos los valores y quedarse con el mayor
        maximo = valores[0]
        for v in valores[1:]:
            if v > maximo:
                maximo = v

        # Mediana: ordenar y tomar el valor central
        ordenados = sorted(valores)
        if n % 2 == 0:
            mediana = (ordenados[n // 2 - 1] + ordenados[n // 2]) / 2
        else:
            mediana = ordenados[n // 2]

        rango = maximo - minimo

        resultados[col] = {
            'Media': round(media, 2),
            'Desv. Std': round(desviacion, 2),
            'Min': round(minimo, 1),
            'Max': round(maximo, 1),
            'Mediana': round(mediana, 1),
            'Rango': round(rango, 1),
        }

    return pd.DataFrame(resultados).T


def calcular_perfil_clusters(df_filtrado, columnas_sel):
    """
    Calcula el perfil promedio de cada cluster y el conteo de empleados.
    Requiere que df_filtrado ya tenga una columna 'Cluster'.

    Retorna: DataFrame con filas=clusters, columnas=[Empleados, ...indicadores].
    """
    perfil = df_filtrado.groupby('Cluster')[columnas_sel].mean().round(2)
    perfil['Empleados'] = df_filtrado.groupby('Cluster')['Cluster'].count().values
    cols_orden = ['Empleados'] + columnas_sel
    return perfil[cols_orden]
