"""
core/reportes.py
Generacion de archivos Excel y PDF con reportlab.
"""
import io
import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generar_excel_filtrado(df_filtrado):
    """Genera un archivo Excel en memoria con los datos filtrados."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_filtrado.to_excel(writer, index=False, sheet_name='Datos_Filtrados')
    buffer.seek(0)
    return buffer.getvalue()


def generar_pdf_estadistica_base(stats_df, habilidades_sel):
    """Genera un PDF con la estadistica base (proceso propio) usando reportlab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    elementos = []

    titulo_style = ParagraphStyle('Titulo', parent=styles['Title'], fontSize=16)
    elementos.append(Paragraph("Reporte: Estadistica Base (Proceso Propio)", titulo_style))
    elementos.append(Spacer(1, 12))

    sub = "Habilidades analizadas: " + ", ".join(habilidades_sel)
    elementos.append(Paragraph(sub, styles['Normal']))
    elementos.append(Spacer(1, 8))
    elementos.append(Paragraph(
        "Estadisticas calculadas con logica propia (operaciones aritmeticas "
        "manuales, sin funciones de ML ni df.describe()).",
        styles['Italic']))
    elementos.append(Spacer(1, 16))

    header = ['Indicador', 'Media', 'Desv. Std', 'Min', 'Max', 'Mediana', 'Rango']
    table_data = [header]
    for idx, row in stats_df.iterrows():
        table_data.append([
            str(idx), str(row['Media']), str(row['Desv. Std']),
            str(row['Min']), str(row['Max']), str(row['Mediana']), str(row['Rango']),
        ])

    tabla = Table(table_data, repeatRows=1)
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elementos.append(tabla)
    doc.build(elementos)
    buffer.seek(0)
    return buffer.getvalue()


def generar_pdf_estadistica_modelo(perfil_cluster, info_modelo, habilidades_sel,
                                   fig_tamano_path, fig_perfil_path):
    """Genera un PDF con la estadistica del modelo (K-Means) usando reportlab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    elementos = []

    titulo_style = ParagraphStyle('Titulo', parent=styles['Title'], fontSize=16)
    elementos.append(Paragraph(
        "Reporte: Estadistica del Modelo (K-Means)", titulo_style))
    elementos.append(Spacer(1, 12))

    sub = "Habilidades analizadas: " + ", ".join(habilidades_sel)
    elementos.append(Paragraph(sub, styles['Normal']))
    elementos.append(Spacer(1, 6))
    elementos.append(Paragraph(
        f"k optimo: {info_modelo['k_optimo']}  |  "
        f"Silhouette Score: {info_modelo['silhouette_score']:.4f}",
        styles['Normal']))
    elementos.append(Spacer(1, 6))
    elementos.append(Paragraph(
        "Modelo cargado desde archivo pre-entrenado. "
        "Pipeline: StandardScaler -> KMeans.",
        styles['Italic']))
    elementos.append(Spacer(1, 16))

    # Tabla de perfil por cluster
    elementos.append(Paragraph("Perfil Promedio por Cluster", styles['Heading2']))
    elementos.append(Spacer(1, 8))

    columnas_datos = [c for c in perfil_cluster.columns if c != 'Empleados']
    header = ['Cluster', 'N Empleados'] + [str(c) for c in columnas_datos]
    table_data = [header]
    for idx, row in perfil_cluster.iterrows():
        fila = [f"Cluster {idx}", str(int(row['Empleados']))]
        for c in columnas_datos:
            fila.append(str(row[c]))
        table_data.append(fila)

    tabla = Table(table_data, repeatRows=1)
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elementos.append(tabla)
    elementos.append(Spacer(1, 16))

    # Graficos
    if fig_tamano_path and os.path.exists(fig_tamano_path):
        elementos.append(Paragraph("Tamano de cada Cluster", styles['Heading2']))
        elementos.append(Spacer(1, 8))
        elementos.append(Image(fig_tamano_path, width=5*inch, height=3*inch))
        elementos.append(Spacer(1, 12))

    if fig_perfil_path and os.path.exists(fig_perfil_path):
        elementos.append(Paragraph(
            "Perfil Promedio por Cluster (Barras Agrupadas)", styles['Heading2']))
        elementos.append(Spacer(1, 8))
        elementos.append(Image(fig_perfil_path, width=6.5*inch, height=4*inch))

    doc.build(elementos)
    buffer.seek(0)
    return buffer.getvalue()
