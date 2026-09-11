"""
app.py -- Interfaz Streamlit (Modo Exposición)
Solo interfaz: toda la lógica está en core/.
Estructurada en 3 pestañas para una presentación limpia y secuencial.
"""
import streamlit as st
import pandas as pd
import os
import tempfile
import joblib
import io

from core.config import (
    HABILIDADES_MAP, NOMBRES_HABILIDADES, K_MIN, K_MAX
)
from core.carga_modelo import buscar_modelo, aplicar_modelo
from core.estadisticas import calcular_estadistica_base, calcular_perfil_clusters
from core.graficos import (
    grafico_tamano_clusters, grafico_perfil_clusters,
    grafico_silhouette_vs_k, grafico_comparacion_combinaciones
)
from core.reportes import (
    generar_excel_filtrado, generar_pdf_estadistica_base,
    generar_pdf_estadistica_modelo
)
from core.entrenamiento import entrenar_todos_los_modelos

# =====================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTADO
# =====================================================================
st.set_page_config(
    page_title="Análisis No Supervisado -- Habilidades Blandas",
    layout="wide"
)

# Funciones de sincronización de multiselects
if 'habilidades_sel' not in st.session_state:
    st.session_state['habilidades_sel'] = NOMBRES_HABILIDADES[:2]

def sync_habs_1():
    st.session_state['habilidades_sel'] = st.session_state['ms_1']
    # Sincronizar el segundo selector si existe
    st.session_state['ms_2'] = st.session_state['ms_1']
    st.session_state['clasificacion_realizada'] = False

def sync_habs_2():
    st.session_state['habilidades_sel'] = st.session_state['ms_2']
    # Sincronizar el primer selector
    st.session_state['ms_1'] = st.session_state['ms_2']
    st.session_state['clasificacion_realizada'] = False

if 'df_crudo' not in st.session_state:
    st.session_state['df_crudo'] = None
if 'modelos_dict' not in st.session_state:
    st.session_state['modelos_dict'] = None
if 'clasificacion_realizada' not in st.session_state:
    st.session_state['clasificacion_realizada'] = False

st.title("Análisis No Supervisado: Habilidades Blandas")
st.markdown(
    "Aplicación diseñada para descubrir perfiles conductuales ocultos en evaluaciones "
    "de desempeño mediante **K-Means Clustering**."
)

# =====================================================================
# PESTAÑAS (Flujo de Exposición)
# =====================================================================
tab_exploracion, tab_clasificacion, tab_transparencia = st.tabs([
    "1. Exploración y Estadística Base",
    "2. Clasificación con K-Means",
    "3. Transparencia del Modelo"
])

# =====================================================================
# PESTAÑA 1: EXPLORACIÓN Y ESTADÍSTICA BASE
# =====================================================================
with tab_exploracion:
    st.header("Exploración de Datos (Información Pura)")
    st.markdown("Aquí analizamos descriptivamente a la población de forma global, "
                "sin aplicar aún ningún algoritmo de agrupamiento.")

    # --- Carga de datos ---
    st.subheader("Paso 1: Cargar Archivo de Respuestas")
    uploaded_csv = st.file_uploader(
        "Sube el archivo CSV del dataset (ej. dataset_habilidades_blandas.csv)",
        type="csv", key="uploader_csv"
    )

    if uploaded_csv is not None:
        # Cargar datos al estado de la sesión
        st.session_state['df_crudo'] = pd.read_csv(uploaded_csv)
        # Resetear clasificación si se sube nuevo archivo
        st.session_state['clasificacion_realizada'] = False

    df = st.session_state['df_crudo']

    if df is not None:
        st.success(f"Archivo cargado correctamente: {len(df)} registros.")

        # --- Selección de habilidades ---
        st.subheader("Paso 2: Seleccionar Habilidades a Analizar")
        habs = st.multiselect(
            "Elige entre 2 y 4 habilidades:",
            options=NOMBRES_HABILIDADES,
            key="ms_1",
            on_change=sync_habs_1
        )
        
        habs = st.session_state['habilidades_sel']

        n_sel = len(habs)
        if n_sel < 2:
            st.warning("Selecciona al menos **2 habilidades** para continuar.")
        elif n_sel > 4:
            st.warning("Selecciona máximo **4 habilidades**.")
        else:
            # Lógica de filtrado
            columnas_sel = []
            for h in habs:
                columnas_sel.extend(HABILIDADES_MAP[h])

            tiene_id = 'id_empleado' in df.columns
            cols_mostrar = (['id_empleado'] + columnas_sel) if tiene_id else columnas_sel
            df_filtrado = df[cols_mostrar].copy()
            df_solo_datos = df[columnas_sel].copy()

            # --- Mostrar tabla y Excel ---
            st.subheader("Datos Filtrados")
            st.dataframe(df_filtrado, use_container_width=True, height=250)

            excel_bytes = generar_excel_filtrado(df_filtrado)
            st.download_button(
                label="📥 Descargar datos filtrados en Excel (.xlsx)",
                data=excel_bytes,
                file_name="datos_filtrados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="btn_excel"
            )

            # --- Estadística Base (Proceso Propio) ---
            st.subheader("Estadística Base (Proceso Propio)")
            st.info("Cálculos aritméticos explícitos elaborados de forma manual "
                    "(sin usar funciones de Machine Learning ni df.describe()).")
            
            stats_base = calcular_estadistica_base(df_solo_datos)
            st.dataframe(stats_base, use_container_width=True)

            pdf_base = generar_pdf_estadistica_base(stats_base, habs)
            st.download_button(
                label="📄 Descargar Estadística Base en PDF",
                data=pdf_base,
                file_name="estadistica_base.pdf",
                mime="application/pdf",
                key="pdf_base",
            )
            
            st.success("✅ **Fase de exploración completada.** Procede a la pestaña '2. Clasificación con K-Means'.")
    else:
        st.info("Sube un dataset para comenzar.")


# =====================================================================
# PESTAÑA 2: CLASIFICACIÓN (K-MEANS)
# =====================================================================
with tab_clasificacion:
    st.header("Clasificación de Perfiles con Machine Learning")
    st.markdown("Esta fase aplica **K-Means** (Aprendizaje No Supervisado) para "
                "segmentar a los empleados en grupos según su similitud conductual.")

    df = st.session_state['df_crudo']
    habs = st.session_state['habilidades_sel']

    if df is None:
        st.warning("⚠️ Debes cargar los datos en la **Pestaña 1** antes de aplicar el modelo.")
    else:
        st.subheader("Seleccionar Habilidades a Analizar")
        st.multiselect(
            "Puedes modificar la selección aquí (se sincronizará con la estadística base):",
            options=NOMBRES_HABILIDADES,
            key="ms_2",
            on_change=sync_habs_2
        )
        
        habs = st.session_state['habilidades_sel']
        
        if len(habs) < 2 or len(habs) > 4:
            st.warning("⚠️ Debes seleccionar entre 2 y 4 habilidades.")
        else:
            # --- Importar Archivo .pkl ---
            st.subheader("Paso 1: Importar Modelos Pre-entrenados")
        st.info("Para garantizar tiempos de respuesta rápidos y consistencia, el modelo "
                "fue entrenado previamente para todas las combinaciones posibles.")
        
        uploaded_pkl = st.file_uploader(
            "Arrastra el archivo de modelo (ej. modelo_habilidades.pkl)",
            type="pkl", key="uploader_pkl"
        )

        if uploaded_pkl is not None:
            # Cargar el diccionario de modelos a session_state
            if st.session_state['modelos_dict'] is None:
                st.session_state['modelos_dict'] = joblib.load(uploaded_pkl)
                st.success("Modelos cargados exitosamente en la memoria del sistema.")
            
            modelos = st.session_state['modelos_dict']
            info_modelo = buscar_modelo(modelos, habs)

            if info_modelo is None:
                st.error(f"El archivo cargado no contiene un modelo válido para la combinación: {habs}")
            else:
                # --- Botón de Ejecución Visual ---
                st.subheader("Paso 2: Aplicar Algoritmo")
                st.markdown("Al presionar este botón, el algoritmo evaluará cada empleado "
                            "midiendo sus distancias euclidianas (previa estandarización) "
                            "y le asignará el arquetipo conductual más cercano.")
                
                if st.button("🚀 Aplicar Clasificación K-Means", type="primary"):
                    st.session_state['clasificacion_realizada'] = True

                # --- Resultados de la Clasificación ---
                if st.session_state['clasificacion_realizada']:
                    st.divider()
                    st.subheader("Resultados del Modelo (Estadística de Clústeres)")

                    pipeline = info_modelo['pipeline']
                    k_opt = info_modelo['k_optimo']
                    sil_score = info_modelo['silhouette_score']

                    st.markdown("### Métricas de Evaluación")
                    met1, met2 = st.columns(2)
                    with met1:
                        st.metric("Número óptimo de grupos (K)", k_opt)
                    with met2:
                        st.metric("Coeficiente de Silueta (Silhouette Score)", f"{sil_score:.4f}")

                    # Aplicar modelo para resultados
                    columnas_sel = []
                    for h in habs:
                        columnas_sel.extend(HABILIDADES_MAP[h])
                    df_solo_datos = df[columnas_sel].copy()
                    
                    X = df_solo_datos.values
                    etiquetas = aplicar_modelo(pipeline, X)
                    
                    tiene_id = 'id_empleado' in df.columns
                    cols_mostrar = (['id_empleado'] + columnas_sel) if tiene_id else columnas_sel
                    df_filtrado = df[cols_mostrar].copy()
                    df_filtrado['Cluster'] = etiquetas

                    # Perfil promedio
                    perfil_cluster = calcular_perfil_clusters(df_filtrado, columnas_sel)
                    st.markdown("### Perfil Promedio por Grupo")
                    st.dataframe(perfil_cluster, use_container_width=True)

                    # Gráficos
                    st.markdown("### Visualización de Segmentos")
                    col_graf1, col_graf2 = st.columns([1, 2])
                    
                    conteo_clusters = df_filtrado['Cluster'].value_counts().sort_index()
                    fig_tamano = grafico_tamano_clusters(conteo_clusters, k_opt)
                    
                    fig_perfil = grafico_perfil_clusters(perfil_cluster, columnas_sel, k_opt)
                    
                    with col_graf1:
                        st.pyplot(fig_tamano)
                    with col_graf2:
                        st.pyplot(fig_perfil)

                    # Generación PDF
                    temp_fig1 = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    fig_tamano.savefig(temp_fig1.name, dpi=150, bbox_inches='tight')
                    temp_fig1.close()

                    temp_fig2 = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    fig_perfil.savefig(temp_fig2.name, dpi=150, bbox_inches='tight')
                    temp_fig2.close()

                    pdf_modelo = generar_pdf_estadistica_modelo(
                        perfil_cluster, info_modelo, habs,
                        temp_fig1.name, temp_fig2.name
                    )
                    st.download_button(
                        label="📄 Descargar Reporte Completo del Modelo en PDF",
                        data=pdf_modelo,
                        file_name="reporte_kmeans.pdf",
                        mime="application/pdf",
                        type="primary"
                    )

                    for tmp in [temp_fig1.name, temp_fig2.name]:
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
        else:
            # Si se quita el archivo
            st.session_state['modelos_dict'] = None
            st.session_state['clasificacion_realizada'] = False


# =====================================================================
# PESTAÑA 3: TRANSPARENCIA DEL MODELO
# =====================================================================
with tab_transparencia:
    st.header("Transparencia y Explicabilidad")
    st.markdown("En esta sección se detalla cómo el sistema determinó "
                "el número óptimo de clústeres y se comparan las distintas combinaciones.")

    modelos = st.session_state['modelos_dict']

    if modelos is None:
        st.info("⚠️ Sube el archivo `.pkl` en la Pestaña 2 para desbloquear el análisis de transparencia.")
    else:
        # Ficha técnica
        primer_modelo = list(modelos.values())[0]
        n_registros = primer_modelo.get('n_registros', 'N/A')

        st.subheader("Ficha Técnica del Pre-entrenamiento")
        ficha = {
            'Parámetro': [
                'Algoritmo', 'Preprocesamiento', 'Rango de K evaluado', 
                'Métrica de selección', 'Combinaciones entrenadas'
            ],
            'Valor': [
                'K-Means', 'StandardScaler (Estandarización)', f'{K_MIN} a {K_MAX}', 
                'Silhouette Score (Máximo)', f'{len(modelos)} modelos paralelos'
            ],
        }
        st.table(pd.DataFrame(ficha))

        # Silhouette vs K
        st.subheader("Determinación de K Óptimo (Curva de Evaluación)")
        habs_actual = st.session_state['habilidades_sel']
        
        if len(habs_actual) >= 2:
            info_actual = buscar_modelo(modelos, habs_actual)
            if info_actual:
                st.write(f"Para la combinación actual (**{' + '.join(habs_actual)}**), "
                         f"el algoritmo evaluó múltiples valores de *K*. El gráfico demuestra "
                         f"matemáticamente por qué eligió **K={info_actual['k_optimo']}**.")
                
                fig_sil = grafico_silhouette_vs_k(
                    info_actual['scores_por_k'], info_actual['k_optimo']
                )
                st.pyplot(fig_sil)
        
        # Comparación general
        st.subheader("Comparación General (Las 11 Combinaciones)")
        st.write("¿Qué combinación de habilidades genera los perfiles más nítidos y separados?")
        fig_comp = grafico_comparacion_combinaciones(modelos)
        st.pyplot(fig_comp)

        # Re-entrenamiento en vivo
        st.divider()
        st.subheader("Auditoría: Re-entrenamiento en Vivo (Opcional)")
        st.markdown("Los modelos fueron pre-entrenados y serializados por eficiencia. "
                    "Sin embargo, puedes forzar el entrenamiento en vivo en este momento "
                    "para observar cómo el algoritmo iteró y evaluó todas las posibilidades.")
        
        if st.button("🔄 Ejecutar Script de Entrenamiento Ahora"):
            df_crudo = st.session_state['df_crudo']
            if df_crudo is None:
                st.error("Sube primero el dataset CSV en la Pestaña 1.")
            else:
                with st.spinner("Entrenando 11 pipelines de K-Means. Calculando Silhouette Scores..."):
                    modelos_nuevos, log_text = entrenar_todos_los_modelos(df_crudo)
                
                st.session_state['entrenamiento_en_vivo_log'] = log_text
                
                # Serializar el modelo en memoria para el botón de descarga
                buffer = io.BytesIO()
                joblib.dump(modelos_nuevos, buffer)
                st.session_state['entrenamiento_en_vivo_pkl'] = buffer.getvalue()

        if 'entrenamiento_en_vivo_log' in st.session_state:
            st.success("Entrenamiento finalizado. Aquí está el registro del proceso:")
            st.code(st.session_state['entrenamiento_en_vivo_log'], language="text")
            
            st.download_button(
                label="📥 Descargar Nuevos Modelos Entrenados (.pkl)",
                data=st.session_state['entrenamiento_en_vivo_pkl'],
                file_name="nuevos_modelos_habilidades.pkl",
                mime="application/octet-stream",
                type="primary"
            )
