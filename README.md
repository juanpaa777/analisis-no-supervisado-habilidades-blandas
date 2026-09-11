# Análisis No Supervisado de Habilidades Blandas con K-Means

Sistema interactivo desarrollado en **Python** y **Streamlit** para la evaluación, segmentación y agrupamiento de perfiles de colaboradores mediante **Machine Learning No Supervisado (K-Means Clustering)**.

---

## 📌 Características Principales

1. **Exploración de Datos y Estadística Descriptiva Propia:**
   - Carga interactiva de datasets en formato `.xlsx` o `.csv`.
   - Selección dinámica de 2 a 4 dimensiones de habilidades blandas:
     - *Comunicación Efectiva* (Claridad, Escucha Activa, Retroalimentación).
     - *Colaboración* (Participación, Apoyo al Equipo, Acuerdos).
     - *Resolución de Conflictos* (Manejo de Tensión, Búsqueda de Acuerdos, Mediación).
     - *Adaptación al Cambio* (Flexibilidad, Tolerancia a la Incertidumbre, Aprendizaje Continuo).
   - Cálculo manual de métricas estadísticas (Media, Desviación Estándar, Mínimo, Máximo, Mediana, Rango).
   - Exportación de la información filtrada a **Excel** y reporte estadístico en **PDF**.

2. **Clasificación con Modelo Pre-entrenado (K-Means):**
   - Carga manual o automática del modelo serializado (`.pkl`).
   - Estandarización automática (`StandardScaler`) y asignación de clústeres óptimos según la combinación seleccionada.
   - Visualización de distribución de clústeres y perfiles promedios mediante gráficos de barras agrupadas.
   - Generación de reportes ejecutivos descargables en **PDF** con análisis comparativo.

3. **Transparencia y Re-entrenamiento:**
   - Panel de auditoría con curvas de evaluación de métrica **Silhouette Score** ($k=2 \dots 5$).
   - Opciones para re-entrenar los modelos con nuevos datos y descargar el nuevo archivo `.pkl`.

---

## 🛠️ Estructura del Proyecto

```text
instrumento r2/
├── app.py                     # Interfaz gráfica principal (Streamlit)
├── entrenar_modelo.py         # Script de entrenamiento independiente
├── requirements.txt           # Dependencias del proyecto
├── .gitignore                 # Archivos excluidos del control de versiones
├── core/                      # Módulos de lógica de negocio y procesamiento
│   ├── __init__.py
│   ├── config.py              # Constantes, nombres de columnas y esquemas
│   ├── generador_datos.py     # Generación de dataset sintético con perfiles ocultos
│   ├── estadisticas.py        # Algoritmo propio de cálculo estadístico descriptivo
│   ├── entrenamiento.py       # Entrenamiento y optimización de K-Means (11 combinaciones)
│   ├── carga_modelo.py        # Carga, validación y predicción con .pkl
│   ├── graficos.py            # Generación de gráficos Plotly interactivos
│   └── reportes.py            # Generación de reportes en PDF con ReportLab
├── data/
│   └── dataset_habilidades_blandas.csv
└── modelos/
    └── modelo_habilidades.pkl # Diccionario de modelos K-Means pre-entrenados
```

---

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/juanpaa777/analisis-no-supervisado-habilidades-blandas.git
cd analisis-no-supervisado-habilidades-blandas
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. (Opcional) Entrenar / regenerar los modelos
```bash
python entrenar_modelo.py
```

### 4. Iniciar la aplicación
```bash
streamlit run app.py
```

---

## 👥 Materia
- **Carrera:** TIC – Ingeniería en Desarrollo y Gestión de Software
- **Materia:** Extracción de Conocimientos en Base de Datos
- **Unidad:** IV. Análisis No Supervisado