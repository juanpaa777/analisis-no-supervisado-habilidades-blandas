"""
core/config.py
Constantes globales del proyecto: habilidades, columnas, rutas, parametros.
"""
from itertools import combinations
import os

# ---------- Mapeo de habilidades a columnas ----------
HABILIDADES_MAP = {
    'Comunicacion Efectiva': [
        'com_claridad', 'com_escucha_activa', 'com_retroalimentacion'
    ],
    'Colaboracion': [
        'col_participacion', 'col_apoyo_companeros', 'col_cumplimiento_acuerdos'
    ],
    'Resolucion de Conflictos': [
        'res_manejo_tension', 'res_busqueda_acuerdos', 'res_mediacion'
    ],
    'Adaptacion al Cambio': [
        'ada_flexibilidad', 'ada_tolerancia_incertidumbre', 'ada_aprendizaje_continuo'
    ],
}

NOMBRES_HABILIDADES = list(HABILIDADES_MAP.keys())

TODAS_LAS_COLUMNAS = []
for _cols in HABILIDADES_MAP.values():
    TODAS_LAS_COLUMNAS.extend(_cols)

# ---------- Rango de k a evaluar ----------
K_MIN = 2
K_MAX = 5

# ---------- Rutas ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELOS_DIR = os.path.join(BASE_DIR, 'modelos')

ARCHIVO_DATASET = os.path.join(DATA_DIR, 'dataset_habilidades_blandas.csv')
ARCHIVO_MODELOS = os.path.join(MODELOS_DIR, 'modelo_habilidades.pkl')

# ---------- 11 combinaciones posibles ----------
TODAS_COMBINACIONES = []
for r in range(2, 5):
    for combo in combinations(NOMBRES_HABILIDADES, r):
        TODAS_COMBINACIONES.append(combo)

# ---------- Parametros de generacion por perfil ----------
PARAMS_PERFILES = {
    'A': {
        'com_claridad': (8.5, 0.9), 'com_escucha_activa': (8.3, 1.0),
        'com_retroalimentacion': (8.0, 1.0),
        'col_participacion': (8.5, 0.8), 'col_apoyo_companeros': (8.7, 0.9),
        'col_cumplimiento_acuerdos': (8.8, 0.7),
        'res_manejo_tension': (8.0, 1.0), 'res_busqueda_acuerdos': (8.2, 0.9),
        'res_mediacion': (7.8, 1.1),
        'ada_flexibilidad': (8.3, 0.9), 'ada_tolerancia_incertidumbre': (7.9, 1.0),
        'ada_aprendizaje_continuo': (8.6, 0.8),
    },
    'B': {
        'com_claridad': (5.0, 1.4), 'com_escucha_activa': (5.5, 1.3),
        'com_retroalimentacion': (4.8, 1.5),
        'col_participacion': (5.2, 1.3), 'col_apoyo_companeros': (5.0, 1.4),
        'col_cumplimiento_acuerdos': (6.0, 1.2),
        'res_manejo_tension': (6.0, 1.3), 'res_busqueda_acuerdos': (5.5, 1.4),
        'res_mediacion': (5.8, 1.3),
        'ada_flexibilidad': (8.5, 0.9), 'ada_tolerancia_incertidumbre': (8.0, 1.0),
        'ada_aprendizaje_continuo': (8.8, 0.7),
    },
    'C': {
        'com_claridad': (6.5, 1.1), 'com_escucha_activa': (6.2, 1.2),
        'com_retroalimentacion': (6.0, 1.3),
        'col_participacion': (6.3, 1.2), 'col_apoyo_companeros': (6.5, 1.1),
        'col_cumplimiento_acuerdos': (6.8, 1.0),
        'res_manejo_tension': (4.2, 1.4), 'res_busqueda_acuerdos': (4.5, 1.5),
        'res_mediacion': (4.0, 1.4),
        'ada_flexibilidad': (4.5, 1.3), 'ada_tolerancia_incertidumbre': (4.0, 1.5),
        'ada_aprendizaje_continuo': (4.3, 1.4),
    },
}
