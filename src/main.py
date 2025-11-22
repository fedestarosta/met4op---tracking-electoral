#%%
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import balance as bal
import os

#Importar procesamientos
from procesamiento import (
    cargar_datos,
    interpretar_nan,
    limpiar_fecha,
    limpiar_encuesta_id,
    limpiar_sexo,
    limpiar_edad,
    crear_rango_etario,
    limpiar_nivel_educativo,
    limpiar_integrantes_hogar,
    limpiar_voto,
    limpiar_voto_anterior,
    limpiar_estrato,
    limpiar_imagen,
    resumen_tracking
)

path = "data/"
file_name = "encuestas_falsas.csv"
ruta_completa = path + file_name

if not os.path.exists(ruta_completa):
    print("El archivo", file_name, "no está en la carpeta data.")
else:
    df = pd.read_csv(ruta_completa)
    print(df.head())

# ---------------------------------------------------------
# 3. MANEJO DE VALORES FALTANTES (PACZKOWSKI)
# ---------------------------------------------------------

df = interpretar_nan(df)

# ---------------------------------------------------------
# 4. LIMPIEZA DE VARIABLES
# ---------------------------------------------------------

df = limpiar_fecha(df)
df = limpiar_encuesta_id(df)
df = limpiar_sexo(df)
df = limpiar_edad(df)
df = crear_rango_etario(df)
df = limpiar_nivel_educativo(df)
df = limpiar_integrantes_hogar(df)
df = limpiar_voto(df)
df = limpiar_voto_anterior(df)
df = limpiar_estrato(df)
df = limpiar_imagen(df)

# ---------------------------------------------------------
# 5. RESUMEN FINAL (TABLA / TORTA SEXO / MEDIA IMAGEN)
# ---------------------------------------------------------

resumen_tracking(df)

