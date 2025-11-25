#%%
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import balance as bal
import os
from pandas.api.types import CategoricalDtype
#%%
#Importar procesamientos
from procesamiento import (
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
)

path = "data/"
file_name = "encuestas_falsas.csv"
ruta_completa = path + file_name

if not os.path.exists(ruta_completa):
    print("El archivo", file_name, "no está en la carpeta data.")
else:
    df = pd.read_csv(ruta_completa)
    print(df.head())
#%%
# 3. MANEJO DE VALORES FALTANTES
df = interpretar_nan(df)
#%%
# 4. Importar funciones de limpieza
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
#%%
# Manejo de pesos
from procesamiento import peso_col
df_poblacion = pd.read_csv("data/pesos_fuente_censo2022.csv", decimal='.')
df = peso_col(df, df_poblacion)

#%%
#OUTPUTS

#Trackeo de imagen - rolling
from procesamiento import tracking_imagen

df = peso_col(df, df_poblacion)
tabla_tracking = tracking_imagen(df, peso_col="peso", window=3)
print(tabla_tracking.head())

#Trackeo de voto
from procesamiento import tracking_voto

tabla_voto = tracking_voto(df, peso_col="peso", umbral_minimo=150) #cambiar umbral segun sea conveniente
print(tabla_voto.head())

    #Grafico tracking voto
from procesamiento import grafico_tracking_voto
grafico_tracking_voto(tabla_voto)

#Heatmap transferencia de voto
from procesamiento import heatmap_transferencia

tabla_transf = heatmap_transferencia(df, peso_col="peso")
print(tabla_transf)

    #CROSSTABLES Simples
from procesamiento import plot_imagen_por_rango
tabla_cruce = plot_imagen_por_rango(df)
print(tabla_cruce)
