#%%
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import balance as bal

#Centralizamos las funciones de procesamiento aca

def cargar_datos(file_paths):
    
#Lee múltiples archivos de encuestas y los ordena en un único DataFrame.
#file_paths: lista de rutas de archivos a leer.
    
    data_frames = []
    for file in file_paths:
# Leer cada archivo CSV.
        df = pd.read_csv(file, na_values=["Ns/Nc", "No sabe", "No contesta"])
        data_frames.append(df)
# Unir todos los DataFrames 
    base = pd.concat(data_frames, ignore_index=True)
 # uniformar nombres de columnas a minúsculas y sin espacios
    base.columns = base.columns.str.lower().str.strip()
    return base

# LIMPIEZA BÁSICA
def limpiar_datos(df):
    """ Limpieza inicial muy básica:
    - convertir fecha si existe
    - convertir edad si existe
    - eliminar duplicados
    - eliminar edades menores a 16
    """
    # Fecha
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha"])

    # Edad
    if "edad" in df.columns:
        df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
        df = df[df["edad"] >= 16]

    # Duplicados
    df = df.drop_duplicates()

    return df

# LIMPIEZAS ESPECÍFICAS POR VARIABLE
# 1 FECHA (ordenar y rango)

def limpiar_fecha(df):
    df = df.sort_values("fecha").reset_index(drop=True)
    return df

# 2 ENCUESTA_ID
def limpiar_encuesta_id(df):
    df["encuesta_id"] = pd.to_numeric(df["encuesta_id"], errors="coerce")
    df["encuesta_id"] = df["encuesta_id"].fillna(method="ffill").astype(int)
    return df


# 3 SEXO
def limpiar_sexo(df):
    df["sexo"] = (
        df["sexo"].astype(str).str.strip().str.capitalize()
    )

    sexos_validos = ["Masculino", "Femenino"]
    df = df[df["sexo"].isin(sexos_validos)]
    return df

# 4 EDAD
def limpiar_edad(df):
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df = df[(df["edad"] >= 16) & (df["edad"] <= 100)]
    return df
# 4.1 Creamos la variable rango etario
def crear_edad_rango(df):
    df["edad_rango"] = pd.cut(
        df["edad"],
        bins=[16, 24, 35, 45, 55, 75, float("inf")],
        labels=["16-24", "25-35", "36-45", "46-55", "56-75", "+76"],
        include_lowest=True,
        right=True
    )
    return df

# 5. VOTO ANTERIOR 

def limpiar_voto_anterior(df):
    # Normalización y limpieza
    df["voto_anterior"] = (
        df["voto_anterior"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Creacion de categorias con contains, mas flexible
    df.loc[df["voto_anterior"].str.contains("blanco"), "voto_anterior"] = "Blanco"

    df.loc[df["voto_anterior"].str.contains("nulo"), "voto_anterior"] = "Nulo"

    df.loc[
        df["voto_anterior"].str.contains("ns") |
        df["voto_anterior"].str.contains("nc") |
        df["voto_anterior"].str.contains("no sabe") |
        df["voto_anterior"].str.contains("no contesta"),
        "voto_anterior"
    ] = "Ns/Nc"

    df.loc[
        df["voto_anterior"].str.contains("no fue"),
        "voto_anterior"
    ] = "No Fue A Votar"
    # Candidatos: cualquier string que no sea categoría fija → es válido
    categorias_fijas = ["Blanco", "Nulo", "Ns/Nc", "No Fue A Votar"]

    #
    df["voto_anterior"] = df["voto_anterior"].apply(
        lambda x: x.title() if x not in categorias_fijas else x
    )

    return df


# 6. VOTO
def limpiar_voto(df):
    # Normalización básica
    df["voto"] = (
        df["voto"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Blanco
    df.loc[df["voto"].str.contains("blanco"), "voto"] = "Blanco"

    # Nulo
    df.loc[df["voto"].str.contains("nulo"), "voto"] = "Nulo"

    # Ns/Nc
    df.loc[
        df["voto"].str.contains("ns") |
        df["voto"].str.contains("nc") |
        df["voto"].str.contains("no sabe") |
        df["voto"].str.contains("no contesta"),
        "voto"
    ] = "Ns/Nc"

    # No fue a votar
    df.loc[
        df["voto"].str.contains("no fue"),
        "voto"
    ] = "No Fue A Votar"

    # Categorías fijas cerradas
    categorias_fijas = ["Blanco", "Nulo", "Ns/Nc", "No Fue A Votar"]

    # Todo lo que no sea categoría fija → candidato
    df["voto"] = df["voto"].apply(
        lambda x: x.title() if x not in categorias_fijas else x
    )

    return df



# 7 INTEGRANTES HOGAR
def limpiar_integrantes_hogar(df):
    df["integrantes_hogar"] = pd.to_numeric(df["integrantes_hogar"], errors="coerce")
    df = df[(df["integrantes_hogar"] >= 1) & (df["integrantes_hogar"] <= 10)]
    return df

# 8 IMAGEN DEL CANDIDATO
# 3.9 IMAGEN DEL CANDIDATO (variable cerrada 0–100)
def limpiar_imagen(df):
    # Normalización básica (por las dudas venga con espacios)
    df["imagen_candidato"] = (
        df["imagen_candidato"]
        .astype(str)
        .str.strip()
    )

    # Convierte a numérico
    df["imagen_candidato"] = pd.to_numeric(df["imagen_candidato"], errors="coerce")

    # Limpiar valores no validos
    df = df[(df["imagen_candidato"] >= 0) & (df["imagen_candidato"] <= 100)]

    # Convertir a entero si la variable debe ser entera
    df["imagen_candidato"] = df["imagen_candidato"].astype("Int64")

    return df



# ----------------------------------------------------------
# 4. VARIABLES AUXILIARES
# ----------------------------------------------------------
def generar_auxiliares(df):
    # Periodo basado en fecha (YYYY-MM)
    if "fecha" in df.columns:
        df["periodo"] = df["fecha"].dt.to_period("M").astype(str)

    return df


# ----------------------------------------------------------
# 5. IMAGEN PROMEDIO (MEDIA SIMPLE O PONDERADA)
# ----------------------------------------------------------
def calcular_imagen_promedio(df, peso_col=None):
    """
    Calcula la imagen promedio del candidato.
    Si hay pesos, usa media ponderada.
    """

    grupo = "fecha" if "fecha" in df.columns else "periodo"

    if peso_col and peso_col in df.columns:
        imagen = df.groupby(grupo).apply(
            lambda g: np.average(g["imagen_candidato"], weights=g[peso_col])
        )
        imagen_prom = imagen.reset_index(name="imagen_promedio")
    else:
        imagen_prom = (
            df.groupby(grupo)["imagen_candidato"]
            .mean()
            .reset_index()
            .rename(columns={"imagen_candidato": "imagen_promedio"})
        )

    return imagen_prom
# %%
