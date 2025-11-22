#%%
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import balance as bal

#Centralizamos las funciones de procesamiento aca
def cargar_datos(file_paths):
    #Lee múltiples archivos CSV y los concatena en un único DataFrame.
    #file_paths: lista de rutas de archivos a leer
    data_frames = []

    for file in file_paths:
        # Leer cada archivo CSV
        df = pd.read_csv(
            file,
            na_values=["Ns/Nc", "No sabe", "No contesta"]
        )
        data_frames.append(df)

    # Unir todos los DataFrames
    base = pd.concat(data_frames, ignore_index=True)

    # Uniformar nombres de columnas
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


# ----------------------------------------------------------
# 3. LIMPIEZAS ESPECÍFICAS POR VARIABLE
# ----------------------------------------------------------

# 3.1 FECHA (ordenar y rango)
def limpiar_fecha(df):
    df = df.sort_values("fecha").reset_index(drop=True)
    return df


# 3.2 ENCUESTA_ID
def limpiar_encuesta_id(df):
    df["encuesta_id"] = pd.to_numeric(df["encuesta_id"], errors="coerce")
    df["encuesta_id"] = df["encuesta_id"].fillna(method="ffill").astype(int)
    return df


# 3.3 SEXO
def limpiar_sexo(df):
    df["sexo"] = (
        df["sexo"].astype(str).str.strip().str.capitalize()
    )

    sexos_validos = ["Masculino", "Femenino"]
    df = df[df["sexo"].isin(sexos_validos)]
    return df


# 3.4 EDAD
def limpiar_edad(df):
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df = df[(df["edad"] >= 16) & (df["edad"] <= 100)]
    return df


# 3.5 RANGO ETARIO
def crear_rango_etario(df):
    df["rango_etario"] = pd.cut(
        df["edad"],
        bins=[16, 24, 35, 45, 55, 75, float("inf")],
        labels=["16-24", "25-35", "36-45", "46-55", "56-75", "+76"],
        include_lowest=True,
        right=True
    )
    return df


# 3.6 VOTO ANTERIOR
def limpiar_voto_anterior(df):
    validos = [
        "Candidato_A",
        "Candidato_B",
        "No Fue A Votar",
        "Blanco",
        "Nulo",
        "Ns/Nc"
    ]
    df["voto_anterior"] = df["voto_anterior"].astype(str).str.strip()
    df = df[df["voto_anterior"].isin(validos)]
    return df


# 3.7 INTEGRANTES HOGAR
def limpiar_integrantes_hogar(df):
    df["integrantes_hogar"] = pd.to_numeric(df["integrantes_hogar"], errors="coerce")
    df = df[(df["integrantes_hogar"] >= 1) & (df["integrantes_hogar"] <= 10)]
    return df


# 3.8 VOTO
def limpiar_voto(df):
    validos = ["Candidato_A", "Candidato_B", "Blanco", "Nulo", "Ns/Nc"]
    df["voto"] = df["voto"].astype(str).str.strip()
    df = df[df["voto"].isin(validos)]
    return df


def limpiar_imagen(df):
    # Normalización mínima
    df["imagen_candidato"] = (
        df["imagen_candidato"]
        .astype(str)
        .str.strip()
    )

    # Convertir a numérico
    df["imagen_candidato"] = pd.to_numeric(df["imagen_candidato"], errors="coerce")

    # Mantener solo valores entre 0 y 100 (variable cerrada)
    df = df[(df["imagen_candidato"] >= 0) & (df["imagen_candidato"] <= 100)]

    # Convertir a entero
    df["imagen_candidato"] = df["imagen_candidato"].astype("Int64")

    return df

# 3.10 NIVEL EDUCATIVO
def limpiar_nivel_educativo(df):
    # Normalización básica
    df["nivel_educativo"] = (
        df["nivel_educativo"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Reemplazos y posibilidades para Ns/Nc
    df.loc[
        df["nivel_educativo"].str.contains("ns") |
        df["nivel_educativo"].str.contains("nc") |
        df["nivel_educativo"].str.contains("no sabe") |
        df["nivel_educativo"].str.contains("no contesta"),
        "nivel_educativo"
    ] = "Ns/Nc"

    # Lista de categorías válidas 
    categorias_validas = {
        "primario completo o incompleto": "Primario completo o incompleto",
        "secundario completo o incompleto": "Secundario completo o incompleto",
        "terciario completo o incompleto": "Terciario completo o incompleto",
        "universitario completo o incompleto": "Universitario completo o incompleto",
        "ns/nc": "Ns/Nc"
    }

    # Aplicar categorías válidas
    df["nivel_educativo"] = df["nivel_educativo"].map(categorias_validas)

    # Eliminar valores inválidos
    df = df.dropna(subset=["nivel_educativo"])

    return df


# 3.11 ESTRATO (provincias)
def limpiar_estrato(df):
    # Normalización básica
    df["estrato"] = (
        df["estrato"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Capitalizar (pone "Buenos Aires", "Cordoba", etc.)
    df["estrato"] = df["estrato"].apply(lambda x: x.title())

    return df

# ----------------------------------------------------------
# 4. TRATAMIENTO DE NaN
# ----------------------------------------------------------

def interpretar_nan(df):

    # A) Variables criticas - eliminar fila 
    criticas = ["fecha", "imagen_candidato"]
    for col in criticas:
        if col in df.columns:
            df = df.dropna(subset=[col])

    # B) Variables categoricas auxiliares → codificar missing como categoría
    categoricas = [
        "sexo",
        "nivel_educativo",
        "voto",
        "voto_anterior",
        "estrato"
    ]
    for col in categoricas:
        if col in df.columns:
            df[col] = df[col].fillna("Missing")

    # C) Variables numericas auxiliares → dejar como NaN (no imputar)
    # edad, integrantes_hogar quedan intactas
    
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
print ("no hay error hasta aca")

#PRUEBA

# ---------------------------------------------------------
# RESUMEN AUTOMÁTICO DEL TRACKING
# ---------------------------------------------------------
def resumen_tracking(df):
    """
    Devuelve:
    1. El DataFrame final ya procesado
    2. Un gráfico de torta con la distribución de sexo
    3. La media de imagen del candidato
    """
    
    # 1. Mostrar tabla final (primeras filas)
    print("\n----------------------------------------")
    print("TABLA FINAL DE ENCUESTADOS (HEAD)")
    print("----------------------------------------")
    print(df.head())
    
    # 2. Gráfico de torta de SEXO
    if "sexo" in df.columns:
        plt.figure(figsize=(5, 5))
        df["sexo"].value_counts().plot(kind="pie", autopct="%1.1f%%")
        plt.title("Distribución por sexo")
        plt.ylabel("")  # saca el label feo
        plt.show()
    else:
        print("\n[ADVERTENCIA] La columna 'sexo' no está en el DataFrame.")
    
    # 3. Media de imagen del candidato
    if "imagen_candidato" in df.columns:
        media_img = df["imagen_candidato"].mean()
        print("\n----------------------------------------")
        print("MEDIA DE IMAGEN DEL CANDIDATO")
        print("----------------------------------------")
        print(f"{media_img:.2f}")
    else:
        print("\n[ADVERTENCIA] 'imagen_candidato' no está en el DataFrame.")
    
    return df

