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
        "candidato_a",
        "candidato_b",
        "no fue a votar",
        "blanco",
        "nulo",
        "ns/nc"
    ]
    df["voto_anterior"] = df["voto_anterior"].astype(str).str.strip().str.lower()
    df = df[df["voto_anterior"].isin(validos)]
    return df


# 3.7 INTEGRANTES HOGAR
def limpiar_integrantes_hogar(df):
    df["integrantes_hogar"] = pd.to_numeric(df["integrantes_hogar"], errors="coerce")
    df = df[(df["integrantes_hogar"] >= 1) & (df["integrantes_hogar"] <= 10)]
    return df


# 3.8 VOTO
def limpiar_voto(df):
    validos = ["candidato_a", "candidato_b", "blanco", "nulo", "ns/nc"]
    df["voto"] = df["voto"].astype(str).str.strip().str.lower()
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

    df["nivel_educativo"] = (
        df["nivel_educativo"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    categorias_validas = {
        "primario completo",
        "primario incompleto",
        "secundario completo",
        "secundario incompleto",
        "terciario completo",
        "terciario incompleto",
        "universitario completo",
        "universitario incompleto",
        "ns/nc"
    }

    df["nivel_educativo"] = df["nivel_educativo"].apply(
        lambda x: x if x in categorias_validas else "Missing"
    )

    return df


# 3.11 ESTRATO (provincias)
def limpiar_estrato(df):
    df["estrato"] = (
        df["estrato"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    provincias_validas = {
        "buenos aires",
        "catamarca",
        "chaco",
        "chubut",
        "ciudad autónoma de buenos aires",
        "corrientes",
        "córdoba",
        "entre ríos",
        "formosa",
        "jujuy",
        "la pampa",
        "la rioja",
        "mendoza",
        "misiones",
        "neuquén",
        "río negro",
        "salta",
        "san juan",
        "san luis",
        "santa cruz",
        "santa fe",
        "santiago del estero",
        "tierra del fuego",
        "antártida e islas del atlántico sur",
    }

    df["estrato"] = df["estrato"].apply(
        lambda x: x if x in provincias_validas else "Missing"
    )

    return df
# 4. TRATAMIENTO DE NaN
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

# 5. IMAGEN PROMEDIO (MEDIA SIMPLE O PONDERADA)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def tracking_imagen(df, peso_col=None, window=3):
    #Tracking final de imagen del candidato.
        # Calcula imagen diaria (ponderada o simple)
        # Aplica rolling window
        # Grafica ambas con seaborn
    
    # 1. Ponderación
    if peso_col is None or peso_col not in df.columns:
        df["peso_temp"] = 1
        peso_col = "peso_temp"

    # 2. Imagen diaria (media ponderada)
    diaria = (
        df.groupby("fecha")
          .apply(lambda g: (g["imagen_candidato"] * g[peso_col]).sum() / g[peso_col].sum())
          .reset_index(name="imagen_diaria")
    )

    # Remover peso temporal si existiera - CORREGIR POST AGREGADO DE PESOS
    if "peso_temp" in df.columns:
        df.drop(columns=["peso_temp"], inplace=True)

    # 3. Rolling window
    diaria = diaria.sort_values("fecha")
    diaria["imagen_rolling"] = diaria["imagen_diaria"].rolling(window, min_periods=1).mean()

    # 4. Grafico
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=diaria, x="fecha", y="imagen_diaria", label="Imagen diaria", alpha=0.4)
    sns.lineplot(data=diaria, x="fecha", y="imagen_rolling", label=f"Rolling {window} días", linewidth=2)

    plt.title("Tracking de Imagen del Candidato")
    plt.xlabel("Fecha")
    plt.ylabel("Imagen (0–100)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return diaria




print ("no hay error hasta aca")

#PRUEBAS - IMPORTANTE BORRAR

# RESUMEN AUTOMÁTICO DEL TRACKING

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
    

#PRUEBA CROSSTAB IMAGEN Y EDAD
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

def plot_imagen_por_rango(df):

    #Crea un crosstab de imagen promedio del candidato por rango etario 
    # y genera un gráfico de barras.

    # Orden explícito (como en tu recodificación)
    orden = ["16-24", "25-35", "36-45", "46-55", "56-75", "+76"]
    cat = CategoricalDtype(categories=orden, ordered=True)

    df["rango_etario"] = df["rango_etario"].astype(cat)

    # Crosstab con media de imagen
    tabla = pd.crosstab(
        df["rango_etario"],
        columns="Imagen Promedio",
        values=df["imagen_candidato"],
        aggfunc="mean"
    )

    # Plot
    plt.figure(figsize=(10, 6))
    tabla["Imagen Promedio"].plot(kind="bar", edgecolor="black")

    plt.title("Imagen Promedio del Candidato por Rango Etario")
    plt.ylabel("Imagen Promedio (0-100)")
    plt.xlabel("Rango Etario")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

    return tabla

