#%%
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import balance as bal


#%%
print("Librerias importadas.")
print("Simulación iniciada, procedemos a importar Faker")

#%%
from faker import Faker
import pandas as pd
from random import randint, choice
from datetime import datetime, timedelta
from balance import raking

fake = Faker("es_AR")

#%%
from random import randint, choice
from datetime import datetime, timedelta

def generar_faker_encuestas(n_registros=500):

    # --- Estratos (Provincias de Argentina, limpias) ---
    estratos = [
        "Buenos Aires",
        "Catamarca",
        "Chaco",
        "Chubut",
        "Ciudad Autónoma de Buenos Aires",
        "Corrientes",
        "Córdoba",
        "Entre Ríos",
        "Formosa",
        "Jujuy",
        "La Pampa",
        "La Rioja",
        "Mendoza",
        "Misiones",
        "Neuquén",
        "Río Negro",
        "Salta",
        "San Juan",
        "San Luis",
        "Santa Cruz",
        "Santa Fe",
        "Santiago del Estero",
        "Tierra del Fuego, Antártida e Islas del Atlántico Sur",
    ]

    # --- Sexo ---
    sexos = ["Masculino", "Femenino"]

    # --- Nivel educativo ---
    niveles_educativos = [
        "Primario completo","Primario incompleto"
        "Secundario completo", "Secundario incompleto"
        "Terciario completo", "Terciario incompleto"
        "Universitario completo", "Universitario incompleto"
        "Ns/Nc",
    ]

    # --- Voto intención ---
    votos = ["Candidato_A", "Candidato_B", "Blanco", "Nulo", "Ns/Nc"]

    # --- Voto anterior ---
    votos_anteriores = [
        "Candidato_A",
        "Candidato_B",
        "No Fue A Votar",
        "Blanco",
        "Nulo",
        "Ns/Nc",
    ]

    # --- Rango de edad ---
    # 16 a 90 según estándar demográfico
    def generar_edad():
        return randint(16, 90)

    # --- Rango temporal ---
    inicio = datetime(2024, 1, 1)
    fin = datetime(2024, 2, 2)
    diferencia = (fin - inicio).days

    # --- Generación de registros ---
    registros = []
    for i in range(n_registros):
        registro = {
            "fecha": (inicio + timedelta(days=randint(0, diferencia))).strftime("%Y-%m-%d"),
            "encuesta_id": i + 1,  # único y secuencial
            "estrato": choice(estratos),
            "sexo": choice(sexos),
            "edad": generar_edad(),
            "nivel_educativo": choice(niveles_educativos),
            "integrantes_hogar": randint(1, 6),
            "imagen_candidato": randint(0, 100),
            "voto": choice(votos),
            "voto_anterior": choice(votos_anteriores),
        }
        registros.append(registro)

    df = pd.DataFrame(registros)
    return df

if __name__ == "__main__":
    df = generar_faker_encuestas(500)

    # Ruta absoluta al directorio /data
    DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "encuestas_falsas.csv"

    df.to_csv(DATA_PATH, index=False, encoding="utf-8")

    print("Archivo faker guardado en:")
    print(DATA_PATH)

#%%



if __name__ == "__main__":
    df = generar_faker_encuestas(500)

    # 1. FECHA
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    if df["fecha"].isna().any():
        print(f"{df['fecha'].isna().sum()} registros con fecha inválida.")
        df = df.dropna(subset=["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    print("Rango temporal de las encuestas:")
    print(f"Desde {df['fecha'].min().date()} hasta {df['fecha'].max().date()}")

    # 2. ENCUESTA_ID
    df["encuesta_id"] = pd.to_numeric(df["encuesta_id"], errors="coerce").astype(
        "Int64"
    )
    if df["encuesta_id"].isna().any():
        print(f"{df['encuesta_id'].isna().sum()} registros sin ID de encuesta.")
        df["encuesta_id"] = df["encuesta_id"].fillna(method="ffill").astype(int)

    print("\nCantidad de encuestas únicas registradas:")
    print(df["encuesta_id"].nunique())
#%% 

    #3. SEXO
# Normalización básica
df["sexo"] = (
    df["sexo"]
    .astype(str)
    .str.strip()
    .str.capitalize()     # "masculino" → "Masculino"
)

sexos_validos = ["Masculino", "Femenino"]

sexos_unicos = df["sexo"].unique().tolist()
sexos_invalidos = [s for s in sexos_unicos if s not in sexos_validos]

if sexos_invalidos:
    print(f"Categorías no válidas en 'sexo': {sexos_invalidos}")
    print("Estos registros serán eliminados para mantener consistencia.\n")
    df = df[ df["sexo"].isin(sexos_validos) ]
else:
    print("Sexo validado correctamente.")

# Mostrar distribución
print("\nDistribución de sexo:")
print(df["sexo"].value_counts())
#%% 
    # 3. EDAD
    # Convertir a numérico y controlar valores fuera de rango
df["edad"] = pd.to_numeric(df["edad"], errors="coerce")

# Detectar valores fuera de rango (menores de 16 o mayores de 90)
fuera_rango = df[(df["edad"] < 16) | (df["edad"] > 100)]

if not fuera_rango.empty:
    print(f" {len(fuera_rango)} registros con edad fuera de rango (menores de 16 o mayores de 90).")
    # Eliminar los registros fuera de rango
    df = df.drop(fuera_rango.index)
    print("✔️ Registros fuera de rango eliminados del DataFrame.")

# Calcular estadísticos descriptivos
print("\nDistribución de edad (años):")
print(df["edad"].describe().round(1))


#%% 
# 3.1. RECODIFICACIÓN DE EDAD EN RANGOS 

edad_rango = pd.cut(
    df["edad"],
    bins=[16, 24, 35, 45, 55, 75, float("inf")],
    labels=["16-24", "25-35", "36-45", "46-55", "56-75", "+76"],
    right=True,
    include_lowest=True
)

df["edad_rango"] = edad_rango

print("Variable 'edad_rango' creada correctamente.\n")
print("Distribución por rango etario (%):")
print(df["edad_rango"].value_counts(normalize=True).mul(100).round(2).sort_index())




#%%     # 4. VOTO ANTERIOR

# Validar categorías esperadas
votos_anteriores_validos = [
    "Candidato_A",
    "Candidato_B",
    "No Fue A Votar",
    "Blanco",
    "Nulo",
    "Ns/Nc"
]
votos_anteriores_unicos = df["voto_anterior"].unique().tolist()
votos_no_validos = [v for v in votos_anteriores_unicos if v not in votos_anteriores_validos]

if votos_no_validos:
    print(f" Categorías no reconocidas en 'voto_anterior': {votos_no_validos}")
else:
    print("Categorías de voto_anterior verificadas correctamente.")

# Distribución de casos por voto anterior
print("\nDistribución de casos por voto anterior (%):")
print(df["voto_anterior"].value_counts(normalize=True).mul(100).round(2))


#%% # 5. CANTIDAD DE INTEGRANTES EN EL HOGAR

df["integrantes_hogar"] = pd.to_numeric(df["integrantes_hogar"], errors="coerce")

fuera_rango = df[(df["integrantes_hogar"] <= 0) | (df["integrantes_hogar"] > 10)]

if not fuera_rango.empty:
    print(f" {len(fuera_rango)} registros fuera del rango permitido (1–10) en 'integrantes_hogar'.")
    print("Estos registros serán eliminados.\n")
    df = df.drop(fuera_rango.index)

# Distribución de frecuencias
print(" Distribución de la cantidad de integrantes en el hogar:")
print(df["integrantes_hogar"].value_counts().sort_index())

# Distribución porcentual
print("\n Distribución porcentual (%):")
print(df["integrantes_hogar"].value_counts(normalize=True).sort_index().mul(100).round(2))

#%% # 6. VOTO
# Validar categorías esperadas
votos_validos = ["Candidato_A", "Candidato_B", "Blanco", "Nulo", "Ns/Nc"]
votos_unicos = df["voto"].unique().tolist()
votos_no_validos = [v for v in votos_unicos if v not in votos_validos]

if votos_no_validos:
    print(f" Categorías no reconocidas en 'voto': {votos_no_validos}")
else:
    print("Categorías de voto verificadas correctamente.")

# Distribución de casos por voto
print("\nDistribución de intención de voto (%):")
print(df["voto"].value_counts(normalize=True).mul(100).round(2))

#%%
# 6. IMAGEN DEL CANDIDATO

df["imagen_candidato"] = (
    df["imagen_candidato"]
    .astype(str)
    .str.strip()
    .str.replace(",", ".", regex=False)
)

df["imagen_candidato"] = pd.to_numeric(df["imagen_candidato"], errors="coerce")

# 6.2 Eliminar valores fuera del rango 0–100
fuera_rango_img = df[(df["imagen_candidato"] < 0) | (df["imagen_candidato"] > 100)]

if not fuera_rango_img.empty:
    print(f" {len(fuera_rango_img)} registros fuera del rango permitido (0–100) en 'imagen_candidato'.")
    print("Estos registros serán eliminados.\n")
    df = df.drop(fuera_rango_img.index)

print(df["imagen_candidato"].describe())


# 6.4 Media diaria (imagen + cantidad de casos)
imagen_diaria = (
    df.groupby("fecha")
      .agg(
          imagen_media=("imagen_candidato", "mean"),
          n_casos=("imagen_candidato", "size")
      )
      .reset_index()
      .sort_values("fecha")
)

print("\nMedia diaria de imagen_candidato:")
print(imagen_diaria.head())


# 6.5 Agrupar cada 3 días con media ponderada
fecha_min = imagen_diaria["fecha"].min()
imagen_diaria["grupo_3d"] = ((imagen_diaria["fecha"] - fecha_min).dt.days // 3)

def media_ponderada_3d(grupo):
    num = (grupo["imagen_media"] * grupo["n_casos"]).sum()
    den = grupo["n_casos"].sum()
    return pd.Series({
        "fecha_inicio": grupo["fecha"].min(),
        "fecha_fin": grupo["fecha"].max(),
        "n_total_casos": den,
        "imagen_candidato_media_3d": num / den if den > 0 else np.nan
    })

imagen_3d = (
    imagen_diaria
    .groupby("grupo_3d")
    .apply(media_ponderada_3d)
    .reset_index(drop=True)
)

# Medir variabilidad de imagen

print("\nTracking de imagen_candidato cada 3 días (media ponderada):")
print(imagen_3d.head())

plt.figure(figsize=(12,5))

plt.bar(
    imagen_3d["fecha_inicio"].astype(str), # convertir fechas a string para el eje x
    imagen_3d["imagen_candidato_media_3d"],
    width=0.8
)

plt.title("Tracking de imagen del candidato (ventanas de 3 días)")
plt.xlabel("Fecha (inicio del grupo de 3 días)")
plt.ylabel("Imagen promedio ponderada")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.show()





