#%%
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import balance as bal
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm

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

file_paths_a_cargar = ['data/encuestas_falsas.csv'] # Ejemplo de ruta
df = cargar_datos(file_paths_a_cargar)

# Le pedimos al df algunas informaciones basicas
print ("Resumen de tipos de datos")
print (df.info)        # Muestra el resumen de tipos de datos y NAs
print ("Columnas de df")
print(df.columns)  # Muestra la lista de columnas
print ("Mostrar 10 filas aleatorias")
df.sample(10)    # Muestra 10 filas aleatorias
print ("Primer vistazo estadisticas descriptivas")
print (df.describe)    # Muestra estadísticas descriptivas

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
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.sort_values("fecha").reset_index(drop=True)
    return df


# 3.2 ENCUESTA_ID
def limpiar_encuesta_id(df):
    df["encuesta_id"] = pd.to_numeric(df["encuesta_id"], errors="coerce")
    df["encuesta_id"] = df["encuesta_id"].ffill().astype(int)
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

    # Normalización básica
    df["voto_anterior"] = (
        df["voto_anterior"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Categorías fijas que estan siempre
    reemplazos_fijos = {
        "blanco": "Blanco",
        "nulo": "Nulo",
        "ns/nc": "Ns/Nc",
        "no sabe": "Ns/Nc",
        "no contesta": "Ns/Nc",
        "no fue a votar": "No Fue A Votar",
        "no fue": "No Fue A Votar"
    }

    # Reemplazar categorías fijas y equivalentes
    df["voto_anterior"] = df["voto_anterior"].replace(reemplazos_fijos)

    # Set de categorías fijas ya normalizadas
    categorias_fijas = set(reemplazos_fijos.values())

    # Normalizar candidatos no fijos
    df["voto_anterior"] = df["voto_anterior"].where(
        df["voto_anterior"].isin(categorias_fijas),
        df["voto_anterior"].str.title()
    )
    return df


# 3.7 INTEGRANTES HOGAR
def limpiar_integrantes_hogar(df):
    df["integrantes_hogar"] = pd.to_numeric(df["integrantes_hogar"], errors="coerce")
    df = df[(df["integrantes_hogar"] >= 1) & (df["integrantes_hogar"] <= 10)]
    return df


# 3.8 VOTO
def limpiar_voto(df):
    # Normalización básica
    df["voto"] = df["voto"].astype(str).str.strip().str.lower()

    # Categorías fijas que siempre existen
    reemplazos_fijos = {
        "blanco": "Blanco",
        "nulo": "Nulo",
        "ns/nc": "Ns/Nc",
        "no sabe": "Ns/Nc",
        "no contesta": "Ns/Nc"
    }

    # Reemplazar lo fijo
    df["voto"] = df["voto"].replace(reemplazos_fijos)

    # Candidatos: todo lo que no sea fijo, lo pasamos a formato Nombre Propio
    categorias_fijas = set(reemplazos_fijos.values())

    df["voto"] = df["voto"].where(
        df["voto"].isin(categorias_fijas),
        df["voto"].str.title()
    )

    return df
# 3.9 IMAGEN
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
    # Reemplazos frecuentes
    reemplazos = {
        "no sabe": "ns/nc",
        "no contesta": "ns/nc",
        }
    # Aplicar reemplazos básicos
    df["nivel_educativo"] = df["nivel_educativo"].replace(reemplazos)
    # Convertir a formato estandarizado: nombre propio
    df["nivel_educativo"] = df["nivel_educativo"].where(
        df["nivel_educativo"].isin(categorias_validas),
        "Missing"
    )
    # igualar formato por las dudas
    df["nivel_educativo"] = df["nivel_educativo"].str.title()

    return df


# 3.11 ESTRATO (provincias)
def limpiar_estrato(df):
    # 1. Normalización inicial a minúsculas para limpieza
    df["estrato"] = df["estrato"].astype(str).str.strip().str.lower()

    # 2. Reemplazos de variaciones comunes a un estándar minúscula
    reemplazos = {
        "caba": "ciudad autonoma de buenos aires",
        "capital federal": "ciudad autonoma de buenos aires",
        "tierra del fuego": "tierra del fuego, antartida e islas del atlantico sur",
        "tdf": "tierra del fuego, antartida e islas del atlantico sur"
    }
    df["estrato"] = df["estrato"].replace(reemplazos)

    # 3. Diccionario de Mapeo FINAL (De minúscula -> Formato EXACTO del CSV de Pesos)
    # Esto garantiza que coincida con tu archivo "data/pesos_fuente_censo2022.csv"
    mapeo_exacto = {
        "buenos aires": "Buenos Aires",
        "catamarca": "Catamarca",
        "chaco": "Chaco",
        "chubut": "Chubut",
        "ciudad autonoma de buenos aires": "Ciudad Autonoma de Buenos Aires", # Nota la 'd' minúscula
        "corrientes": "Corrientes",
        "córdoba": "Córdoba",
        "entre ríos": "Entre Ríos",
        "formosa": "Formosa",
        "jujuy": "Jujuy",
        "la pampa": "La Pampa",
        "la rioja": "La Rioja",
        "mendoza": "Mendoza",
        "misiones": "Misiones",
        "neuquén": "Neuquén",
        "río negro": "Río Negro",
        "salta": "Salta",
        "san juan": "San Juan",
        "san luis": "San Luis",
        "santa cruz": "Santa Cruz",
        "santa fe": "Santa Fe",
        "santiago del estero": "Santiago del Estero",
        "tierra del fuego, antartida e islas del atlantico sur": "Tierra del Fuego, Antartida e Islas del Atlantico Sur",
        "tucuman": "Tucuman"
    }

    # 4. Aplicar el mapeo
    # Si algo no está en el mapa (no es provincia válida), se convierte en NaN o "Missing"
    df["estrato"] = df["estrato"].map(mapeo_exacto).fillna("Missing")

    return df    
# TRATAMIENTO DE NaN
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

#INCORPORACION DE PESOS - Ponderacion
df_poblacion = pd.read_csv("data/pesos_fuente_censo2022.csv", decimal='.')
def peso_col(df, df_poblacion):
    
    #Aplica Raking (Iterative Proportional Fitting) de forma manual con Pandas.
    #Asegura la creación de la columna 'peso' con valor 1.0 como fallback si el Raking falla.
    
    # 1. Inicialización
    df.columns = df.columns.str.lower()  # Normalizar nombres a minúsculas
    df['peso'] = 1.0                     # Inicializar la columna de peso en 1.0 (este es el peso de fallback)
    
    # Identificar las variables de raking disponibles en el archivo de población
    variables_a_ajustar = df_poblacion['variable'].unique()
    
    # Filtrar solo las variables que existen en el DataFrame (df)
    variables_validas = [v for v in variables_a_ajustar if v in df.columns]
    
    if not variables_validas:
        print("Advertencia: No se encontraron variables de cruce válidas. Se retornan pesos = 1.")
        return df

    print(f"Iniciando Raking manual sobre: {variables_validas}")
    
    # Guardamos el total de casos para mantener el peso total de la muestra constante
    total_muestra = len(df)
    
    # 2. Bloque TRY/EXCEPT para el Raking Iterativo
    try:
        # Ciclo Iterativo (IPF): 50 iteraciones para asegurar convergencia
        for i in range(50):
            
            for var in variables_validas:
                
                # a) Obtener el Target para esta variable
                datos_target = df_poblacion[df_poblacion['variable'] == var]
                target_props = dict(zip(datos_target['valor'].astype(str), datos_target['proporcion']))
                
                # b) Calcular la distribución actual de la muestra (suma de pesos)
                peso_por_cat = df.groupby(var, observed=True)['peso'].sum()
                total_peso_actual = peso_por_cat.sum()
                current_props = peso_por_cat / total_peso_actual
                
                # c) Calcular Factores de Ajuste
                factores = {}
                for cat in current_props.index:
                    if cat in target_props and current_props[cat] > 0:
                        factores[cat] = target_props[cat] / current_props[cat]
                    else:
                        # Categoría sin objetivo o con 0 casos, no se ajusta (factor = 1.0)
                        factores[cat] = 1.0
                
                # d) Aplicar el ajuste a los pesos (SOLUCIONES A CONFLICTOS DE TIPOS)
                factores_ajuste = df[var].astype(str).map(factores).fillna(1.0)
                df['peso'] = df['peso'] * factores_ajuste.astype(float)
            
            # e) Re-normalización: Asegurar que la suma de pesos sea igual al total de casos original
            suma_pesos_final = df['peso'].sum()
            if suma_pesos_final > 0:
                factor_correccion_total = total_muestra / suma_pesos_final
                df['peso'] = df['peso'] * factor_correccion_total

        print("Raking manual finalizado.")
    
    except Exception as e:
        # Si ocurre cualquier error, el peso ya está en 1.0 y se imprime el error.
        print(f"Error CRÍTICO durante el Raking Manual: {e}")
        print("Se ha detenido el Raking. Se retornan pesos = 1.0.")
        # No se necesita un 'else' porque la inicialización y el return ya lo manejan.
        pass

    return df

#TRATAMIENTO DE VARIABLES CLAVE

#DESCRIPTIVOS DE IMAGEN
#Incluye: moda, mediana, media, cuartiles, varianza, desvío estándar y CV.

#IMAGEN CANDIDATO PROCESAMIENTO
def tracking_imagen(df, peso_col=None, window=3):
   # Tracking final de imagen del candidato.
        #- Media ponderada
        #- Varianza ponderada
        #- SD ponderada
        #- Error estándar
        #- Intervalo de confianza 95%
        #- Rolling window
        #- Gráfico


        # Estadísticos diarios
    def stats_diarios(g):
        w = g[peso_col]
        x = g["imagen_candidato"]

        # Media ponderada
        mean = (x * w).sum() / w.sum()

        # Varianza ponderada
        var = (w * (x - mean)**2).sum() / w.sum()

        # SD ponderada
        sd = np.sqrt(var)

        # n casos
        n = len(g)

        # Error estándar
        se = sd / np.sqrt(n) if n > 1 else np.nan

        # Intervalo de confianza del 95%
        ic_inf = mean - 1.96 * se if n > 1 else np.nan
        ic_sup = mean + 1.96 * se if n > 1 else np.nan

        # Recorte por escala 0–100
        if n > 1:
            ic_inf = max(0, ic_inf)
            ic_sup = min(100, ic_sup)

        return pd.Series({
            "imagen_diaria": mean,
            "sd_diaria": sd,
            "n_casos": n,
            "se_diaria": se,
            "ic_inf": ic_inf,
            "ic_sup": ic_sup
        })

        # Aplicación
    diaria = (
         df.groupby("fecha")
            .apply(stats_diarios, include_groups=False)
            .reset_index()
            .sort_values("fecha")
        )

    # 3. Rolling window
    diaria["imagen_rolling"] = diaria["imagen_diaria"].rolling(
        window,
        min_periods=1
        ).mean()

    # 4. Gráfico
    fig = plt.figure(figsize=(12, 6))

    # Banda del IC 95%
    plt.fill_between(
        diaria["fecha"],
        diaria["ic_inf"],
        diaria["ic_sup"],
        alpha=0.2,
        color="lightblue",
        label="IC 95% (diario)"
    )

    # Línea imagen diaria
    sns.lineplot(
        data=diaria,
        x="fecha",
        y="imagen_diaria",
        label="Imagen diaria",
        alpha=0.4
    )

    # Línea rolling window
    sns.lineplot(
        data=diaria,
        x="fecha",
        y="imagen_rolling",
        label=f"Rolling {window} días",
        linewidth=2
    )

    plt.title("Tracking de Imagen del Candidato")
    plt.xlabel("Fecha")
    plt.ylabel("Imagen (0–100)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return diaria,fig 

#INTENCION DE VOTO
def tracking_voto(df, peso_col, umbral_minimo=0):
#Tracking simple de intencion de voto
    #ponderado por fecha.
    #Incluye umbral mínimo de casos ponderados por fecha.
    
    # Suma de pesos por voto y fecha    
    diario = (
        df.groupby(["fecha", "voto"])[peso_col]
          .sum()
          .reset_index(name="peso_sumado")
    )
        

    # Total de pesos por fecha (casos efectivos)
    total_por_fecha = diario.groupby("fecha")["peso_sumado"].transform("sum")

    diario["total_fecha"] = total_por_fecha

    # Proporción ponderada
    diario["porcentaje"] = diario["peso_sumado"] / diario["total_fecha"]

    # Filtrar fechas que NO cumplen con el mínimo de casos
    diario_filtrado = diario[diario["total_fecha"] >= umbral_minimo].copy()

    return diario_filtrado

    #Grafico
import seaborn as sns 
import matplotlib.pyplot as plt
def grafico_tracking_voto(tabla_voto):
    #Genera el gráfico de líneas de tracking de intención de voto."""

        fig = plt.figure(figsize=(12, 6))

        sns.lineplot(
            data=tabla_voto,
            x="fecha",
            y="porcentaje",
            hue="voto",
            linewidth=2,
            marker="o"
        )

        plt.title("Tracking de Intención de Voto")
        plt.xlabel("Fecha")
        plt.ylabel("Porcentaje ponderado")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.show()

        return fig

#TRANSFERENCIA DE VOTO
def heatmap_transferencia(df, peso_col):
    #Filas = voto anterior
    #Columnas = voto actual
    #Valores = proporción ponderada

    # tabla de contingencia ponderada
    tabla = (
        df.pivot_table(
            index="voto_anterior",
            columns="voto",
            values=peso_col,
            aggfunc="sum",
            fill_value=0
        )
    )

    # convertir a proporciones por fila (voto anterior)
    tabla_prop = tabla.div(tabla.sum(axis=1), axis=0)

    # gráfico
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        tabla_prop,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        ax=ax
    )

    plt.title("Transferencia de voto (proporciones por voto anterior)")
    plt.ylabel("Voto anterior")
    plt.xlabel("Voto actual")

    plt.tight_layout()

    return tabla_prop, fig

    print ("no hay error hasta aca")

# CROSSTAB IMAGEN Y EDAD
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
    fig = plt.figure(figsize=(10, 6))
    tabla["Imagen Promedio"].plot(kind="bar", edgecolor="black")

    plt.title("Imagen Promedio del Candidato por Rango Etario")
    plt.ylabel("Imagen Promedio (0-100)")
    plt.xlabel("Rango Etario")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    return tabla, fig

# REGRESIÓN LINEAL SIMPLE: Imagen vs edad
def regresion_imagen_edad(df):
    
    #Regresión Lineal Simple Ponderada
    
    # Crear variable binaria: vota A (1) vs no vota A (0)
    df = df.copy()
    df["voto_A"] = (df["voto"] == "Candidato_A").astype(int)

    # 2. Variables X (Edad) e Y (Voto A)
    x = df["edad"]
    y = df["voto_A"]
    
    # Obtener pesos (si existen)
    weights = df[peso_col] if peso_col and peso_col in df.columns else None

    # 3. Cálculo de correlación de Pearson
    pearson = x.corr(y)
    print("CORRELACIÓN EDAD ↔ VOTO A")
    print(f"Coef. Pearson: {pearson:.4f}")

    # 4. Ajuste del modelo OLS
    X = sm.add_constant(x)        # agrega intercepto
    
    # Aplicar pesos al modelo OLS
    if weights is not None:
         modelo = sm.Logit(y, X, weights=weights).fit()
    else:
         modelo = sm.Logit(y, X).fit()

    print("RESUMEN REGRESIÓN LINEAL (Voto_A ~ Edad)")
    print(modelo.summary())

    # 5. Generar recta de regresión
    intercepto, pendiente = modelo.params
    x_vals = np.linspace(x.min(), x.max(), 100)
    y_vals = intercepto + pendiente * x_vals

    # 6. Gráfico - FIX para el TypeError y uso de pesos
    fig, ax = plt.subplots(figsize=(10,6)) # Crear figura y ejes
    
    if weights is not None:
         # Plot scatterplot usando pesos para el tamaño. 
         # Se usa legend=False para evitar el TypeError y luego se añade manualmente.
         sns.scatterplot(
             x=x, 
             y=y, 
             size=weights, 
             sizes=(20, 200), # Rango de tamaño de los marcadores
             alpha=0.4, 
             color="gray", 
             ax=ax,
             legend=False # Deshabilita la leyenda automática que causa el error
         )
         
         # Añadir una entrada de leyenda manual para los puntos
         ax.scatter([], [], s=100, alpha=0.4, color="gray", label="Datos observados (Ponderado)")
    else:
         # Si no hay pesos, simple scatterplot
         sns.scatterplot(
             x=x, 
             y=y, 
             alpha=0.4, 
             color="gray", 
             label="Datos observados (No Ponderado)", 
             ax=ax
         )

    # Plotear la recta de regresión
    ax.plot(x_vals, y_vals, color="blue", linewidth=2,
             label=f"Recta de regresión\npendiente={pendiente:.3f}")

    ax.set_title("Relación entre Edad del Votante e Intención de Voto (Candidato A)")
    ax.set_xlabel("Edad del votante (años)")
    ax.set_ylabel("Probabilidad de votar A (0-1)")
    ax.grid(alpha=0.3)
    ax.legend() # Genera la leyenda con las entradas manuales
    plt.tight_layout()
    # plt.show()

    return modelo, fig
    
