#%%
#Importamos librerias
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import balance as bal
import os

#llamamos al csv que se encuentra en input.py
path = "input/"
file_name = "encuestas.csv"
ruta_completa = path + file_name

if not os.path.exists(ruta_completa):
    print("El archivo", file_name, "no está en la carpeta input.")
else:
    df = pd.read_csv(ruta_completa)
    print(df.head())
