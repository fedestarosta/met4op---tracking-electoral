
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import geopandas as gpd

print ("Librerias importadas.")

def read_survey_file(path):

  path = Path(path)
<<<<<<< HEAD
# Verificar que la extensión sea .csv
if path.suffix.lower() != ".csv":
        raise ValueError(f"Formato no soportado: {path.suffix}. Solo se admiten archivos .csv.")
    
>>>>>>> 1546244178a8d5df454ac9176809403b14dc45f5
    