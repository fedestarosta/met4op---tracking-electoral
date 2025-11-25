TRACKING ELECTORAL - MET4OP
Catalina Feltrin - Victoria Guerrero de la Vega - Federico Starosta

# Tracking Electoral

Este repositorio contiene el código y los datos para realizar un seguimiento (tracking) de encuestas electorales. El proyecto toma datos crudos de encuestas, los limpia y los pondera estadísticamente para mostrar cómo evoluciona la opinión pública día a día.

## Objetivo del Trabajo

El objetivo central de este proyecto es construir una herramienta capaz de unificar y corregir datos de encuestas electorales, aplicando un algoritmo de ponderación (Raking) que ajusta la muestra por sexo, edad y ubicación geográfica (usando datos del Censo 2022) para obtener estimaciones precisas sobre la imagen de los candidatos y la intención de voto a lo largo del tiempo.

## Qué se necesita

El proyecto está hecho en Python. Las librerías clave que usamos para procesar los datos, calcular los pesos y hacer los gráficos son:

* pandas y numpy: Para manejar las tablas de datos.
* balance: Para hacer el ajuste de la muestra (Raking).
* scipy: Para los tests estadísticos.
* matplotlib y seaborn: Para hacer los gráficos.

### Instalación rápida

Para instalar todo lo necesario, corre este comando en la terminal:

pip install -r requirements.txt

## Cómo funciona

El código principal está en src/main.py. Al ejecutarlo, el programa hace todo el trabajo sucio: carga los CSV de la carpeta data/, limpia los errores (como nombres de provincias mal escritos), calcula los pesos para que la muestra sea representativa y genera los gráficos de tendencias.

Para correrlo:

python src/main.py

## Estructura

* src/: Acá está todo el código (main.py y las funciones en procesamiento.py).
* data/: Acá van las encuestas y el archivo con los datos del Censo para ponderar.
* notebooks/: Pruebas y análisis sueltos.
* requirements.txt: La lista de librerías para instalar.
