# La-Liga-Match-Predictor
This notebook takes a kaggle dataset that has data for 6 years of La Liga games (2019 to 2025) and with it I developed a model that can predict the outcome of this matches.

A continuación se presenta un borrador de un resumen para el archivo README de tu proyecto, diseñado para ser claro, atractivo y directo para quienes lo visiten en GitHub.

Resumen del Proyecto ⚽📊
Este repositorio contiene un modelo predictivo de machine learning para predecir los resultados de los partidos de la LaLiga de España. Utilizando un enfoque de data science, el proyecto tiene como objetivo identificar los factores clave que influyen en las victorias y derrotas de los equipos.

¿Qué se hizo?
Ingeniería de Características Avanzada: Se crearon métricas de rendimiento basadas en medias móviles para capturar el estado actual y el momentum de los equipos, evitando el data leakage.

Modelo de Machine Learning: Se implementó un modelo XGBoost, conocido por su robustez y alto rendimiento, para clasificar los partidos en victoria o derrota.

Análisis y Resultados: El modelo logró identificar que la diferencia de goles esperados (xg_diff) y la posesión efectiva son los principales factores que determinan el resultado de un partido.

Validación del Modelo: Se validó el rendimiento del modelo en las últimas 6 temporadas de la liga para asegurar su fiabilidad.

Uso
El notebook LaLiga_Predictor_Final.ipynb contiene todo el flujo de trabajo, desde la preparación de los datos hasta el entrenamiento, la evaluación y la interpretación del modelo.

Este proyecto es ideal para entusiastas del fútbol, analistas de datos y estudiantes de data science que busquen un ejemplo práctico de cómo aplicar modelos predictivos a datos deportivos.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Resumen del Proyecto ⚽📊 (Español)
Este repositorio contiene un modelo predictivo de machine learning para predecir los resultados de los partidos de la LaLiga de España. Utilizando un enfoque de data science, el proyecto tiene como objetivo identificar los factores clave que influyen en las victorias y derrotas de los equipos.

¿Qué se hizo?
Ingeniería de Características Avanzada: Se crearon métricas de rendimiento basadas en medias móviles para capturar el estado actual y el momentum de los equipos, evitando el data leakage.

Modelo de Machine Learning: Se implementó un modelo XGBoost, conocido por su robustez y alto rendimiento, para clasificar los partidos en victoria o derrota.

Análisis y Resultados: El modelo logró identificar que la diferencia de goles esperados (xg_diff) y la posesión efectiva son los principales factores que determinan el resultado de un partido.

Validación del Modelo: Se validó el rendimiento del modelo en las últimas 6 temporadas de la liga para asegurar su fiabilidad.

Uso
El notebook LaLiga_Predictor_Final.ipynb contiene todo el flujo de trabajo, desde la preparación de los datos hasta el entrenamiento, la evaluación y la interpretación del modelo.

Este proyecto es ideal para entusiastas del fútbol, analistas de datos y estudiantes de data science que busquen un ejemplo práctico de cómo aplicar modelos predictivos a datos deportivos.

Project Summary ⚽📊 (English)
This repository contains a machine learning model to predict the outcomes of Spanish LaLiga matches. Using a data science approach, the project aims to identify the key factors that influence team victories and losses.

What was done?
Advanced Feature Engineering: Performance metrics based on rolling averages were created to capture the current state and momentum of teams while avoiding data leakage.

Machine Learning Model: An XGBoost model, known for its robustness and high performance, was implemented to classify matches as either a win or a loss.

Analysis and Results: The model successfully identified that expected goals difference (xg_diff) and effective possession are the main factors determining a match's outcome.

Model Validation: The model's performance was validated using the last 6 seasons of league data to ensure its reliability.

Usage
The LaLiga_Predictor_Final.ipynb notebook contains the entire workflow, from data preparation to model training, evaluation, and interpretation.

This project is ideal for football enthusiasts, data analysts, and data science students looking for a practical example of how to apply predictive models to sports data.
