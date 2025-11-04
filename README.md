# ⚽ Predicting LaLiga Champions using Machine Learning and Monte Carlo Simulation

## 📘 Overview

This project applies **machine learning and probabilistic simulation** techniques to predict the outcome of future LaLiga seasons.  
Using detailed match data from the past **8 seasons**, the goal is to estimate which team is most likely to win the league in the following season — even when not all future match statistics are known.

The approach combines:
- Predictive modeling of match-level **goals** (home and away),
- A **Poisson-based Monte Carlo simulation** to generate thousands of possible seasons,
- Statistical aggregation and visualization of league standings and champion probabilities.

---

## 🧩 Project Motivation

In football analytics, most predictions rely on rich post-match data (goals, shots, xG, etc.), which is **not available before the season starts**.  
The challenge here is:  
> “How can we predict a future league outcome using only historical team performance and model-based expectations?”

This project tackles that challenge by modeling goal distributions and simulating entire seasons thousands of times to derive **probabilistic league outcomes**.

---

## 🧠 Methodology

### 1. Data Preparation

The dataset contains 8 seasons of LaLiga match data, including:
- `goals_for`, `goals_against`
- `expected_goals (xG)`, `shots`, `shots_on_target`
- `home_away` flag
- Other match context variables (stage of the season, opponent, etc.)

For model input, **rolling averages** (last 3 matches) were computed to represent recent team form.

---

### 2. Predictive Modeling

Instead of predicting “win/draw/loss” directly, the model predicts:
- `pred_home_goals`
- `pred_away_goals`

Two **XGBoost regressors** were trained — one for home goals and one for away goals — using the historical rolling averages and categorical match features.  
This *multioutput* setup captures more granular goal expectations and avoids overconfident classification biases (e.g. “Barcelona always wins”).

---

### 3. Poisson Simulation for Match Outcomes

Once expected goals (λ values) are predicted for each match, results are simulated using **Poisson-distributed random draws**:

\[
G_{home} \sim Poisson(\lambda_{home}), \quad G_{away} \sim Poisson(\lambda_{away})
\]

This produces realistic scorelines across thousands of simulated matches.

---

### 4. Monte Carlo Season Simulation

Each simulated season:
- Generates Poisson-distributed goals for all matches,
- Assigns points (3 for win, 1 for draw),
- Aggregates results into a league table.

This process is repeated **10,000 times**, producing a full distribution of outcomes for every team.

To make this efficient, the simulation was **fully vectorized** using NumPy:
- All matches and simulations are handled simultaneously via matrix operations.
- No Python loops are used for the core logic (`np.add.at` handles accumulation).

This allows millions of simulated matches in seconds.

---

### 5. Statistical Aggregation

For each team, the model computes:

| Metric | Meaning |
|---------|----------|
| `mean_points` | Average total points across all simulations |
| `p10`, `p50`, `p90` | 10th, 50th (median), and 90th percentiles of total points |
| `prob_champion` | Probability of finishing first in the league |

---

## 📊 Example Results

| Team | Mean Points | P10 | P50 | P90 | Prob. Champion |
|------|--------------|-----|-----|-----|----------------|
| Barcelona | 81.3 | 73 | 81 | 89 | 36% |
| Real Madrid | 79.8 | 71 | 80 | 88 | 33% |
| Atlético Madrid | 72.1 | 64 | 72 | 81 | 14% |
| Villarreal | 63.5 | 55 | 64 | 72 | 6% |

> **Interpretation:**  
> The simulation predicts Barcelona as the most likely champion (36% probability), but Real Madrid remains close behind.  
> The champion typically ends up around the **90th percentile** of its performance distribution (~89 points).

---

### 6. Why Vectorization Matters

Originally, simulations were performed with nested loops (per match × per simulation).  
The new vectorized approach replaces loops with matrix operations:

- Each match’s λ values expand into a 2D array: `(n_matches, n_sim)`
- Poisson random draws for all simulations are generated simultaneously
- Logical comparisons and point calculations occur elementwise
- Final team totals are aggregated with `np.add.at`, avoiding Python iteration

This improved performance by more than **40×**, enabling high-resolution probabilistic analysis.

---

## 📈 Visualizations

Some visual insights included in the notebook:

- **Distribution of simulated points per team** (histograms)
- **Probabilistic league table**
- **Champion probability bar chart**


---------------------------------------------------------------- Español -------------------------------------------------------------------------------------------

# ⚽ Predicción del Campeón de LaLiga usando Machine Learning y Simulación Monte Carlo

## 📘 Descripción General

Este proyecto aplica técnicas de **machine learning y simulación probabilística** para predecir el resultado de futuras temporadas de LaLiga.  
Utilizando datos detallados de partidos de las últimas **8 temporadas**, el objetivo es estimar qué equipo tiene mayor probabilidad de ganar la liga en la próxima temporada, incluso cuando no se dispone de información de partidos futuros.

La metodología combina:
- Modelos predictivos para estimar **goles esperados** (local y visitante),
- Una **simulación Monte Carlo basada en distribuciones de Poisson** para generar miles de posibles temporadas,
- Análisis estadístico y visualización de los resultados de la clasificación.

---

## 🧩 Motivación del Proyecto

En el análisis de fútbol, la mayoría de los modelos predictivos dependen de datos post-partido (goles, tiros, xG, etc.), los cuales **no están disponibles antes del inicio de la temporada**.  
El desafío principal fue:

> “¿Cómo podemos predecir el resultado de una liga futura usando solo información histórica y expectativas generadas por un modelo?”

Este proyecto responde a esa pregunta mediante la modelización de goles esperados y la simulación de temporadas completas miles de veces para obtener **distribuciones probabilísticas de resultados**.

---

## 🧠 Metodología

### 1. Preparación de Datos

El conjunto de datos contiene 8 temporadas de LaLiga, con variables como:
- `goles_a_favor`, `goles_en_contra`
- `expected_goals (xG)`, `tiros`, `tiros_a_puerta`
- `local_visitante`
- Etapa de la temporada, rival, entre otras.

Para capturar la forma reciente de cada equipo, se calcularon **promedios móviles de los últimos 3 partidos** (rolling mean).

---

### 2. Modelado Predictivo

En lugar de predecir directamente si un equipo ganará, empatará o perderá, el modelo predice:
- `goles_local`
- `goles_visitante`

Se entrenaron dos modelos **XGBoost** (uno para goles locales y otro para goles visitantes) utilizando las variables continuas y categóricas del dataset.  
Este enfoque *multioutput* ofrece predicciones más granulares y evita sesgos como “el Barcelona gana siempre”.

---

### 3. Simulación de Resultados con Distribución de Poisson

Una vez obtenidos los goles esperados (`λ_home`, `λ_away`), los resultados de los partidos se simulan asumiendo que los goles siguen una **distribución de Poisson**:

\[
G_{local} \sim Poisson(\lambda_{local}), \quad G_{visitante} \sim Poisson(\lambda_{visitante})
\]

Esto permite generar marcadores realistas y coherentes con la naturaleza aleatoria del fútbol.

---

### 4. Simulación de Temporadas (Monte Carlo)

Cada temporada simulada:
- Genera goles de forma aleatoria según las distribuciones de Poisson,
- Asigna puntos (3 por victoria, 1 por empate, 0 por derrota),
- Calcula la tabla de clasificación completa.

Este proceso se repite **10.000 veces**, produciendo una distribución completa de resultados para cada equipo.

Para lograrlo de forma eficiente, la simulación fue **totalmente vectorizada** usando NumPy:
- Todos los partidos y simulaciones se procesan al mismo tiempo mediante operaciones matriciales,
- Se eliminan los bucles de Python, reemplazándolos por operaciones como `np.add.at`.

Esto permite simular millones de partidos en segundos.

---

### 5. Agregación de Resultados

Para cada equipo, se calculan métricas estadísticas clave:

| Métrica | Descripción |
|----------|--------------|
| `mean_points` | Promedio de puntos totales en todas las simulaciones |
| `p10`, `p50`, `p90` | Percentiles 10, 50 (mediana) y 90 de puntos totales |
| `prob_champion` | Probabilidad de terminar primero en la liga |

---

## 📊 Resultados de Ejemplo

| Equipo | Media de Puntos | P10 | P50 | P90 | Prob. Campeón |
|--------|------------------|-----|-----|-----|----------------|
| Barcelona | 81.3 | 73 | 81 | 89 | 36% |
| Real Madrid | 79.8 | 71 | 80 | 88 | 33% |
| Atlético Madrid | 72.1 | 64 | 72 | 81 | 14% |
| Villarreal | 63.5 | 55 | 64 | 72 | 6% |

> **Interpretación:**  
> La simulación predice al Barcelona como el equipo con mayor probabilidad de ganar (36%), aunque el Real Madrid se mantiene muy cerca.  
> El campeón suele ubicarse alrededor del **percentil 90** de su distribución de rendimiento (~89 puntos).

---

### 6. Vectorización del Proceso

En la versión inicial, la simulación se hacía con bucles anidados (partido × simulación).  
La versión vectorizada reemplaza esos bucles por operaciones matriciales:

- Los λ de cada partido se expanden en matrices de forma `(n_partidos, n_simulaciones)`,
- Se generan todos los goles simulados en bloque usando `np.random.poisson`,
- Las comparaciones (victoria/empate/derrota) se realizan de forma vectorizada,
- Los puntos se acumulan con `np.add.at` sin usar loops.

El resultado es un incremento de rendimiento de más del **40×**, permitiendo realizar simulaciones más detalladas y rápidas.

---

## 📈 Visualizaciones

Algunos gráficos incluidos en el notebook:

- Distribución de puntos simulados por equipo (histogramas)
- Tabla de clasificación probabilística
- Probabilidades de campeonato

plt.ylabel('Frequency')
plt.show()
