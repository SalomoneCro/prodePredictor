# Reporte de Exploración — Dataset de Fútbol Internacional (para Prode Mundial 2026)

**Fuente:** Kaggle `martj42/international-football-results-from-1872-to-2017` (actualizado hasta jun-2026)
**Fecha del análisis:** 2026-06-04
**Objetivo:** entender los datos antes de modelar la predicción de la fase de grupos del Mundial 2026.

> ⚠️ Todavía **no** se construyó ningún modelo. Esto es solo exploración.

---

## 🔑 Hallazgo principal (el más importante para el prode)

**Los 72 partidos de la fase de grupos del Mundial 2026 ya están cargados en el dataset** como filas con `home_team`, `away_team`, `date`, `city`, `country` y `neutral`, pero **con `home_score`/`away_score` en blanco** (los 72 nulos del dataset son exactamente estos).

- 72 fixtures = 12 grupos × 6 partidos → toda la fase de grupos.
- Aparecen los 48 equipos clasificados, con fecha, sede y el flag `neutral` ya correcto.
- Los anfitriones (México, USA, Canadá) figuran con `neutral=False` cuando juegan de local (9 partidos); el resto `neutral=True` (63).

👉 **No hace falta armar el fixture a mano: el target a predecir ya viene en el archivo.** Filtramos `date > 2026-06-04 & home_score.isna()` y tenemos exactamente los partidos del prode.

```
date        home_team      away_team               city            country        neutral
2026-06-11  Mexico         South Africa            Mexico City     Mexico         False
2026-06-11  South Korea    Czech Republic          Zapopan         Mexico         True
2026-06-12  Canada         Bosnia and Herzegovina  Toronto         Canada         False
2026-06-12  United States  Paraguay                Inglewood       United States  False
2026-06-13  Qatar          Switzerland             Santa Clara     United States  True
...
```

---

## 1. Archivos y columnas

El dataset trae **4 CSVs**:

### `results.csv` — la tabla central (49.378 partidos)
| Columna | Significado |
|---|---|
| `date` | Fecha del partido (1872-11-30 → 2026-06-27) |
| `home_team` | Equipo local (o el "nominalmente local" si es neutral) |
| `away_team` | Equipo visitante |
| `home_score` | Goles del local (tiempo reglamentario + alargue, **sin** penales) |
| `away_score` | Goles del visitante |
| `tournament` | Tipo de competencia (199 valores distintos) |
| `city` | Ciudad sede |
| `country` | País sede |
| `neutral` | `True` si se jugó en cancha neutral (ninguno es local real) |

### `goalscorers.csv` (47.601 filas)
Goleadores por partido: `date, home_team, away_team, team, scorer, minute, own_goal, penalty`. Útil para features ofensivas/timing, pero arranca recién en 1916 y tiene huecos.

### `shootouts.csv` (675 filas)
Definiciones por penales: `date, home_team, away_team, winner, first_shooter`. Clave porque `results.csv` registra el empate del tiempo reglamentario; quién avanzó por penales está acá.

### `former_names.csv` (36 filas)
Mapeo de nombres históricos → actuales (ver sección 4).

---

## 2. Rango de fechas y totales

- **Primer partido:** 1872-11-30 (Escocia 0-0 Inglaterra).
- **Último:** 2026-06-27 (fixtures futuros del Mundial).
- **Total:** 49.378 partidos a lo largo de ~154 años.
- El volumen crece fuerte con el tiempo: ~3.000 partidos/década en los '60 → ~9.500/década en 2000s y 2010s.

| Década | Partidos |
|---|---|
| 1960 | 2.971 |
| 1980 | 5.025 |
| 2000 | 9.526 |
| 2010 | 9.787 |
| 2020 | 6.003 (en curso) |

**Implicancia:** datos antiguos abundan pero el fútbol cambió mucho. Para el modelo conviene ponderar/limitar a años recientes (p.ej. 2014+).

---

## 3. Tipos de torneo

199 tipos distintos. Los relevantes:

| Torneo | Partidos | Nota |
|---|---|---|
| Friendly | 18.301 (37%) | Amistosos: menor intensidad, cuidado al pesarlos igual |
| FIFA World Cup qualification | 8.771 | Eliminatorias mundialistas |
| UEFA Euro qualification | 2.824 | |
| African Cup of Nations qualification | 2.327 | |
| **FIFA World Cup (fase final)** | **1.036** | 1930→2026; lo más parecido a lo que predecimos |
| Copa América | 869 | |
| African Cup of Nations | 845 | |
| UEFA Nations League | 658 | Competitivo y reciente |
| Gold Cup / CONCACAF Nations League | 420 / 422 | |

- Partidos de **cualquier "qualification":** 15.928.
- Los amistosos son más de 1/3 del dataset → señal más "ruidosa" que partidos de torneo.

---

## 4. Calidad de datos

**Muy limpio en general.** Detalles:

- **Nulos en `results`:** solo `home_score`/`away_score` (72 cada uno) = exactamente los fixtures futuros del Mundial 2026. **No es un problema, es el target a predecir.**
- **Duplicados exactos:** 0. Duplicados por `(date, home, away)`: 1 (despreciable).
- **Scores:** 0 negativos. 268 partidos con diferencia ≥10 goles (goleadas reales, no errores — típicas de eliminatorias OFC/Asia).
- `goalscorers.csv`: 48 `scorer` nulos y 256 `minute` nulos (huecos menores).
- `shootouts.csv`: 429 `first_shooter` nulos (dato poco crítico).

### Cambios de nombre de países (⚠️ importante)
`former_names.csv` mapea 36 nombres históricos. **`results.csv` ya usa el nombre actual** (verificado: los 48 clasificados aparecen con su nombre moderno). Casos a tener en cuenta si se cruza data histórica:
- **Curaçao** ← Netherlands Antilles (hasta 2010)
- **DR Congo** ← Zaïre / Congo-Kinshasa / Belgian Congo
- **Russia** ← Soviet Union / CIS
- **Serbia** ← FR Yugoslavia / Serbia and Montenegro
- **Czechia/Czech Republic** — ojo: **Czechoslovakia** es una entidad aparte (no se mapea a Czech Republic en este archivo).
- Turkey, Germany, Ireland (Northern vs Republic), etc.

336 equipos distintos en total (incluye selecciones no-FIFA, islas, regiones).

---

## 5. Tasas Victoria / Empate / Derrota

**Global (perspectiva del equipo `home_team`):**
| Resultado | % |
|---|---|
| Gana local | 48.9% |
| Empate | 22.8% |
| Gana visitante | 28.2% |

- Goles promedio: local **1.76** vs visitante **1.18** → **2.94 goles/partido**.
- Empate ≈ 23% de forma bastante estable: es la clase más difícil de acertar y la base contra la que comparar cualquier modelo.

**Por tipo de torneo:**
| Torneo | n | Gana "local" | Empate | Gana "visit." |
|---|---|---|---|---|
| FIFA World Cup | 1.036 | 42.4% | 27.6% | 30.0% |
| Friendly | 18.301 | 47.3% | 25.1% | 27.6% |
| WC qualification | 8.771 | 50.8% | 21.1% | 28.0% |
| UEFA Euro | 388 | 39.4% | 26.3% | 34.3% |
| Copa América | 869 | 50.9% | 21.6% | 27.5% |

👉 En **Mundial la ventaja "local" se diluye** (42% vs 49% global) y los empates suben (27.6%) — lógico, porque casi todo se juega en cancha neutral (ver sección 6).

---

## 6. Efecto localía vs cancha neutral

| | Gana local | Empate | Gana visit. | Goles local | Goles visit. |
|---|---|---|---|---|---|
| **Con local real** (n=36.306) | 50.7% | 22.9% | 26.4% | 1.79 | 1.11 |
| **Cancha neutral** (n=13.072) | 44.0% | 22.8% | 33.2% | 1.67 | 1.37 |

- La ventaja de localía es **real y grande**: ~50.7% vs 26.4% con local real. En neutral el "local nominal" baja a 44% y el "visitante" sube a 33% (casi se empareja).
- **El 87.5% de los partidos de Mundial se juegan en cancha neutral.** Por eso, para predecir el Mundial hay que **modelar sobre la base neutral**, no la global con localía.
- **Excepción 2026:** los 3 anfitriones (México, USA, Canadá) SÍ juegan de local — son 9 de los 72 partidos de grupos, marcados `neutral=False`. Hay que darles el bonus de localía; al resto, no.

---

## 7. Cobertura de los 48 clasificados a 2026

✅ **Los 48 equipos matchean exactamente con los nombres del dataset** (cero problemas de naming).

- Promedio: **699 partidos/equipo** (rango 237 → 1.103).
- Promedio desde 2010: **196 partidos/equipo** (~20/año) → muestra reciente sólida para casi todos.
- Partidos donde **ambos** rivales son de los 48: **7.587** (1.503 desde 2014) → buen material de head-to-head entre clasificados.

**Equipos con MENOS historia (más incertidumbre para el modelo):**
| Equipo | Total | Primer partido | Desde 2010 |
|---|---|---|---|
| Cape Verde | 237 | 1959 | 132 |
| Bosnia and Herzegovina | 285 | 1995 | 160 |
| Uzbekistan | 356 | 1992 | 186 |
| Czech Republic | 362 | 1994 | 178 |
| Curaçao | 387 | 1924* | 120 |
| New Zealand | 418 | 1922 | 115 |

\* Curaçao "antiguo" es en realidad Netherlands Antilles; como Curaçao moderno tiene menos.

**Debutantes / proyectos jóvenes:** Cape Verde, Curaçao, Uzbekistan, Jordan son los de perfil más "nuevo" en este nivel → menos partidos top-level, predicción más ruidosa. Igual, todos tienen ≥110 partidos desde 2010, así que **alcanza para features recientes**.

Los grandes (Argentina, Brasil, Alemania, Inglaterra, etc.) tienen >1.000 partidos y ~210 desde 2010 — cobertura excelente.

---

## 8. Figuras generadas (`./figures/`)
- `partidos_por_anio.png` — crecimiento del volumen de partidos.
- `localia_vs_neutral.png` — comparación de resultados local vs neutral.
- `cobertura_48_desde2010.png` — partidos desde 2010 por equipo clasificado.

---

## 9. Qué parece más útil para predecir (insumos para la charla de modelado)

1. **El target ya está servido:** los 72 fixtures de grupos están en el archivo; solo hay que predecir scores/resultado.
2. **Modelar sobre base neutral**, con bonus de localía solo para los 3 anfitriones.
3. **Ponderar por recencia** (decaimiento temporal) y por **tipo de torneo** (competitivo > amistoso). Amistosos = 37% del data pero menos informativos.
4. **Fuerza de equipo:** construir un rating dinámico (Elo / Poisson bivariado / fuerza ataque-defensa) a partir del historial. Goles promedio 1.76–1.18 → un **modelo Poisson de goles** es un punto de partida natural y permite derivar P(V/E/D) y marcadores exactos (útil si el prode puntúa el resultado exacto).
5. **El empate (~23–28%)** es la clase difícil: cualquier modelo debe calibrarse para no subestimarlo.
6. **Penales:** `shootouts.csv` no afecta la fase de grupos (los empates quedan empates), así que para el prode de grupos se puede ignorar; importaría solo en eliminación directa.
7. **Datos faltantes para enriquecer (no están en este dataset):** ranking FIFA, valor de plantel, lesiones/convocados. Se podrían sumar después como features externas.

---

### Reproducir
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# completar KAGGLE_USERNAME / KAGGLE_KEY en .env
.venv/bin/python download_data.py   # baja a ./data
.venv/bin/python explore.py         # imprime stats + genera ./figures
```
