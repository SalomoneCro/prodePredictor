# prode_predictor — Predictor de fútbol internacional (Dixon-Coles)

Modelo de goles tipo **Dixon-Coles** (Poisson ataque/defensa) para predecir resultados
de partidos de selecciones. Construido originalmente para el **prode del Mundial 2026**
(fase de grupos), pero el núcleo es **reutilizable para cualquier prode** cambiando solo
las reglas de puntaje.

## Estado: modelo validado + tuneado (Etapa 4a)
Modelo Dixon-Coles **validado** por backtesting temporal (`v1.0-base-model` fue el freeze
sin tuning). Hiperparámetros tuneados: vida media decay = 3 años, amistosos a peso completo.

| Backtest | PTS/partido (modelo) | PTS/partido (baseline) | acc 1X2 | exacto |
|---|---|---|---|---|
| Mundiales 2014/18/22 (grupos, n=144) | **2.229** | 1.417 | 59.7% | 14.6% |
| Competitivos 2018-2025 (n=5564) | **2.266** | 1.732 | 61.5% | 14.1% |

## Estructura
| Archivo | Qué hace |
|---|---|
| `download_data.py` | Baja el dataset de Kaggle a `./data` (credenciales en `.env`) |
| `confederations.py` | Mapa equipo → confederación (grafía exacta del dataset) |
| `features.py` | **Etapa 0:** `load_data()` → (histórico, fixtures) con features |
| `model.py` | **Etapa 1:** clase `DixonColes` (fit + `score_matrix`) y utilidades de matriz |
| `validate.py` | **Etapa 2/3:** backtesting temporal + métricas + baselines |
| `explore.py`, `confed_check.py`, `tier_check.py` | Scripts de exploración (one-off) |
| `plan.md`, `exploration_report.md` | Plan por etapas y reporte de exploración |

## Cómo correrlo
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# completar KAGGLE_USERNAME / KAGGLE_KEY en .env
.venv/bin/python download_data.py     # baja datos
.venv/bin/python features.py          # chequeos de Etapa 0
.venv/bin/python model.py             # ajusta y muestra ejemplos
.venv/bin/python validate.py          # backtesting completo
```

## 🔁 Reutilizar en OTRO prode con reglas distintas
El modelo (att/def → `score_matrix`) es **independiente de las reglas**. Lo único atado al
prode es cómo se elige la jugada y cómo se puntúa, y ya está **parametrizado**:

```python
from model import DixonColes, best_bet
from validate import prode_points

model = DixonColes().fit(history, cutoff="2026-06-04")
M = model.score_matrix("Argentina", "Brazil", neutral=True)

# Prode actual: 6 exacto / 3 acierto 1X2
i, j, ev = best_bet(M, pts_exact=6, pts_outcome=3)

# Otro prode (ej: 5 exacto / 2 acierto 1X2): solo cambiás los números
i, j, ev = best_bet(M, pts_exact=5, pts_outcome=2)
```

`best_bet(M, pts_exact, pts_outcome)` devuelve el marcador que maximiza los puntos
esperados bajo esas reglas. `prode_points(pred, actual, pts_exact, pts_outcome)` puntúa.
La **matriz de marcadores** `M` ya tiene toda la información probabilística; cualquier
otra regla (acertar diferencia de gol, doble chance, etc.) se implementa como una nueva
función de selección sobre `M`, sin tocar el modelo.

## Decisiones de diseño
Ver `plan.md`. Resumen: localía solo para anfitriones, decay temporal, amistosos con peso
reducido (son el puente inter-confederación), matriz de marcadores **analítica** (no Monte
Carlo). Pendiente (Etapa 4+): tuning de hiperparámetros, término de confederación, ajuste
por brecha de nivel.
