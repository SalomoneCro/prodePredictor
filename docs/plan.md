# Plan de Implementación — Predictor Mundial 2026 (Prode)

> Documento de plan. **Todavía no se escribe el modelo.** Esto define el orden, las
> decisiones ya tomadas y los criterios de validación para aprobar antes de codear.

## Objetivo
Predecir el **marcador** de los **72 partidos de la fase de grupos** del Mundial 2026
(ya cargados en `results.csv` como filas con score nulo) para maximizar puntos en el prode.

**Puntaje del prode:** `+6` marcador exacto · `+3` acertar ganador/empate (1X2) · `0` errar.
**Métrica norte:** puntos esperados del prode (no accuracy ni log-loss a secas).

## Decisiones de diseño ya acordadas
- **Modelo:** Dixon-Coles (Poisson de goles con ataque/defensa por equipo).
- **λ del partido:** `λ_local = exp(μ + att_local − def_visit + ventaja_local)`, idem visitante.
- **Resultado:** matriz de marcadores analítica (sin Monte Carlo) → de ahí salen P(1X2), P(exacto) y la jugada óptima.
- **Localía:** bonus **solo para anfitriones** (México, USA, Canadá); resto cancha neutral.
- **Recencia:** decay temporal exponencial (partidos viejos pesan menos).
- **Amistosos:** peso reducido, **no** eliminados (son el puente entre confederaciones).
- **Conectividad:** término de confederación desde el modelo base.
- **Brecha de nivel:** ajuste secundario, **solo si** los residuos lo justifican.
- **Ranking FIFA:** fuera por ahora (reconsiderable si falla lo cross-confederación).
- **Datos de entrenamiento:** resultados hasta lo más reciente posible (jun-2026), con decay.

---

## Etapa 0 — Pipeline de datos y features
**Objetivo:** dejar una tabla de partidos lista para modelar y el set de los 72 a predecir.

- [ ] Cargar `results.csv`, separar **histórico** (score no nulo) de **fixtures a predecir** (score nulo, `date > 2026-06-04`).
- [ ] Mapa de **confederación** por equipo (ampliar el de los 48 a todos los rivales del histórico).
- [ ] Features por partido: `is_neutral`, `host_home` (flag anfitrión local), `tournament_weight` (amistoso vs competitivo), `days_ago` (para decay), confederación de cada lado.
- [ ] Sanity checks: nombres consistentes, sin nulos donde no corresponde, los 48 presentes.

**Entregable:** `features.py` + tabla cacheada. **Criterio de avance:** los 72 fixtures quedan aislados y sin fuga de datos.

## Etapa 1 — Modelo base Dixon-Coles
**Objetivo:** primera versión funcional, simple y honesta.

- [ ] Ajuste por **máxima verosimilitud** (Poisson) de `att_i`, `def_i`, `μ`, `ventaja_local`.
  - Implementación vía regresión de Poisson con dummies de equipo (statsmodels) o `scipy.optimize`.
  - Restricción de identificabilidad (p.ej. `mean(att)=0`).
- [ ] **Decay temporal** `w = exp(−ξ · días_ago)` como peso de cada partido (ξ fijo inicial, se tunea después).
- [ ] **Peso por torneo** (amistoso < competitivo) como multiplicador del peso.
- [ ] **Corrección Dixon-Coles** `ρ` para calibrar 0-0 / 1-0 / 0-1 / 1-1.
- [ ] Función `score_matrix(home, away, neutral, host)` → grilla de probabilidades.

**Entregable:** `model.py` con `fit()` y `score_matrix()`. **Criterio:** produce matrices coherentes para partidos conocidos (p.ej. Argentina vs equipo chico → favorito claro).

## Etapa 2 — Validación (alineada al prode)
**Objetivo:** medir bien antes de creerle al modelo. Sin esto no se toca nada más.

- [ ] **Backtesting temporal:** entrenar con datos hasta fecha T, predecir partidos posteriores. Repetir sobre **Mundiales pasados** (2014, 2018, 2022) y eliminatorias recientes — escenarios parecidos al objetivo (cancha neutral, cruces inter-confederación).
- [ ] **Cero fuga:** en cada fold, el decay y los att/def usan solo info anterior a la fecha del partido.
- [ ] **Métricas:**
  - **Puntos del prode** (la que importa) con la jugada óptima por EV.
  - **RPS** (Ranked Probability Score) y **log-loss** para calibración del 1X2.
  - **Brier / calibración** de marcadores frecuentes.
- [ ] **Baselines para comparar:** (a) siempre el favorito 1-0, (b) tabla de frecuencias histórica, (c) ranking-only si quisiéramos. El modelo tiene que ganarles.

**Entregable:** `validate.py` + reporte de métricas. **Criterio de avance:** el base supera a los baselines en puntos de prode.

## Etapa 3 — Jugada óptima (selección de marcador)
**Objetivo:** convertir cada matriz en la predicción que maximiza puntos esperados.

- [ ] Implementar `best_bet(matrix)`:
  `EV(s) = 3·P(1X2 de s) + 3·P(marcador exacto s)` → elegir el `s` que maximiza.
- [ ] Verificar el comportamiento esperado: partidos desparejos → asegurar el 1X2; partidos parejos de pocos goles → apostar al exacto (1-0 / 1-1).

**Entregable:** función de selección + chequeo sobre el ejemplo trabajado (λ=1.5/1.1 → debe elegir 1-0).

## Etapa 4 — Ajustes finos (solo los que los datos pidan)
**Objetivo:** mejorar sobre el base, midiendo cada cambio contra Etapa 2. Agregar de a uno.

- [ ] **Tuning de hiperparámetros:** `ξ` (decay), peso de amistosos, `ρ` (DC). Grid/optimización contra la métrica de prode en backtest.
- [ ] **Término de confederación:** offset promedio por confederación estimado de los cruces; estabiliza a equipos con pocos puentes (Cape Verde, DR Congo, Curaçao).
- [ ] **Brecha de nivel:** mirar residuos por brecha; si subestima goleadas → término de interacción **o** pasar a **Binomial Negativa** (sobre-dispersión). Peso secundario.
- [ ] **Shrinkage** para equipos con poca data (regularización de att/def).

**Entregable:** changelog de qué se probó y cuánto sumó (o no) en puntos de prode. **Regla:** si un ajuste no mejora el backtest, se descarta.

## Etapa 5 — Predicción final y robustez
**Objetivo:** generar las 72 jugadas para cargar en el prode.

- [ ] Reentrenar con **todo** el histórico hasta el corte (con decay) y predecir los 72 fixtures.
- [ ] Tabla final: por partido → P(1X2), top-3 marcadores con probabilidad, **jugada elegida** y EV.
- [ ] Chequeo de sensatez (favoritos, anfitriones de local) y análisis de incertidumbre por partido.
- [ ] (Opcional) ensamble / suavizado si dos configuraciones del modelo discrepan mucho.

**Entregable:** `predictions_2026.csv` + un resumen legible.

---

## Riesgos y cómo los manejamos
| Riesgo | Mitigación |
|---|---|
| Fuga de datos en backtest | Splits estrictamente temporales; decay solo con pasado |
| Escala cross-confederación floja | Término de confederación; revisar residuos; (ranking como plan B) |
| Sobre-ajuste con muchos parámetros | Shrinkage + validar cada ajuste en backtest |
| Goleadas mal calibradas | Residuos por brecha → interacción o Binomial Negativa |
| Equipos con poca data (debutantes) | Shrinkage hacia la media; peso de amistosos |
| Brecha jun-2026 sin ranking actualizado | No dependemos del ranking; usamos resultados hasta jun-2026 |

## Orden de ejecución sugerido
`Etapa 0 → 1 → 2 → 3` (mínimo viable que ya predice y se valida) → `4 → 5` (mejoras incrementales).
Cada etapa se aprueba antes de seguir.
