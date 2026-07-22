"""Etapa 5 — Predicción final de los 72 partidos de la fase de grupos del Mundial 2026.

Reentrena con TODO el histórico hasta hoy y, para cada fixture, calcula la matriz de
marcadores, P(1/X/2), los top-3 marcadores y la JUGADA ÓPTIMA por EV bajo las reglas
del prode 2026 (3 exacto / 1.5 close / 1 resultado / 0 errar).
Genera predictions_2026.csv + un resumen legible.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from prode import PREDICTIONS_DIR
from prode.features import load_data
from prode.model import DixonColes, outcome_probs, most_likely_score, best_bet_ci

PREDICTIONS_DIR.mkdir(exist_ok=True)
OUT = PREDICTIONS_DIR / "predictions_2026.csv"


def top_scores(M, k=3):
    flat = np.argsort(M, axis=None)[::-1][:k]
    out = []
    for f in flat:
        i, j = np.unravel_index(f, M.shape)
        out.append(f"{i}-{j} ({M[i, j]:.0%})")
    return out


def label(i, j):
    return "1" if i > j else ("X" if i == j else "2")


if __name__ == "__main__":
    history, fixtures = load_data()
    print("Reentrenando con todo el histórico hasta hoy...")
    model = DixonColes().fit(history, cutoff="2026-06-04")

    rows = []
    for m in fixtures.sort_values("date").itertuples():
        M = model.score_matrix(m.home_team, m.away_team, m.neutral)
        ph, pdr, pa = outcome_probs(M)
        bi, bj, ev, p_exact, p_close, p_res = best_bet_ci(M)
        mi, mj, mp = most_likely_score(M)
        lh, la = model._lambdas(m.home_team, m.away_team, m.neutral)
        rows.append({
            "date": m.date.date(),
            "home_team": m.home_team,
            "away_team": m.away_team,
            "host_home": "sí" if m.home_advantage else "",
            "p_home_%": round(ph * 100, 1),
            "p_draw_%": round(pdr * 100, 1),
            "p_away_%": round(pa * 100, 1),
            "lambda_home": round(lh, 2),
            "lambda_away": round(la, 2),
            "pick": f"{bi}-{bj}",
            "pick_1x2": label(bi, bj),
            "ev_pts": round(ev, 2),
            "p_exact_%": round(p_exact * 100, 1),
            "p_close+_%": round(p_close * 100, 1),
            "p_result+_%": round(p_res * 100, 1),
            "most_likely": f"{mi}-{mj}",
            "top3": " | ".join(top_scores(M)),
        })
    pred = pd.DataFrame(rows)
    pred.to_csv(OUT, index=False)
    print(f"Guardado: {OUT.name} ({len(pred)} partidos)\n")

    # resumen legible
    with pd.option_context("display.max_rows", None, "display.width", 200):
        show = pred[["date", "home_team", "away_team", "host_home",
                     "p_home_%", "p_draw_%", "p_away_%", "pick", "pick_1x2",
                     "ev_pts", "p_exact_%", "p_close+_%"]]
        print(show.to_string(index=False))

    # sanity / panorama
    print("\n" + "=" * 60)
    print("PANORAMA DE LAS JUGADAS")
    print("=" * 60)
    print("Distribución del 1X2 elegido:")
    print(pred["pick_1x2"].value_counts().to_string())
    print(f"\nMarcadores elegidos más frecuentes:")
    print(pred["pick"].value_counts().head(8).to_string())
    print(f"\nEV promedio por partido: {pred['ev_pts'].mean():.2f} pts "
          f"-> esperado total ~{pred['ev_pts'].sum():.0f} pts (cota optimista del modelo)")
    print(f"\nFavoritos más claros (mayor P del 1X2 elegido):")
    pred["pmax"] = pred[["p_home_%", "p_draw_%", "p_away_%"]].max(axis=1)
    top = pred.nlargest(5, "pmax")[["home_team", "away_team", "pick", "pmax"]]
    print(top.to_string(index=False))
    print(f"\nPartidos más parejos (menor P máxima):")
    bot = pred.nsmallest(5, "pmax")[["home_team", "away_team", "pick",
                                     "p_home_%", "p_draw_%", "p_away_%"]]
    print(bot.to_string(index=False))
