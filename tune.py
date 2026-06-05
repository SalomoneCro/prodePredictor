"""Etapa 4 — Tuning de hiperparámetros.

Barre vida media del decay y peso de amistosos midiendo PUNTOS DE PRODE (métrica norte)
en un rolling backtest 2018-2025, con set de test FIJO para comparar configs sin sesgo.
Métrica secundaria: RPS. Cada config reentrena por año (sin fuga).
"""
import numpy as np
import pandas as pd

from features import load_data
from model import DixonColes, outcome_probs, best_bet
from validate import prode_points, rps

YEARS = range(2018, 2026)


def build_test_sets(history):
    """Test fijo: partidos competitivos entre selecciones 'establecidas' (>=25 PJ desde 2014)."""
    recent = history[history["date"] >= "2014-01-01"]
    appe = pd.concat([recent["home_team"], recent["away_team"]]).value_counts()
    established = set(appe[appe >= 25].index)
    tests = {}
    for y in YEARS:
        t = history[(history["date"].dt.year == y) & (~history["is_friendly"])
                    & history["is_fifa"]
                    & history["home_team"].isin(established)
                    & history["away_team"].isin(established)]
        tests[y] = t
    return tests


def eval_config(history, tests, half_life, friendly_w, window):
    pts, rpss, n = 0.0, 0.0, 0
    for y in YEARS:
        cutoff = pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=1)
        model = DixonColes().fit(history, cutoff=cutoff, half_life_days=half_life,
                                 friendly_weight=friendly_w, window_years=window,
                                 verbose=False)
        for row in tests[y].itertuples():
            M = model.score_matrix(row.home_team, row.away_team, row.neutral)
            i, j, _ = best_bet(M)
            pts += prode_points((i, j), (row.home_score, row.away_score))
            probs = np.array(outcome_probs(M))
            oidx = 0 if row.home_score > row.away_score else (1 if row.home_score == row.away_score else 2)
            rpss += rps(probs, oidx)
            n += 1
    return pts / n, rpss / n, n


if __name__ == "__main__":
    history, _ = load_data()
    tests = build_test_sets(history)
    n_total = sum(len(t) for t in tests.values())
    print(f"Set de test fijo: {n_total} partidos competitivos (2018-2025)\n")

    # --- Pase 1: vida media × peso amistosos (ventana fija 12) ---
    half_lives = [365, 547, 730, 1095, 1460]
    friendly_ws = [0.25, 0.5, 0.75, 1.0]
    print("PASE 1 — vida media (días) × peso amistosos  [ventana=12 años]")
    print(f"{'half_life':>10} {'friendly_w':>11} {'PTS/part':>9} {'RPS':>8}")
    grid = []
    for hl in half_lives:
        for fw in friendly_ws:
            p, r, _ = eval_config(history, tests, hl, fw, 12)
            grid.append((hl, fw, p, r))
            print(f"{hl:>10} {fw:>11} {p:>9.4f} {r:>8.4f}")
    best = max(grid, key=lambda g: g[2])
    print(f"\n  Mejor por PTS: half_life={best[0]}, friendly_w={best[1]} "
          f"-> {best[2]:.4f} pts/part (RPS {best[3]:.4f})")
    print(f"  (Default v1.0 era half_life=730, friendly_w=0.5)")

    # --- Pase 2: ventana de entrenamiento, con lo mejor del pase 1 ---
    print("\nPASE 2 — ventana de entrenamiento (años)  [half_life y friendly_w óptimos]")
    print(f"{'window':>8} {'PTS/part':>9} {'RPS':>8}")
    for w in [6, 8, 10, 12, 16]:
        p, r, _ = eval_config(history, tests, best[0], best[1], w)
        print(f"{w:>8} {p:>9.4f} {r:>8.4f}")
