"""Validación de las reglas del prode 2026 (3 exacto / 1.5 close / 1 resultado / 0).

1) Tests unitarios del Closeness Index contra los ejemplos oficiales del reglamento.
2) WC backtest (fase de grupos 2014/2018/2022) puntuado con las reglas nuevas,
   comparando la jugada óptima (best_bet_ci) contra heurísticas simples.
"""
import numpy as np
import pandas as pd

from features import load_data
from model import DixonColes, best_bet_ci, ci_points, closeness_index, most_likely_score


# ---------- 1) tests del reglamento ----------
def test_rules():
    cases = [
        # (pick, real, puntos_esperados, comentario)
        ((2, 1), (2, 1), 3.0, "exacto"),
        ((3, 1), (2, 1), 1.5, "1 gol de diferencia -> close"),
        ((2, 0), (2, 1), 1.5, "1 gol -> close"),
        ((3, 2), (2, 1), 1.5, "2 goles afuera pero MISMA dif de goles -> close"),
        ((4, 1), (2, 1), 1.0, "acierta ganador pero lejos -> result"),
        ((0, 1), (2, 1), 0.0, "ganador equivocado -> 0"),
        ((1, 1), (2, 1), 0.0, "empate vs gana local -> 0"),
    ]
    print("TESTS DEL REGLAMENTO (Closeness Index):")
    ok = True
    for pick, real, want, note in cases:
        got = ci_points(*pick, *real)
        ci = closeness_index(*pick, *real)
        flag = "OK " if abs(got - want) < 1e-9 else "FALLA"
        ok &= flag == "OK "
        print(f"  [{flag}] pick {pick[0]}-{pick[1]} vs real {real[0]}-{real[1]}: "
              f"{got} pts (CI={ci:.2f})  — {note}")
    print(f"  => {'TODOS OK' if ok else 'HAY FALLAS'}\n")
    return ok


# ---------- 2) backtest puntuado con las reglas nuevas ----------
def ci_points_pick(pick, real):
    return ci_points(pick[0], pick[1], real[0], real[1])


def wc_backtest_ci(history):
    wc = history[history["tournament"] == "FIFA World Cup"].copy()
    rows = []
    for year in (2014, 2018, 2022):
        ed = wc[wc["date"].dt.year == year].sort_values("date")
        group = ed.head(48)
        cutoff = group["date"].min() - pd.Timedelta(days=1)
        model = DixonColes().fit(history, cutoff=cutoff, verbose=False)
        for _, m in group.iterrows():
            actual = (int(m.home_score), int(m.away_score))
            M = model.score_matrix(m.home_team, m.away_team, m.neutral)
            bi, bj, ev, *_ = best_bet_ci(M)
            mi, mj, _ = most_likely_score(M)
            rows.append({
                "edition": year,
                "opt_pts": ci_points_pick((bi, bj), actual),     # jugada óptima
                "mode_pts": ci_points_pick((mi, mj), actual),    # marcador más probable
                "naive10_pts": ci_points_pick((1, 0), actual),   # siempre 1-0 local
                "ev": ev,
            })
        print(f"  Mundial {year}: {len(group)} partidos (entrenado hasta {cutoff.date()})")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    test_rules()

    history, _ = load_data()
    print("WC backtest puntuado con reglas 2026 (3/1.5/1/0):")
    df = wc_backtest_ci(history)
    n = len(df)
    print(f"\n{'='*56}\nFASE DE GRUPOS 2014/18/22  (n={n})\n{'='*56}")
    print(f"  jugada óptima (modelo): {df.opt_pts.mean():.3f} pts/part  "
          f"-> {df.opt_pts.sum():.0f} pts en {n}")
    print(f"  marcador más probable : {df.mode_pts.mean():.3f} pts/part")
    print(f"  baseline siempre 1-0  : {df.naive10_pts.mean():.3f} pts/part")
    print(f"\n  EV medio que predijo el modelo: {df.ev.mean():.3f} pts/part "
          f"(vs realizado {df.opt_pts.mean():.3f})")
    print("\n  Por edición (pts/part óptima vs moda vs 1-0):")
    print(df.groupby("edition")[["opt_pts", "mode_pts", "naive10_pts"]].mean().round(3).to_string())

    # extrapolación a los 72 partidos del prode
    pred = pd.read_csv("predictions_2026.csv")
    print(f"\n  Extrapolación a los 72 del prode 2026: EV total = {pred.ev_pts.sum():.0f} pts "
          f"(cota optimista), realista ~{df.opt_pts.mean()*72:.0f} pts según backtest.")
