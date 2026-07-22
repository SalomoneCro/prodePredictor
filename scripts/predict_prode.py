"""Predictor parametrizable por reglas de prode — reusa el modelo Dixon-Coles ya entrenado.

Soporta los dos tipos de reglamento que aparecieron hasta ahora:
  - 'exact'   : acertar 1X2 da `outcome` pts; acertar el marcador exacto da `exact` pts
                en total (NO se suman). Ej. prode v1: exact=6, outcome=3.
                Ej. aditivo: el exacto da outcome+bonus, así que exact=outcome+bonus.
  - 'close'   : 3 exacto / 1.5 close / 1 resultado / 0 (prode 2026 close points).

Uso:
    python predict_prode.py --rule exact --outcome 2 --exact 5 --out preds_2y5.csv
    python predict_prode.py --rule close
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from prode import PREDICTIONS_DIR
from prode.features import load_data
from prode.model import DixonColes, outcome_probs, most_likely_score, best_bet, best_bet_ci

ABBR = {
    "Bosnia and Herzegovina": "Bosnia", "United States": "USA",
    "Czech Republic": "Czechia", "South Africa": "S.Africa",
    "South Korea": "S.Korea", "New Zealand": "N.Zealand",
    "Saudi Arabia": "S.Arabia", "Cape Verde": "C.Verde",
    "Ivory Coast": "I.Coast", "Switzerland": "Switzerl.",
    "Netherlands": "Netherl.",
}


def short(n):
    return ABBR.get(n, n)


def label(i, j):
    return "1" if i > j else ("X" if i == j else "2")


def extra_time_matrix(M, lh, la, et_frac=1 / 3.0):
    """Matriz de marcadores a 90'+alargue (fase eliminatoria) a partir de la de 90'.

    Los partidos que NO terminan empatados a 90' quedan definidos ahí. Los empatados
    juegan 30' de alargue (~1/3 del ritmo de gol de los 90'): se suman goles Poisson
    independientes a cada lado. Si siguen empatados a 120', se define por penales, que
    NO cuentan para el prode -> ese marcador queda como empate (la X sigue siendo
    jugada válida). Devuelve la matriz del marcador final (90+30), misma forma que M.
    """
    n = M.shape[0]
    pet_h = poisson.pmf(np.arange(n), lh * et_frac)
    pet_a = poisson.pmf(np.arange(n), la * et_frac)
    M120 = np.zeros_like(M)
    for i in range(n):
        for j in range(n):
            p = M[i, j]
            if p == 0:
                continue
            if i != j:                       # decidido en los 90'
                M120[i, j] += p
            else:                            # empate a 90' -> alargue
                for a in range(n - i):       # goles extra del local (capeado en n-1)
                    for b in range(n - j):   # goles extra del visitante
                        M120[i + a, j + b] += p * pet_h[a] * pet_a[b]
    return M120 / M120.sum()


def build(model, fixtures, rule, exact, outcome, extra_time=False):
    rows = []
    for m in fixtures.sort_values("date").itertuples():
        M = model.score_matrix(m.home_team, m.away_team, m.neutral)
        if extra_time:
            lh, la = model._lambdas(m.home_team, m.away_team, m.neutral)
            M = extra_time_matrix(M, lh, la)
        ph, pdr, pa = outcome_probs(M)
        if rule == "close":
            bi, bj, ev, p_ex, p_close, p_res = best_bet_ci(M)
        else:  # 'exact' (incluye aditivo)
            bi, bj, ev = best_bet(M, pts_exact=exact, pts_outcome=outcome)
            p_ex = float(M[bi, bj])
        rows.append({
            "date": m.date.date(), "home_team": m.home_team, "away_team": m.away_team,
            "host_home": "sí" if m.home_advantage else "",
            "p_home_%": round(ph * 100, 1), "p_draw_%": round(pdr * 100, 1),
            "p_away_%": round(pa * 100, 1),
            "pick": f"{bi}-{bj}", "pick_1x2": label(bi, bj),
            "ev_pts": round(ev, 2), "p_exact_%": round(p_ex * 100, 1),
        })
    return pd.DataFrame(rows)


def print_table(df, title, max_pts, exact_thresh=14):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "home_team"]).reset_index(drop=True)
    W = 10
    sep = "-" * 56
    print(f"\n  {title}")
    print(sep)
    print(f"  {'#':>2} {'Fecha':<6} {'LOCAL':<{W}} {'VISITA':<{W}} {'JUGAR':>5} {'%':>3} {'X'}")
    print(sep)
    cur = None
    for i, r in df.iterrows():
        d = r["date"].strftime("%d-%b")
        dlabel = d if d != cur else ""
        if d != cur and cur is not None:
            print()
        cur = d
        host = "*" if r["host_home"] == "sí" else ""
        home = (short(r["home_team"]) + host)[:W]
        away = short(r["away_team"])[:W]
        pmax = max(r["p_home_%"], r["p_draw_%"], r["p_away_%"])
        star = "x" if r["p_exact_%"] >= exact_thresh else " "
        print(f"  {i+1:>2} {dlabel:<6} {home:<{W}} {away:<{W}} {r['pick']:>5} "
              f"{pmax:>3.0f} {r['pick_1x2']}{star}")
    print(sep)
    print(f"  {len(df)} partidos | 1:{(df.pick_1x2=='1').sum()}  "
          f"X:{(df.pick_1x2=='X').sum()}  2:{(df.pick_1x2=='2').sum()} | "
          f"EV total ~{df.ev_pts.sum():.0f} pts (máx {max_pts}/part)")
    print(f"  * = anfitrion local   x = buena chance de exacto   % = prob del pick")
    print(sep + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["exact", "close"], default="exact")
    ap.add_argument("--exact", type=float, default=5.0, help="pts al pegar el marcador exacto (total)")
    ap.add_argument("--outcome", type=float, default=2.0, help="pts al acertar solo ganador/empate")
    ap.add_argument("--extra-time", action="store_true",
                    help="suma alargue (90+30) a los empates: usar en fase eliminatoria")
    ap.add_argument("--out", default="predictions_prode.csv")
    args = ap.parse_args()

    history, fixtures = load_data()
    print("Reentrenando con todo el histórico hasta hoy (incluye fase de grupos 2026)...")
    model = DixonColes().fit(history, cutoff="2026-07-18")

    df = build(model, fixtures, args.rule, args.exact, args.outcome,
               extra_time=args.extra_time)
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    out = PREDICTIONS_DIR / args.out
    df.to_csv(out, index=False)
    print(f"Guardado: {out.name} ({len(df)} partidos)")

    if args.rule == "close":
        title, maxp = "PRODE - close points (3/1.5/1/0)", 3
    else:
        title = f"PRODE - exacto {args.exact:g} pts / acertar 1X2 {args.outcome:g} pts"
        maxp = args.exact
    if args.extra_time:
        title += " | 90'+alargue"
    print_table(df, title, maxp)
