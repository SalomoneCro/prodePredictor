"""Etapa 2 — Validación por backtesting temporal.

Dos backtests, siempre entrenando SOLO con datos anteriores a cada partido (sin fuga):
  1. WC backtest: fase de grupos de los Mundiales 2014/2018/2022 (escenario gemelo:
     cancha neutral, cruces inter-confederación). Es el headline.
  2. Rolling anual: partidos competitivos (no amistosos) 2018-2025, muestra grande
     para robustez y para tunear hiperparámetros (Etapa 4).

Métricas: puntos de prode (con jugada óptima por EV), RPS y log-loss del 1X2,
accuracy 1X2 y % de marcador exacto. Se compara el modelo contra baselines.
"""
import numpy as np
import pandas as pd

from features import load_data
from model import DixonColes, outcome_probs, best_bet

EPS = 1e-12


# ---------- métricas ----------
def prode_points(pred, actual, pts_exact=6, pts_outcome=3):
    """Puntos según las reglas del prode (default: 6 exacto / 3 acierto 1X2 / 0).

    Parametrizado para reutilizar el modelo en prodes con otras reglas."""
    pi, pj = pred
    ai, aj = actual
    if pi == ai and pj == aj:
        return pts_exact
    if np.sign(pi - pj) == np.sign(ai - aj):
        return pts_outcome
    return 0


def actual_outcome_idx(ai, aj):
    return 0 if ai > aj else (1 if ai == aj else 2)   # home / draw / away


def rps(probs, outcome_idx):
    """Ranked Probability Score para 3 categorías ordenadas [home, draw, away]."""
    o = np.zeros(3); o[outcome_idx] = 1
    cp = np.cumsum(probs); co = np.cumsum(o)
    return float(np.sum((cp - co) ** 2) / 2.0)


def logloss(probs, outcome_idx):
    return float(-np.log(max(probs[outcome_idx], EPS)))


# ---------- baselines ----------
def frequency_matrices(train, max_goals=10):
    """Matrices de marcadores empíricas (neutral y no-neutral) del set de entrenamiento.

    Baseline 'sin info por equipo': misma matriz para todos los partidos según neutral.
    """
    mats = {}
    for neutral in (True, False):
        sub = train[train["neutral"] == neutral]
        M = np.zeros((max_goals + 1, max_goals + 1))
        for h, a in zip(sub["home_score"].clip(upper=max_goals),
                        sub["away_score"].clip(upper=max_goals)):
            M[int(h), int(a)] += 1
        mats[neutral] = M / M.sum() if M.sum() else np.full_like(M, 1 / M.size)
    return mats


# ---------- evaluación de un conjunto de partidos ----------
def evaluate(model, test, train_for_baseline):
    freq = frequency_matrices(train_for_baseline)
    rows = []
    for _, m in test.iterrows():
        actual = (int(m.home_score), int(m.away_score))
        oidx = actual_outcome_idx(*actual)

        # --- modelo ---
        M = model.score_matrix(m.home_team, m.away_team, m.neutral)
        probs = np.array(outcome_probs(M))
        bi, bj, _ = best_bet(M)

        # --- baselines ---
        Mf = freq[bool(m.neutral)]
        pf = np.array(outcome_probs(Mf))
        fi, fj, _ = best_bet(Mf)

        rows.append({
            "model_pts": prode_points((bi, bj), actual),
            "model_rps": rps(probs, oidx),
            "model_ll": logloss(probs, oidx),
            "model_1x2_ok": int(np.argmax(probs) == oidx),
            "model_exact_ok": int((bi, bj) == actual),
            "freq_pts": prode_points((fi, fj), actual),
            "freq_rps": rps(pf, oidx),
            "freq_ll": logloss(pf, oidx),
            "naive10_pts": prode_points((1, 0), actual),   # siempre 1-0 al local
            "naive11_pts": prode_points((1, 1), actual),   # siempre 1-1
        })
    return pd.DataFrame(rows)


# ---------- backtests ----------
def wc_backtest(history):
    wc = history[history["tournament"] == "FIFA World Cup"].copy()
    results = []
    for year in (2014, 2018, 2022):
        ed = wc[wc["date"].dt.year == year].sort_values("date")
        group = ed.head(48)                       # fase de grupos (32 equipos -> 48 PJ)
        start = group["date"].min()
        cutoff = start - pd.Timedelta(days=1)
        train = history[history["date"] <= cutoff]
        model = DixonColes().fit(history, cutoff=cutoff, verbose=False)
        ev = evaluate(model, group, train)
        ev["edition"] = year
        results.append(ev)
        print(f"  Mundial {year}: {len(group)} partidos, entrenado hasta {cutoff.date()}")
    return pd.concat(results, ignore_index=True)


def rolling_backtest(history, years=range(2018, 2026)):
    res = []
    for y in years:
        cutoff = pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=1)
        test = history[(history["date"].dt.year == y) & (~history["is_friendly"])
                       & history["is_fifa"]]
        # solo equipos que el modelo conocerá (con muestra) -> se filtra dentro de score
        if len(test) == 0:
            continue
        model = DixonColes().fit(history, cutoff=cutoff, verbose=False)
        known = model.teams
        test = test[test["home_team"].isin(known) & test["away_team"].isin(known)]
        train = history[history["date"] <= cutoff]
        ev = evaluate(model, test, train)
        ev["year"] = y
        res.append(ev)
        print(f"  {y}: {len(test)} partidos competitivos")
    return pd.concat(res, ignore_index=True)


def summary(df, label):
    n = len(df)
    print(f"\n{'='*64}\n{label}  (n={n})\n{'='*64}")
    print(f"{'':18} {'PTS/part':>9} {'RPS':>8} {'logloss':>8} {'1X2 acc':>8} {'exacto':>8}")
    print(f"{'MODELO':18} {df.model_pts.mean():9.3f} {df.model_rps.mean():8.4f} "
          f"{df.model_ll.mean():8.4f} {df.model_1x2_ok.mean():8.1%} {df.model_exact_ok.mean():8.1%}")
    print(f"{'baseline frecuen.':18} {df.freq_pts.mean():9.3f} {df.freq_rps.mean():8.4f} "
          f"{df.freq_ll.mean():8.4f} {'-':>8} {'-':>8}")
    print(f"{'baseline 1-0':18} {df.naive10_pts.mean():9.3f}")
    print(f"{'baseline 1-1':18} {df.naive11_pts.mean():9.3f}")
    print(f"\n  Total puntos modelo: {df.model_pts.sum()}  vs  "
          f"frecuencia: {df.freq_pts.sum()}  (mejora "
          f"{df.model_pts.sum()-df.freq_pts.sum():+d} pts)")


if __name__ == "__main__":
    history, _ = load_data()

    print("WC backtest (fase de grupos 2014/2018/2022):")
    wc = wc_backtest(history)
    summary(wc, "WORLD CUP — FASE DE GRUPOS (escenario gemelo)")
    print("\nPor edición (PTS/partido modelo vs frecuencia):")
    print(wc.groupby("edition")[["model_pts", "freq_pts"]].mean().round(3).to_string())

    print("\n\nRolling backtest (partidos competitivos por año):")
    roll = rolling_backtest(history)
    summary(roll, "ROLLING 2018-2025 — COMPETITIVOS (muestra grande)")
