"""Etapa 4b/4c — Diagnóstico de residuos OUT-OF-SAMPLE del modelo tuneado.

Decide si vale agregar (4b) término de confederación y (4c) ajuste por brecha,
mirando si el modelo tiene sesgos sistemáticos en el rolling backtest:
  - 4b: ¿predice mal los goles según el par de confederaciones?
  - 4c: ¿subestima las goleadas? ¿están calibradas las probabilidades?
"""
import numpy as np
import pandas as pd

from features import load_data
from model import DixonColes, outcome_probs

CONFS = ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"]


def collect(history, years=range(2018, 2026)):
    rows = []
    for y in years:
        cutoff = pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=1)
        model = DixonColes().fit(history, cutoff=cutoff, verbose=False)
        known = model.teams
        test = history[(history["date"].dt.year == y) & (~history["is_friendly"])
                       & history["is_fifa"]
                       & history["home_team"].isin(known)
                       & history["away_team"].isin(known)]
        for r in test.itertuples():
            lh, la = model._lambdas(r.home_team, r.away_team, r.neutral)
            M = model.score_matrix(r.home_team, r.away_team, r.neutral)
            ph, pd_, pa = outcome_probs(M)
            n = M.shape[0]
            i = np.arange(n)[:, None]; j = np.arange(n)[None, :]
            p_blow = M[np.abs(i - j) >= 3].sum()
            rows.append((r.home_conf, r.away_conf, r.home_score, r.away_score,
                         lh, la, ph, pd_, pa, p_blow))
    return pd.DataFrame(rows, columns=[
        "hc", "ac", "hs", "as_", "lh", "la", "ph", "pd", "pa", "p_blow"])


def confed_diag(df):
    """4b: ratio goles reales / predichos por (confed atacante -> confed defensora)."""
    att = pd.concat([
        df[["hc", "ac", "hs", "lh"]].rename(columns={"hc": "att", "ac": "deff", "hs": "real", "lh": "pred"}),
        df[["ac", "hc", "as_", "la"]].rename(columns={"ac": "att", "hc": "deff", "as_": "real", "la": "pred"}),
    ])
    g = att.groupby(["att", "deff"]).agg(real=("real", "sum"), pred=("pred", "sum"),
                                         n=("real", "size"))
    g["ratio"] = (g["real"] / g["pred"]).round(2)
    mat = g["ratio"].unstack().reindex(index=CONFS, columns=CONFS)
    print("4b — RATIO goles reales/predichos  (filas=conf atacante, cols=conf defensora)")
    print("     ~1.00 = bien calibrado; >1 subestima ataque de esa confed, <1 sobreestima")
    print(mat.to_string())
    off = (g["ratio"] - 1).abs()
    big = g[(g["n"] >= 40) & (off > 0.15)].sort_values("ratio")
    print(f"\nPares con sesgo >15% y n>=40: {len(big)}")
    if len(big):
        print(big.round(2).to_string())


def blowout_diag(df):
    """4c: calibración de goleadas (margen >=3)."""
    df = df.copy()
    df["blow"] = (df["hs"] - df["as_"]).abs() >= 3
    df["bin"] = pd.cut(df["p_blow"], [0, 0.1, 0.2, 0.3, 0.5, 1.0])
    g = df.groupby("bin", observed=True).agg(n=("blow", "size"),
                                             pred=("p_blow", "mean"),
                                             real=("blow", "mean"))
    print("\n4c — CALIBRACIÓN DE GOLEADAS (margen >=3)")
    print("     pred = prob media que da el modelo; real = frecuencia observada")
    print(g.assign(pred=g["pred"].round(3), real=g["real"].round(3)).to_string())
    print(f"\n  Global: pred {df['p_blow'].mean():.3f} vs real {df['blow'].mean():.3f}")


def outcome_calib(df):
    """Calibración de P(gana local)."""
    df = df.copy()
    df["home_win"] = df["hs"] > df["as_"]
    df["bin"] = pd.cut(df["ph"], np.linspace(0, 1, 11))
    g = df.groupby("bin", observed=True).agg(n=("home_win", "size"),
                                             pred=("ph", "mean"),
                                             real=("home_win", "mean"))
    print("\nCALIBRACIÓN P(gana local)  (pred vs real por bin)")
    print(g.assign(pred=g["pred"].round(3), real=g["real"].round(3)).to_string())


if __name__ == "__main__":
    history, _ = load_data()
    df = collect(history)
    print(f"Partidos out-of-sample analizados: {len(df)}\n")
    confed_diag(df)
    blowout_diag(df)
    outcome_calib(df)
