"""Etapa 1 — Modelo base Dixon-Coles (Poisson de goles ataque/defensa).

log λ_local = μ + att[local] + def[visit] + γ·ventaja_local
log λ_visit = μ + att[visit] + def[local]

- att[t]  : fuerza ofensiva (más alto => mete más)
- def[t]  : debilidad defensiva (más alto => recibe más)
- γ       : ventaja de localía (solo aplica si NO es cancha neutral)
- ρ       : corrección Dixon-Coles para marcadores bajos (0-0,1-0,0-1,1-1)

Ajuste: regresión de Poisson ponderada (decay temporal × peso por torneo) para
att/def/μ/γ, y ρ por verosimilitud perfilada. El resultado de un partido sale de
la matriz de marcadores analítica (sin Monte Carlo).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
import statsmodels.api as sm

from prode.features import load_data, time_weight

# Hiperparámetros por defecto (tuneados en Etapa 4 vía tune.py)
HALF_LIFE_DAYS = 1095.0     # vida media del decay (~3 años)
TRAIN_WINDOW_YEARS = 12     # ventana de entrenamiento (con decay lo viejo pesa ~0)
MIN_MATCHES = 8             # mínimo de partidos en ventana para incluir a un equipo
MAX_GOALS = 10              # truncado de la matriz de marcadores


@dataclass
class DixonColes:
    intercept: float = 0.0
    gamma: float = 0.0          # ventaja de localía
    rho: float = 0.0            # corrección DC
    att: dict = field(default_factory=dict)
    deff: dict = field(default_factory=dict)
    teams: set = field(default_factory=set)

    # ---- ajuste ----
    def fit(self, history, cutoff, half_life_days=HALF_LIFE_DAYS,
            window_years=TRAIN_WINDOW_YEARS, min_matches=MIN_MATCHES,
            friendly_weight=None, verbose=True):
        cutoff = pd.Timestamp(cutoff)
        start = cutoff - pd.DateOffset(years=window_years)
        df = history[(history["date"] <= cutoff) & (history["date"] >= start)
                     & history["is_fifa"]].copy()

        # equipos con muestra suficiente en la ventana
        counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        eligible = set(counts[counts >= min_matches].index)
        df = df[df["home_team"].isin(eligible) & df["away_team"].isin(eligible)].copy()
        self.teams = eligible

        # peso = decay temporal × competitividad (con override opcional de amistosos)
        cw = df["comp_weight"]
        if friendly_weight is not None:
            cw = cw.where(~df["is_friendly"], friendly_weight)
        df["w"] = time_weight(df["date"], cutoff, half_life_days) * cw

        # formato largo: cada partido -> 2 observaciones (ataque/defensa)
        home_obs = pd.DataFrame({
            "goals": df["home_score"], "att_team": df["home_team"],
            "def_team": df["away_team"], "home": df["home_advantage"], "w": df["w"]})
        away_obs = pd.DataFrame({
            "goals": df["away_score"], "att_team": df["away_team"],
            "def_team": df["home_team"], "home": 0, "w": df["w"]})
        long = pd.concat([home_obs, away_obs], ignore_index=True)

        # diseño: dummies de ataque y defensa (drop_first => equipo de referencia)
        A = pd.get_dummies(long["att_team"], prefix="att", drop_first=True, dtype=float)
        D = pd.get_dummies(long["def_team"], prefix="def", drop_first=True, dtype=float)
        X = pd.concat([long[["home"]].astype(float), A, D], axis=1)
        X = sm.add_constant(X)

        if verbose:
            print(f"Ajustando: {len(df):,} partidos ({df['date'].min().date()} -> "
                  f"{df['date'].max().date()}), {len(eligible)} equipos, "
                  f"{X.shape[1]} parámetros...")

        glm = sm.GLM(long["goals"], X, family=sm.families.Poisson(),
                     freq_weights=long["w"].values)
        res = glm.fit()
        p = res.params

        self.intercept = float(p["const"])
        self.gamma = float(p["home"])
        self.att = {c[4:]: float(v) for c, v in p.items() if c.startswith("att_")}
        self.deff = {c[4:]: float(v) for c, v in p.items() if c.startswith("def_")}
        # equipo de referencia (dropeado) => coef 0
        for t in eligible:
            self.att.setdefault(t, 0.0)
            self.deff.setdefault(t, 0.0)

        self._fit_rho(df)
        if verbose:
            print(f"  μ(intercept)={self.intercept:.3f}  γ(localía)={self.gamma:.3f}  "
                  f"ρ(DC)={self.rho:.4f}")
        return self

    def _fit_rho(self, df):
        """ρ por verosimilitud perfilada sobre marcadores bajos (usa df['w'])."""
        lh = np.exp(self.intercept + df["home_team"].map(self.att).fillna(0)
                    + df["away_team"].map(self.deff).fillna(0)
                    + self.gamma * df["home_advantage"]).values
        la = np.exp(self.intercept + df["away_team"].map(self.att).fillna(0)
                    + df["home_team"].map(self.deff).fillna(0)).values
        x = df["home_score"].values
        y = df["away_score"].values
        w = df["w"].values

        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)

        def negll(rho):
            tau = np.ones_like(lh)
            tau[m00] = 1 - lh[m00] * la[m00] * rho
            tau[m01] = 1 + lh[m01] * rho
            tau[m10] = 1 + la[m10] * rho
            tau[m11] = 1 - rho
            if np.any(tau <= 0):
                return 1e10
            return -np.sum(w * np.log(tau))

        r = minimize_scalar(negll, bounds=(-0.2, 0.2), method="bounded")
        self.rho = float(r.x)

    # ---- predicción ----
    def _lambdas(self, home, away, neutral):
        ha = 0 if neutral else 1
        lh = np.exp(self.intercept + self.att.get(home, 0.0)
                    + self.deff.get(away, 0.0) + self.gamma * ha)
        la = np.exp(self.intercept + self.att.get(away, 0.0)
                    + self.deff.get(home, 0.0))
        return lh, la

    def score_matrix(self, home, away, neutral, max_goals=MAX_GOALS):
        lh, la = self._lambdas(home, away, neutral)
        ph = poisson.pmf(np.arange(max_goals + 1), lh)
        pa = poisson.pmf(np.arange(max_goals + 1), la)
        M = np.outer(ph, pa)            # filas: goles local, cols: goles visit
        # corrección Dixon-Coles en las 4 celdas bajas
        M[0, 0] *= 1 - lh * la * self.rho
        M[0, 1] *= 1 + lh * self.rho
        M[1, 0] *= 1 + la * self.rho
        M[1, 1] *= 1 - self.rho
        return M / M.sum()              # renormalizo (DC rompe la suma exacta)


# ---- utilidades sobre la matriz ----
def outcome_probs(M):
    """(P_local, P_empate, P_visit) a partir de la matriz."""
    p_home = np.tril(M, -1).sum()
    p_draw = np.trace(M)
    p_away = np.triu(M, 1).sum()
    return p_home, p_draw, p_away


def most_likely_score(M):
    i, j = np.unravel_index(np.argmax(M), M.shape)
    return int(i), int(j), float(M[i, j])


def best_bet(M, pts_exact=6.0, pts_outcome=3.0):
    """Jugada que maximiza los puntos esperados, AGNÓSTICA a las reglas del prode.

    Asume que acertar el marcador exacto otorga `pts_exact` (e incluye acertar el 1X2)
    y acertar solo el 1X2 otorga `pts_outcome`. Para el prode actual: 6 / 3.

        EV(s) = pts_outcome·P(1X2 de s) + (pts_exact − pts_outcome)·P(marcador exacto s)

    Cambiando estos dos números se adapta a cualquier prode futuro. Devuelve (i, j, EV).
    """
    n = M.shape[0]
    idx = np.arange(n)
    di = idx[:, None]
    dj = idx[None, :]
    p_home, p_draw, p_away = outcome_probs(M)
    outcome_p = np.where(di > dj, p_home, np.where(di == dj, p_draw, p_away))
    EV = pts_outcome * outcome_p + (pts_exact - pts_outcome) * M
    i, j = np.unravel_index(np.argmax(EV), EV.shape)
    return int(i), int(j), float(EV[i, j])


def closeness_index(a, b, i, j):
    """CI = |dif_goles_pick − dif_goles_real| + |goles_tot_pick − goles_tot_real| / 2.

    Pick (a-b) vs marcador real (i-j). Se considera "close" si CI <= 1.5 (y además
    se acertó el ganador/empate). Regla del prode 2026.
    """
    return abs((a - b) - (i - j)) + abs((a + b) - (i + j)) / 2.0


def ci_points(a, b, i, j, p_exact=3.0, p_close=1.5, p_result=1.0, ci_thresh=1.5):
    """Puntos que da el pick (a-b) si el marcador real es (i-j), con las reglas 2026.

    3 exacto / 1.5 acierto-de-ganador-y-close / 1 acierto-de-ganador / 0 errar.
    El exacto NO se suma al resto (lo reemplaza).
    """
    if a == i and b == j:
        return p_exact
    same_outcome = np.sign(a - b) == np.sign(i - j)
    if not same_outcome:
        return 0.0
    if closeness_index(a, b, i, j) <= ci_thresh:
        return p_close
    return p_result


def best_bet_ci(M, p_exact=3.0, p_close=1.5, p_result=1.0, ci_thresh=1.5):
    """Jugada que maximiza los puntos esperados bajo las reglas del prode 2026.

        EV(pick) = Σ_{marcadores reales r} P(r) · puntos(pick, r)

    A diferencia de `best_bet`, hay un escalón intermedio ("close") que premia quedar
    cerca, así que no se reduce a dos constantes: se suma sobre toda la matriz de
    marcadores. Devuelve (i, j, EV, p_exact_hit, p_close_or_better, p_result_or_better).
    """
    n = M.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    gd_real, tot_real = ii - jj, ii + jj
    best = (-1.0, 0, 0)
    detail = (0.0, 0.0, 0.0)
    for a in range(n):
        for b in range(n):
            ci = np.abs((a - b) - gd_real) + np.abs((a + b) - tot_real) / 2.0
            same = np.sign(a - b) == np.sign(gd_real)
            is_exact = (ii == a) & (jj == b)
            is_close = same & (ci <= ci_thresh) & ~is_exact
            is_res = same & (ci > ci_thresh) & ~is_exact
            pts = np.where(is_exact, p_exact,
                           np.where(is_close, p_close,
                                    np.where(is_res, p_result, 0.0)))
            ev = float((M * pts).sum())
            if ev > best[0]:
                best = (ev, a, b)
                detail = (float(M[a, b]),
                          float((M * (is_exact | is_close)).sum()),
                          float((M * same).sum()))
    ev, a, b = best
    return a, b, ev, detail[0], detail[1], detail[2]


if __name__ == "__main__":
    history, fixtures = load_data()
    model = DixonColes().fit(history, cutoff="2026-06-04")

    # ranking de fuerza (att - def) para sanity check
    strength = {t: model.att[t] - model.deff[t] for t in model.teams}
    s = pd.Series(strength).sort_values(ascending=False)
    print("\nTop 12 más fuertes (att - def):")
    print(s.head(12).round(3).to_string())
    print("\nBottom 6:")
    print(s.tail(6).round(3).to_string())
    print(f"\nVentaja de localía: x{np.exp(model.gamma):.2f} goles esperados.")

    print("\nEjemplos de fixtures 2026:")
    for _, m in fixtures.head(6).iterrows():
        M = model.score_matrix(m.home_team, m.away_team, m.neutral)
        ph, pd_, pa = outcome_probs(M)
        i, j, p = most_likely_score(M)
        lh, la = model._lambdas(m.home_team, m.away_team, m.neutral)
        print(f"  {m.home_team:>14} vs {m.away_team:<16} "
              f"λ {lh:.2f}-{la:.2f} | P(1/X/2) {ph:.0%}/{pd_:.0%}/{pa:.0%} | "
              f"más prob: {i}-{j} ({p:.0%})")
