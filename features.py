"""Etapa 0 — Pipeline de datos y features.

Separa el histórico (partidos jugados) de los 72 fixtures del Mundial 2026
(score nulo) y agrega las features acordadas. NO modela nada.

Uso:
    from features import load_data
    history, fixtures = load_data()
"""
from pathlib import Path
import numpy as np
import pandas as pd

from confederations import confederation

DATA = Path(__file__).parent / "data"
CUTOFF = pd.Timestamp("2026-06-04")          # hoy: separa jugados de fixtures
HOSTS = {"Mexico", "United States", "Canada"}  # anfitriones (juegan de local)

# Pesos por competitividad del torneo (tuneables en Etapa 4).
W_FRIENDLY = 0.5
W_QUALIFIER = 0.9
W_MAJOR = 1.0
W_OTHER = 0.7

# Torneos "mayores" (fases finales de selecciones) -> peso máximo.
MAJOR_TOURNAMENTS = {
    "FIFA World Cup", "UEFA Euro", "Copa América", "African Cup of Nations",
    "AFC Asian Cup", "Gold Cup", "CONCACAF Championship", "UEFA Nations League",
    "CONCACAF Nations League", "Confederations Cup", "OFC Nations Cup",
}


def tournament_weight(t: str) -> float:
    if t == "Friendly":
        return W_FRIENDLY
    if t in MAJOR_TOURNAMENTS:
        return W_MAJOR
    if "qualification" in t.lower():
        return W_QUALIFIER
    return W_OTHER


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_conf"] = df["home_team"].map(confederation)
    df["away_conf"] = df["away_team"].map(confederation)
    df["same_conf"] = df["home_conf"] == df["away_conf"]
    df["is_friendly"] = df["tournament"] == "Friendly"
    df["comp_weight"] = df["tournament"].map(tournament_weight)
    # Ventaja de localía = hay local real (no neutral). En los fixtures 2026 esto
    # ya queda True solo para los 9 partidos de los anfitriones.
    df["home_advantage"] = (~df["neutral"]).astype(int)
    # ¿Ambos lados son selecciones FIFA? (para poder filtrar ruido no-FIFA luego)
    df["is_fifa"] = (df["home_conf"] != "Other") & (df["away_conf"] != "Other")
    return df


def load_data():
    """Devuelve (history, fixtures) con features agregadas.

    history : partidos jugados (home_score/away_score no nulos), con result y goles.
    fixtures: los 72 partidos de grupos del Mundial 2026 a predecir (sin score).
    """
    r = pd.read_csv(DATA / "results.csv", parse_dates=["date"])

    played = r[r["home_score"].notna()].copy()
    fixtures = r[r["home_score"].isna()].copy()

    played = add_features(played)
    fixtures = add_features(fixtures)

    # Variables objetivo / derivadas solo para el histórico
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    played["total_goals"] = played["home_score"] + played["away_score"]
    played["result"] = np.select(
        [played["home_score"] > played["away_score"],
         played["home_score"] < played["away_score"]],
        ["H", "A"], default="D",
    )
    return played, fixtures


def time_weight(dates: pd.Series, cutoff: pd.Timestamp, half_life_days: float) -> np.ndarray:
    """Peso por recencia con vida media dada (para el decay del modelo, Etapa 1)."""
    days = (cutoff - dates).dt.days.clip(lower=0)
    return 0.5 ** (days / half_life_days)


def sanity_checks(history: pd.DataFrame, fixtures: pd.DataFrame) -> None:
    print("=" * 70)
    print("ETAPA 0 — CHEQUEOS DE SANIDAD")
    print("=" * 70)

    # 1. Partición
    print(f"\nHistórico (jugados): {len(history):,} partidos "
          f"({history['date'].min().date()} -> {history['date'].max().date()})")
    print(f"Fixtures a predecir: {len(fixtures)} (esperado 72)")
    assert len(fixtures) == 72, "Se esperaban 72 fixtures de fase de grupos"
    assert history["home_score"].notna().all(), "Hay scores nulos en el histórico"
    assert (fixtures["date"] > CUTOFF).all(), "Algún fixture no es futuro"

    # 2. Los 48 presentes en los fixtures
    teams_fx = pd.unique(fixtures[["home_team", "away_team"]].values.ravel())
    print(f"\nEquipos distintos en fixtures: {len(teams_fx)} (esperado 48)")
    assert len(teams_fx) == 48, "Los fixtures no tienen 48 equipos"

    # 3. Cobertura de confederaciones (que ningún clasificado quede en 'Other')
    fx_other = [t for t in teams_fx if confederation(t) == "Other"]
    print(f"Clasificados sin confederación (Other): {fx_other if fx_other else 'ninguno ✓'}")
    assert not fx_other, "Hay clasificados mapeados a 'Other'"

    # 4. Localía en fixtures: solo anfitriones
    home_adv_fx = fixtures[fixtures["home_advantage"] == 1]["home_team"].unique()
    print(f"Equipos con localía en fixtures: {sorted(home_adv_fx)} "
          f"({len(fixtures[fixtures['home_advantage']==1])} partidos)")
    assert set(home_adv_fx) <= HOSTS, "Localía asignada a un no-anfitrión"

    # 5. Cobertura confederación en el histórico
    in_fifa = history["is_fifa"].mean() * 100
    print(f"\n% de partidos históricos entre dos selecciones FIFA: {in_fifa:.1f}%")
    other_teams = sorted({t for t in pd.unique(history[["home_team", "away_team"]].values.ravel())
                          if confederation(t) == "Other"})
    print(f"Equipos en 'Other' (no-FIFA/regionales), {len(other_teams)}: "
          f"{other_teams[:12]}{' ...' if len(other_teams) > 12 else ''}")

    # 6. Distribución de pesos por torneo
    print("\nPeso por competitividad (histórico):")
    print(history.groupby("comp_weight").size().to_string())

    # 7. Nulos en columnas clave
    key_cols = ["home_team", "away_team", "home_score", "away_score", "date",
                "neutral", "home_conf", "away_conf", "comp_weight"]
    nulls = history[key_cols].isna().sum()
    print(f"\nNulos en columnas clave del histórico: "
          f"{'ninguno ✓' if nulls.sum() == 0 else nulls[nulls>0].to_dict()}")
    assert nulls.sum() == 0

    print("\n>>> ETAPA 0 OK — datos listos para modelar <<<")


if __name__ == "__main__":
    history, fixtures = load_data()
    sanity_checks(history, fixtures)
    print("\nMuestra de fixtures con features:")
    cols = ["date", "home_team", "away_team", "home_conf", "away_conf",
            "same_conf", "neutral", "home_advantage"]
    print(fixtures[cols].head(10).to_string(index=False))
