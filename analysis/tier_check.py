"""¿Hay efectos no-lineales según la BRECHA de nivel entre equipos?
Proxy de fuerza simple (sin modelo): promedio de (GF - GA) por partido, 2014+.
Después miramos cómo cambian empates / goleadas / goles del favorito según la brecha.
Esto NO es el modelo, es un chequeo exploratorio de la idea de 'clusters/tiers'.
"""
from pathlib import Path
import pandas as pd
import numpy as np

from prode import DATA_DIR as DATA
r = pd.read_csv(DATA / "results.csv", parse_dates=["date"]).dropna(subset=["home_score"])
r = r[r["date"] >= "2014-01-01"].copy()

# fuerza proxy: media de diferencia de gol por partido (todos los rivales)
gf = pd.concat([
    r.assign(team=r.home_team, dgf=r.home_score - r.away_score)[["team", "dgf"]],
    r.assign(team=r.away_team, dgf=r.away_score - r.home_score)[["team", "dgf"]],
])
strength = gf.groupby("team")["dgf"].mean()
# nos quedamos con equipos con muestra razonable
counts = gf.groupby("team").size()
strength = strength[counts >= 30]

r = r[r.home_team.isin(strength.index) & r.away_team.isin(strength.index)].copy()
r["s_home"] = r.home_team.map(strength)
r["s_away"] = r.away_team.map(strength)
r["gap"] = r["s_home"] - r["s_away"]          # >0 => local más fuerte
r["abs_gap"] = r["gap"].abs()
r["fav_is_home"] = r["gap"] > 0
r["goal_diff_fav"] = np.where(r.fav_is_home, r.home_score - r.away_score,
                              r.away_score - r.home_score)
r["draw"] = r.home_score == r.away_score
r["fav_win"] = r["goal_diff_fav"] > 0
r["blowout3"] = r["goal_diff_fav"] >= 3
r["total_goals"] = r.home_score + r.away_score

# bins por brecha absoluta de nivel
bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 10]
labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3+"]
r["bin"] = pd.cut(r["abs_gap"], bins=bins, labels=labels)

agg = r.groupby("bin", observed=True).agg(
    n=("draw", "size"),
    pct_empate=("draw", "mean"),
    pct_gana_favorito=("fav_win", "mean"),
    pct_goleada_3plus=("blowout3", "mean"),
    goldif_favorito=("goal_diff_fav", "mean"),
    goles_totales=("total_goals", "mean"),
)
for c in ["pct_empate", "pct_gana_favorito", "pct_goleada_3plus"]:
    agg[c] = (agg[c] * 100).round(1)
agg[["goldif_favorito", "goles_totales"]] = agg[["goldif_favorito", "goles_totales"]].round(2)

print("Efecto de la BRECHA de nivel (proxy) sobre los resultados (2014+):")
print(agg.to_string())

print("\nLectura clave:")
print("- Si pct_empate baja suave y goldif sube suave => att/def continuo ya lo maneja.")
print("- Si hay saltos/kinks o el goldif se 'aplana' en brechas grandes => hay no-linealidad")
print("  que un término de interacción por brecha podría capturar.")

# Test extra: ¿el goldif crece lineal con la brecha o satura?
print("\nGol-dif del favorito vs brecha (para ver saturación):")
print(r.groupby("bin", observed=True)["goal_diff_fav"].mean().round(2).to_string())
