"""Chequeo de conectividad inter-confederación entre los 48 clasificados.
Responde: ¿se cruzan lo suficiente las confederaciones como para calibrar
los att/def en una escala común?
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / "data"

QUALIFIED = {
    "UEFA": ["England", "France", "Croatia", "Norway", "Portugal", "Germany",
             "Netherlands", "Switzerland", "Scotland", "Spain", "Austria",
             "Belgium", "Bosnia and Herzegovina", "Sweden", "Turkey", "Czech Republic"],
    "CONMEBOL": ["Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador", "Paraguay"],
    "CONCACAF": ["United States", "Canada", "Mexico", "Panama", "Curaçao", "Haiti"],
    "CAF": ["Morocco", "Tunisia", "Egypt", "Algeria", "Ghana", "Cape Verde",
            "Senegal", "South Africa", "Ivory Coast"],
    "AFC": ["Japan", "Iran", "South Korea", "Australia", "Uzbekistan", "Jordan",
            "Saudi Arabia", "Qatar"],
    "OFC": ["New Zealand"],
    "Playoff": ["DR Congo", "Iraq"],  # DR Congo=CAF, Iraq=AFC realmente
}
# confederación real (corrijo los de playoff)
CONF = {}
for c, teams in QUALIFIED.items():
    for t in teams:
        CONF[t] = c
CONF["DR Congo"] = "CAF"
CONF["Iraq"] = "AFC"

r = pd.read_csv(DATA / "results.csv", parse_dates=["date"])
r = r.dropna(subset=["home_score"])  # saco fixtures futuros

for since in ["2014-01-01", "2018-01-01"]:
    sub = r[(r["date"] >= since) &
            r["home_team"].isin(CONF) & r["away_team"].isin(CONF)].copy()
    sub["ch"] = sub["home_team"].map(CONF)
    sub["ca"] = sub["away_team"].map(CONF)
    sub["pair"] = sub.apply(lambda x: tuple(sorted([x.ch, x.ca])), axis=1)
    intra = sub[sub.ch == sub.ca]
    inter = sub[sub.ch != sub.ca]
    print("\n" + "=" * 70)
    print(f"Partidos ENTRE los 48, desde {since}: {len(sub)}")
    print(f"  intra-confederación: {len(intra)}  |  inter-confederación: {len(inter)}")
    confs = ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"]
    mat = pd.DataFrame(0, index=confs, columns=confs)
    for (a, b), n in sub["pair"].value_counts().items():
        mat.loc[a, b] += n
        if a != b:
            mat.loc[b, a] += n
    print("\nMatriz de cruces (cuántos partidos entre confederaciones):")
    print(mat.to_string())

# Puentes más amplios: cada uno de los 48 vs CUALQUIER rival, contando solo
# si el rival NO es de su confederación. Para eso necesito clasificar rivales
# no clasificados -> aprox: marco "otro" a los desconocidos.
print("\n" + "=" * 70)
print("Partidos inter-confederación de cada equipo de los 48 (vs los otros 47), desde 2018:")
sub = r[(r["date"] >= "2018-01-01") &
        r["home_team"].isin(CONF) & r["away_team"].isin(CONF)].copy()
sub["ch"] = sub["home_team"].map(CONF); sub["ca"] = sub["away_team"].map(CONF)
inter = sub[sub.ch != sub.ca]
long = pd.concat([inter[["home_team"]].rename(columns={"home_team": "team"}),
                  inter[["away_team"]].rename(columns={"away_team": "team"})])
cnt = long.value_counts().rename("inter_vs_48").to_frame()
cnt["conf"] = cnt.index.get_level_values(0).map(CONF)
print(cnt.reset_index().to_string(index=False))
