"""Exploración del dataset de resultados de fútbol internacional (martj42).

Genera estadísticas para el reporte y guarda figuras en ./figures
NO construye ningún modelo.
"""
from pathlib import Path
import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 140)

DATA = Path(__file__).parent / "data"
FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

# --- Los 48 clasificados al Mundial 2026 (confirmados, incl. repechajes mar-2026) ---
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
    "Playoff": ["DR Congo", "Iraq"],
}
ALL48 = [t for teams in QUALIFIED.values() for t in teams]


def banner(txt):
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)


# ============================================================
banner("1. ARCHIVOS Y COLUMNAS")
results = pd.read_csv(DATA / "results.csv", parse_dates=["date"])
goals = pd.read_csv(DATA / "goalscorers.csv", parse_dates=["date"])
shootouts = pd.read_csv(DATA / "shootouts.csv", parse_dates=["date"])
former = pd.read_csv(DATA / "former_names.csv")

print(f"results.csv     : {results.shape[0]:,} filas x {results.shape[1]} cols -> {list(results.columns)}")
print(f"goalscorers.csv : {goals.shape[0]:,} filas x {goals.shape[1]} cols -> {list(goals.columns)}")
print(f"shootouts.csv   : {shootouts.shape[0]:,} filas x {shootouts.shape[1]} cols -> {list(shootouts.columns)}")
print(f"former_names.csv: {former.shape[0]:,} filas x {former.shape[1]} cols -> {list(former.columns)}")
print("\nDtypes results:")
print(results.dtypes)

# ============================================================
banner("2. RANGO DE FECHAS Y TOTALES")
print(f"Primer partido: {results['date'].min().date()}")
print(f"Último partido: {results['date'].max().date()}")
print(f"Total partidos: {len(results):,}")
print(f"Span: {(results['date'].max() - results['date'].min()).days / 365.25:.1f} años")
print("\nPartidos por década:")
dec = results.groupby(results["date"].dt.year // 10 * 10).size()
print(dec.to_string())

# ============================================================
banner("3. TIPOS DE TORNEO")
tt = results["tournament"].value_counts()
print(f"Tipos distintos: {results['tournament'].nunique()}")
print("\nTop 25 por cantidad:")
print(tt.head(25).to_string())
wc = results[results["tournament"] == "FIFA World Cup"]
wcq = results[results["tournament"].str.contains("qualification", case=False, na=False)]
print(f"\nFIFA World Cup (fase final): {len(wc):,} partidos, {wc['date'].dt.year.min()}-{wc['date'].dt.year.max()}")
print(f"Cualquier 'qualification': {len(wcq):,} partidos")
print(f"Friendly: {tt.get('Friendly', 0):,} ({tt.get('Friendly',0)/len(results)*100:.1f}% del total)")

# ============================================================
banner("4. CALIDAD DE DATOS")
print("Nulos por columna (results):")
print(results.isna().sum().to_string())
print(f"\nFilas duplicadas exactas (results): {results.duplicated().sum()}")
dup_keys = results.duplicated(subset=["date", "home_team", "away_team"]).sum()
print(f"Duplicados por (date, home, away): {dup_keys}")
print(f"\nNulos goalscorers:\n{goals.isna().sum().to_string()}")
print(f"\nNulos shootouts:\n{shootouts.isna().sum().to_string()}")

teams = pd.unique(results[["home_team", "away_team"]].values.ravel())
print(f"\nEquipos distintos en results: {len(teams)}")
print("\nEjemplos de equipos históricos / nombres cambiados (former_names.csv):")
print(former.to_string(index=False))

# Score negativos o raros
print(f"\nScores negativos: {((results['home_score'] < 0) | (results['away_score'] < 0)).sum()}")
print(f"Goleadas >=10 de diferencia: {(abs(results['home_score']-results['away_score'])>=10).sum()}")

# ============================================================
banner("5. TASAS W/D/L (perspectiva equipo LOCAL)")
def outcome(r):
    if r.home_score > r.away_score:
        return "home_win"
    if r.home_score < r.away_score:
        return "away_win"
    return "draw"
results["result"] = results.apply(outcome, axis=1)
overall = results["result"].value_counts(normalize=True)
print("Global:")
print((overall * 100).round(1).to_string())
print(f"\nPromedio goles local: {results['home_score'].mean():.2f} | visitante: {results['away_score'].mean():.2f}")
print(f"Promedio goles por partido: {(results['home_score']+results['away_score']).mean():.2f}")

# Por tipo de torneo (los principales)
banner("5b. W/D/L POR TIPO DE TORNEO (top tipos)")
for t in ["FIFA World Cup", "Friendly", "FIFA World Cup qualification", "UEFA Euro", "Copa América"]:
    sub = results[results["tournament"] == t]
    if len(sub) == 0:
        continue
    r = sub["result"].value_counts(normalize=True) * 100
    print(f"{t:32s} (n={len(sub):6,}) | local_win {r.get('home_win',0):4.1f}% draw {r.get('draw',0):4.1f}% visit_win {r.get('away_win',0):4.1f}%")

# ============================================================
banner("6. EFECTO LOCALÍA vs CANCHA NEUTRAL")
print("Distribución neutral:")
print(results["neutral"].value_counts().to_string())
for label, sub in [("NO neutral (hay local real)", results[~results["neutral"]]),
                   ("NEUTRAL", results[results["neutral"]])]:
    r = sub["result"].value_counts(normalize=True) * 100
    print(f"\n{label} (n={len(sub):,}):")
    print(f"  local_win {r.get('home_win',0):.1f}% | draw {r.get('draw',0):.1f}% | visit_win {r.get('away_win',0):.1f}%")
    print(f"  goles 'local' {sub['home_score'].mean():.2f} vs 'visitante' {sub['away_score'].mean():.2f}")

# Mundial: cuánto se juega en neutral
wc_neutral = wc["neutral"].mean() * 100
print(f"\nEn FIFA World Cup, % de partidos en cancha neutral: {wc_neutral:.1f}%")

# ============================================================
banner("7. COBERTURA DE LOS 48 CLASIFICADOS A 2026")
# Validar nombres contra el dataset
teams_set = set(teams)
missing = [t for t in ALL48 if t not in teams_set]
print(f"Equipos de los 48 que NO aparecen con ese nombre exacto en results: {missing if missing else 'ninguno ✓'}")

# Matches por equipo
long = pd.concat([
    results[["date", "home_team"]].rename(columns={"home_team": "team"}),
    results[["date", "away_team"]].rename(columns={"away_team": "team"}),
])
per_team = long.groupby("team").agg(partidos=("date", "size"),
                                    primer=("date", "min"),
                                    ultimo=("date", "max"))
cov = per_team.loc[per_team.index.intersection(ALL48)].copy()
cov["primer"] = cov["primer"].dt.year
cov["ultimo"] = cov["ultimo"].dt.year
# partidos desde 2010 (recientes, más relevantes)
recent = long[long["date"] >= "2010-01-01"].groupby("team").size()
cov["desde_2010"] = recent.reindex(cov.index).fillna(0).astype(int)
cov = cov.sort_values("partidos", ascending=False)
print(f"\nCobertura por equipo (de los que matchean, {len(cov)}/48):")
print(cov.to_string())
print(f"\nMenos datos históricos (top 8 con menos partidos totales):")
print(cov.nsmallest(8, "partidos").to_string())
print(f"\nDebutantes esperados / data más corta (primer partido más reciente):")
print(cov.nlargest(8, "primer")[["partidos", "primer", "desde_2010"]].to_string())

print(f"\nResumen cobertura: media {cov['partidos'].mean():.0f} partidos/equipo, "
      f"mín {cov['partidos'].min()}, máx {cov['partidos'].max()}")
print(f"Media partidos desde 2010: {cov['desde_2010'].mean():.0f}/equipo")

# Partidos entre dos equipos de los 48 (head-to-head útil)
mask48 = results["home_team"].isin(ALL48) & results["away_team"].isin(ALL48)
print(f"\nPartidos donde AMBOS equipos son de los 48: {mask48.sum():,}")
recent48 = results[mask48 & (results['date'] >= '2014-01-01')]
print(f"  de esos, desde 2014: {len(recent48):,}")

# ============================================================
banner("8. FIGURAS")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(10, 4))
results.groupby(results["date"].dt.year).size().plot(ax=ax)
ax.set_title("Partidos internacionales por año")
ax.set_ylabel("partidos")
fig.tight_layout(); fig.savefig(FIG / "partidos_por_anio.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
comp = pd.DataFrame({
    "no_neutral": results[~results["neutral"]]["result"].value_counts(normalize=True),
    "neutral": results[results["neutral"]]["result"].value_counts(normalize=True),
}).reindex(["home_win", "draw", "away_win"]) * 100
comp.plot(kind="bar", ax=ax)
ax.set_title("Resultado (%) según localía vs cancha neutral")
ax.set_ylabel("%"); ax.set_xticklabels(["local_win", "draw", "visit_win"], rotation=0)
fig.tight_layout(); fig.savefig(FIG / "localia_vs_neutral.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 8))
cov_sorted = cov.sort_values("desde_2010")
ax.barh(cov_sorted.index, cov_sorted["desde_2010"])
ax.set_title("Partidos desde 2010 — equipos clasificados 2026")
fig.tight_layout(); fig.savefig(FIG / "cobertura_48_desde2010.png", dpi=110); plt.close(fig)

print("Figuras guardadas en ./figures:", [p.name for p in FIG.glob("*.png")])
print("\n>>> EXPLORACIÓN COMPLETA <<<")
