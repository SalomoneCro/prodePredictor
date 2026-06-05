"""Confirma en el WC backtest (escenario neutral) los candidatos del tuning."""
import pandas as pd
from features import load_data
from model import DixonColes
from validate import evaluate

history, _ = load_data()
wc = history[history["tournament"] == "FIFA World Cup"]

configs = {
    "default (hl=730, fw=0.5)": (730, 0.5),
    "hl=730, fw=1.0": (730, 1.0),
    "hl=1095, fw=1.0": (1095, 1.0),
    "hl=1095, fw=0.75": (1095, 0.75),
}
print(f"{'config':28} {'PTS/part':>9} {'RPS':>8} {'total pts':>10}")
for name, (hl, fw) in configs.items():
    evs = []
    for year in (2014, 2018, 2022):
        ed = wc[wc["date"].dt.year == year].sort_values("date").head(48)
        cutoff = ed["date"].min() - pd.Timedelta(days=1)
        train = history[history["date"] <= cutoff]
        model = DixonColes().fit(history, cutoff=cutoff, half_life_days=hl,
                                 friendly_weight=fw, verbose=False)
        evs.append(evaluate(model, ed, train))
    df = pd.concat(evs)
    print(f"{name:28} {df.model_pts.mean():>9.4f} {df.model_rps.mean():>8.4f} "
          f"{df.model_pts.sum():>10}")
