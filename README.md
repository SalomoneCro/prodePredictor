# prode-predictor

A **Dixon–Coles** goal model for predicting international football matches, built to
play a *prode* — the Argentine name for a prediction pool where you bet the **exact
scoreline** of every match. It was made for the **2026 FIFA World Cup** pool, but the
core is reusable for any pool: you only swap the scoring rules.

No neural nets, no deep learning — just a weighted Poisson regression from a 1997 paper
([Dixon & Coles, *Applied Statistics*](https://www.math.ku.dk/~rolf/teaching/thesis/DixonColes.pdf)),
about 440 parameters, fit in a couple of seconds.

## Track record

- **Local pool (~11M players):** tied for **11th** (≈1.2k of us hit the same score).
- **Global pool (~400k players):** **4,700th** — top ~1%.

Backtested by temporal cross-validation (always trained only on matches *before* each
game predicted, so there is no leakage). Metric is **prode points per match** under the
2026 rules (3 exact / 1.5 close / 1 outcome / 0 miss):

| Backtest | Model pts/match | Baseline pts/match | 1X2 acc | exact-score |
|---|---|---|---|---|
| World Cups 2014/18/22 — group stage (n=144) | **2.229** | 1.417 | 59.7% | 14.6% |
| Competitive matches 2018–2025 (n=5,564)     | **2.266** | 1.732 | 61.5% | 14.1% |

## How it works

### 1. Goals as a Poisson process

Each team scores goals at an average rate $\lambda$ per match. The number of goals $k$
is modelled as a Poisson random variable:

$$P(X = k) = \frac{\lambda^{k} e^{-\lambda}}{k!}$$

The two rates in a match, $\lambda_H$ (home) and $\lambda_A$ (away), come from a
log-linear model of each team's **attack** and **defence** strength:

$$\log \lambda_H = \mu + \alpha_{\text{home}} + \delta_{\text{away}} + \gamma \cdot \mathbb{1}[\text{home advantage}]$$

$$\log \lambda_A = \mu + \alpha_{\text{away}} + \delta_{\text{home}}$$

where

- $\mu$ — global baseline scoring rate (intercept),
- $\alpha_t$ — **attack** of team $t$ (higher ⇒ scores more),
- $\delta_t$ — **defensive weakness** of team $t$ (higher ⇒ concedes more),
- $\gamma$ — **home advantage**, applied only when the match is *not* on neutral ground
  (in the 2026 fixtures, only the three hosts — Mexico, USA, Canada — play at home).

### 2. The Dixon–Coles low-score correction

A plain independent-Poisson model misprices the frequent low scores (0-0, 1-0, 0-1,
1-1). Dixon–Coles multiply those four cells by a correction $\tau$ governed by a single
parameter $\rho$:

$$
\tau_{\lambda_H,\lambda_A}(i,j) =
\begin{cases}
1 - \lambda_H \lambda_A \rho & (i,j)=(0,0) \\
1 + \lambda_H \rho & (i,j)=(0,1) \\
1 + \lambda_A \rho & (i,j)=(1,0) \\
1 - \rho & (i,j)=(1,1) \\
1 & \text{otherwise}
\end{cases}
$$

The full probability of the scoreline $i\text{-}j$ is then

$$P(\text{home}=i,\ \text{away}=j) = P_{\text{Pois}}(i;\lambda_H)\ P_{\text{Pois}}(j;\lambda_A)\ \tau_{\lambda_H,\lambda_A}(i,j)$$

Evaluated over $i,j \in \lbrace 0,\dots,10 \rbrace$ and renormalised, this gives a **score matrix**
$M$ — the joint distribution of the final scoreline. Everything downstream (1X2
probabilities, the optimal bet) is read off $M$ analytically, with no Monte Carlo:

$$P(\text{home win}) = \sum_{i>j} M_{ij}, \qquad P(\text{draw}) = \sum_{i} M_{ii}, \qquad P(\text{away win}) = \sum_{i<j} M_{ij}$$

## How it is trained

**Data.** ~190k international matches since 1872
([martj42 Kaggle dataset](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)),
overlaid with `data/wc2026.csv`, a hand-maintained file that carries the real 2026 group
stage results and the knockout fixtures still to be predicted.

**Weighted Poisson regression.** Every match becomes two observations (home attack vs.
away defence, and vice-versa). Attack/defence dummies, the intercept $\mu$ and the home
term $\gamma$ are fit by a **weighted Poisson GLM** — maximum likelihood on the goal
counts. Each observation carries a weight

$$w = w_{\text{time}} \cdot w_{\text{comp}}, \qquad w_{\text{time}}(t) = 0.5^{\,\Delta_t / h}$$

- $w_{\text{time}}$ — **exponential time decay**: $\Delta_t$ is how many days before the
  training cutoff the match was played, and $h$ is the half-life (default **1,095 days ≈
  3 years**), so a match 3 years old counts half as much as today's.
- $w_{\text{comp}}$ — **tournament importance**: World Cups / continental finals = 1.0,
  qualifiers = 0.9, friendlies = 1.0 (kept full — they are the main bridge between
  confederations), other = 0.7.

**The $\rho$ parameter** is fit separately by profiled maximum likelihood over the
low-score cells, maximising $\sum w \log \tau$.

**Default hyperparameters** (tuned on a rolling backtest, see `analysis/tune.py`): decay
half-life 3 years, 12-year training window, minimum 8 matches in-window for a team to be
included, score matrix truncated at 10 goals.

## From probabilities to a bet

The score matrix is rule-agnostic; the bet is whatever **maximises expected points**
under the pool's scoring table.

**Exact rule** (score gives `pts_exact`, right 1X2 gives `pts_outcome`, not additive):

$$\text{EV}(s) = \text{pts}_\text{outcome}\cdot P(\text{1X2 of } s) + (\text{pts}_\text{exact} - \text{pts}_\text{outcome})\cdot P(\text{score}=s)$$

**Close rule** (the 2026 pool: 3 exact / 1.5 close / 1 outcome / 0 miss). Because there
is an intermediate "close" tier, the expected value is summed over the whole matrix:

$$\text{EV}(\text{pick}) = \sum_{r} P(r)\ \text{points}(\text{pick}, r)$$

A pick $a\text{-}b$ is "close" to a real score $i\text{-}j$ when the outcome is right **and**

$$\text{CI}(a\text{-}b,\ i\text{-}j) = \big|(a-b)-(i-j)\big| + \tfrac{1}{2}\big|(a+b)-(i+j)\big| \le 1.5$$

**Knockout stage.** With `--extra-time`, draws at 90′ play a 30′ extra period (~⅓ of the
90′ scoring rate is added to each side as independent Poisson goals). Penalty shootouts
do not count for the pool, so a 120′ draw stays an `X`.

## Repository layout

```
src/prode/            installable package (the model)
  confederations.py   team → confederation map (exact dataset spelling)
  features.py         load_data() → (history, fixtures) with features + weights
  model.py            DixonColes: fit, score_matrix, and matrix utilities
scripts/              command-line entry points
  download_data.py    fetch the Kaggle dataset into data/
  predict.py          full 2026 group-stage prediction (predictions_2026.csv)
  predict_prode.py    rule-parametrised predictor (exact / close, extra time)
analysis/             development scripts (exploration, tuning, validation)
docs/                 plan.md (staged plan) and exploration_report.md
predictions/          generated prediction CSVs
kalshi/               side project: pricing Kalshi markets with the same model
data/                 datasets (Kaggle files git-ignored; wc2026.csv versioned)
```

## Setup & usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                      # installs the `prode` package
pip install -e ".[analysis,data]"     # + tuning/validation and Kaggle download extras

# put KAGGLE_USERNAME / KAGGLE_KEY in .env, then:
python scripts/download_data.py       # fetch the dataset into data/

# predict with the 2026 close-points rule:
python scripts/predict_prode.py --rule close

# a different pool, e.g. 5 pts exact / 2 pts for the right 1X2:
python scripts/predict_prode.py --rule exact --exact 5 --outcome 2

# knockout round (adds extra time to draws):
python scripts/predict_prode.py --rule close --extra-time
```

## Reusing it for another pool

The model (`att`/`def` → `score_matrix`) is independent of the scoring rules. Only the
bet selection is pool-specific, and it is already parametrised:

```python
from prode.model import DixonColes, best_bet
from prode.features import load_data

history, fixtures = load_data()
model = DixonColes().fit(history, cutoff="2026-06-04")
M = model.score_matrix("Argentina", "Brazil", neutral=True)

# 6 pts exact / 3 pts right 1X2:
i, j, ev = best_bet(M, pts_exact=6, pts_outcome=3)

# another pool (5 / 2): just change the numbers
i, j, ev = best_bet(M, pts_exact=5, pts_outcome=2)
```

The score matrix `M` holds all the probabilistic information; any other rule (goal
difference, double chance, …) is a new selection function over `M`, with no change to the
model itself. See `docs/plan.md` for the full design rationale.
