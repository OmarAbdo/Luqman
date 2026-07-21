# Luqman

![Luqman Cover Image](luqman.jpg)

A stock forecasting engine that ships only what beat a real baseline out of sample.

Most of what it tried did not work, and the parts that did not work are documented as carefully as the parts that did. That is the point of the project: forecasting equity returns is mostly a story of rejecting plausible ideas, and a tool that hides its failures cannot be trusted about its successes.

---

## What Luqman actually does

**Two things are proven and served:**

| Capability | What it answers | Status |
|---|---|---|
| **Volatility forecasting** | "How big is the next move likely to be?" | Beats persistence on 8/8 daily tickers (mean +10.6% skill, 2000–2026) and again on 5-minute bars (+21% to +30%) |
| **Cross-sectional factor ranking** | "Which stocks should be held long vs short, as a market-neutral book?" | Two admitted factors; best is net Sharpe +1.55 after 10 bps costs over 2000–2026 |

**One thing is not offered, on purpose:**

> **There is no single-name direction forecast** — no "will AAPL go up tomorrow". Ridge regression, gradient boosting, and an LSTM with attention were each evaluated against a persistence baseline at 1-, 5- and 20-day horizons, and a ridge model again on 5-minute bars. **None of them beat the baseline at any horizon.** A feature audit found the reason: every feature tested had a rank correlation with future returns below 0.06, an order of magnitude under what is considered actionable. Serving a direction number would mean shipping noise with a confident face on it.

Volatility is the honest product here. Knowing that a stock's typical 5-day move is 4.7% rather than 1.2% is genuinely useful for position sizing and risk budgeting — it just is not a buy signal, and Luqman will not dress it up as one.

---

## Install

Requires Python 3.12+.

```bash
git clone <your-repo-url>
cd Luqman

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -e ".[dev,serve]"     # add ",torch" for the neural-network experiments
```

Verify:

```bash
pytest          # 188 tests
luqman --help
```

### API keys (optional)

Core features — price data, volatility forecasting, both admitted factors — need **no API keys**. Price data comes from Yahoo Finance and factor data from the Ken French library, both free and unauthenticated.

Keys are only needed to re-run the *rejected* experiments (news sentiment, fundamentals, macro). Copy `.env-example` to `.env` and fill in what you need:

```bash
TICKER=AAPL                 # default ticker for commands that take one
FMP_API=...                 # Financial Modeling Prep — fundamentals, earnings, analyst grades
ALPHA_VANTAGE_API=...       # Alpha Vantage — news sentiment, macro series
```

---

## Quickstart

```bash
# 1. Fetch daily price data for one ticker
luqman fetch --ticker AAPL

# 2. Forecast its volatility — the project's most robust result
luqman forecast-vol --ticker AAPL --horizon 5

# 3. See everything that is proven, and everything that was rejected
luqman list-features
```

For the cross-sectional factors you need the whole universe plus factor returns:

```bash
python scripts/fetch_universe.py 2000-01-01   # 52 large-caps, ~3 minutes
luqman fetch-factors                          # Fama-French daily factors

luqman backtest-factor --factor residual_momentum_xs
```

Then start the API:

```bash
luqman serve      # http://127.0.0.1:8000/docs
```

---

## Using the proven features

### 1. Volatility forecasting — for position sizing

The one single-name feature that cleared every proof gate is **`vol_cluster`**: trailing realized volatility predicting future realized volatility. This is volatility clustering (the ARCH/GARCH effect) — calm periods follow calm periods, turbulent follow turbulent. It is one of the most robust effects in empirical finance, and it held here with IC 0.21–0.55 across all 8 test tickers with perfect sign consistency.

```bash
# Walk-forward evaluation: how well does it actually forecast?
luqman forecast-vol --ticker AAPL --horizon 5
```

```text
                  mae      rmse      mase  skill_vs_persist
ridge        0.006777  0.008260  0.343061          0.113889   <- beats the naive baseline
hist_gbm     0.007254  0.008797  0.367234          0.051455
persist_vol  0.007648  0.009661  0.387118          0.000000
```

`skill_vs_persist > 0` means beating "tomorrow's volatility equals today's" — the standard naive volatility benchmark. `mase < 1` means beating a naive in-sample scaling. Ridge wins on **8/8 tickers** and beat purpose-built GJR-GARCH head to head.

Per-ticker skill over 2000–2026: AAPL +11.4%, MSFT +10.0%, JPM +6.2%, XOM +11.6%, JNJ +13.7%, WMT +16.1%, NVDA +5.3%, KO +10.3% — **mean +10.6%, positive on every name.**

**Practical use — sizing a position:**

```bash
luqman backtest-vol-sizing --ticker AAPL --target-vol-annual 0.15
```

This sizes positions as `target_vol / forecast_vol`, so exposure shrinks when the model expects turbulence. Across 8 tickers it cut realized volatility substantially (AAPL: 62.8% → 39.4% annualized).

> **Honest caveat, because it matters:** against *naive persistence* specifically, ridge lands closer to the target volatility on only 2 of 8 tickers. Its consistent win is **turnover** — lower on 8/8, often by half — meaning smoother sizing and lower trading costs. It is a cost-efficiency win, not an accuracy win. Do not oversell it.

Via the API:

```bash
curl "http://127.0.0.1:8000/forecast/volatility/AAPL?horizon=5"
```

```json
{
  "ticker": "AAPL",
  "horizon_days": 5,
  "forecast_annualized_vol": 0.331,
  "expected_move_pct": 4.67,
  "interpretation": "Over the next 5 trading days, a typical (one standard deviation) move in AAPL is about 4.67%, in EITHER direction. ..."
}
```

### 2. Cross-sectional factors — for a market-neutral book

Two factors cleared the bar. Both are **cross-sectional**: they rank the whole universe and trade the spread between the top and bottom quintile. Neither says anything about an individual stock in isolation.

| Factor | Mechanism | Net Sharpe @10bps | Max drawdown |
|---|---|---|---|
| `momentum_12_1_xs` | Jegadeesh-Titman momentum: 12-month return excluding the last month | +1.01 | −13.77 |
| **`residual_momentum_xs`** | Same ranking, but on Fama-French **residual** returns | **+1.55** | **−5.79** |

**Use `residual_momentum_xs`.** Stripping market/size/value exposure out of the ranking gives higher risk-adjusted return *and less than half the drawdown*. Notably, blending the two is **worse** (+1.46) than residual momentum alone — residual momentum supersedes raw momentum rather than complementing it.

```bash
# Backtest with costs
luqman backtest-factor --factor residual_momentum_xs --cost-bps 10

# Today's long/short book
curl "http://127.0.0.1:8000/factors/residual_momentum_xs/ranking"
```

> **Caveats that travel with these numbers.** The backtest rebalances daily against overlapping forward returns, which inflates Sharpe through autocorrelation — figures are comparable *between factors here* but not to published values. The universe was selected from 2026 large-caps, so its history is survivorship-biased. And the momentum premium is documented to have decayed (roughly 10%/yr in the 1990s to ~2%/yr recently), so treat the historical magnitude as an upper bound.
>
> Factor rankings also lag by a few weeks: the Ken French library publishes with a delay, so `as_of` on a residual-momentum ranking will trail today's date.

### 3. Intraday

The same methodology re-run on 5-minute bars (AAPL 2012–2024, 246k regular-hours bars, plus NVDA and SAP):

- **Direction: still nothing.** Skill vs zero baseline between −0.28% and +0.12%; directional accuracy ~51% against a ~51% base rate.
- **Volatility: stronger than daily.** +21% to +30% skill vs persistence, MASE 0.33–0.39 on every ticker.

```bash
python scripts/run_intraday_proof.py
```

Intraday returns are **session-aware**: nothing is ever computed across the overnight gap. Treating a 15:55 → 09:30 gap as a five-minute move would inject the sample's largest returns at fixed daily intervals and manufacture a time-of-day pattern any model would happily learn.

---

## How something becomes "proven"

Nothing reaches a model without clearing four gates. This process exists because the project's first iteration added features on the basis of "this data is available" and ended up with 24 features none of which carried signal.

1. **Named mechanism** — a documented, citable reason the feature should predict the target, and the horizon at which that mechanism operates. Not a vibe, and not chosen after seeing results.
2. **Isolated IC audit** at the mechanism's own horizon — `|IC| ≥ 0.08` and sign consistency ≥ 0.70 across chronological blocks.
3. **Multi-ticker stability** — must hold on at least 6 of 8 sector-diverse tickers. Cross-sectional factors instead face a factor-research bar (Sharpe > 0.3, IC+ rate > 0.52) **plus a split-half sign-stability check**.
4. **Marginal contribution** — must improve out-of-fold skill when added to what is already admitted, not merely score well alone.

Then the **costed backtest is the final arbiter.** A signal that looks good and dies on transaction costs has not been proven; it has been disproven.

The registry (`luqman/features/registry.py`) enforces this structurally: a model can only consume registered features, so "an unproven feature reached a model" is a lookup failure rather than a matter of discipline. Rejected candidates are kept in a separate dict with their full post-mortem, and there is a test asserting the two can never overlap.

```bash
luqman list-features    # the full inventory, admitted and rejected
```

### What this process caught

Two Wave 3 candidates passed the headline statistics and were still rejected:

- **Idiosyncratic volatility** posted the best numbers of anything ever tested here (net Sharpe +1.39, best IC, biggest blend improvement) — and produced the *opposite sign* to the effect it claimed to measure. Inspecting the book showed why: the long leg was AMD/NVDA/TSLA/NFLX at mean market beta 1.35, the short leg JNJ/KO/PG at 0.73, with IVOL correlating +0.61 with beta. It was a levered long-beta tech bet measured across a tech bull market — compensated risk, not alpha.
- **Amihud illiquidity** posted a +2.27 pooled Sharpe that decomposed into +5.11 in the first half of the sample and −0.12 in the second. Entirely a pre-2013 artifact.

Both would have looked like discoveries under a process that only checked whether the number was good.

---

## Command reference

| Command | Purpose |
|---|---|
| `luqman fetch --ticker X` | Fetch and store daily adjusted OHLCV |
| `luqman fetch-factors` | Fetch Fama-French daily factors (needed for residual features) |
| `luqman list-features` | Show every admitted feature and every rejected candidate |
| `luqman forecast-vol --ticker X` | **Volatility forecast, walk-forward evaluated** |
| `luqman backtest-vol-sizing --ticker X` | Does the vol forecast improve position sizing? |
| `luqman backtest-factor --factor NAME` | **Costed long-short factor backtest** (repeat `--factor` to blend) |
| `luqman serve` | Run the HTTP API |
| `luqman baseline --ticker X` | Walk-forward baselines + ridge on daily returns |
| `luqman experiment --name N` | Tracked experiment, recorded to `experiments/` |
| `luqman audit-features --ticker X` | Per-feature IC audit — does the data carry signal at all? |
| `luqman backtest --ticker X` | Costed backtest of out-of-fold return predictions |
| `luqman nn --horizons 1 5 20` | LSTM+attention (requires the `torch` extra) |
| `luqman fetch-news/-fundamentals/-macro` | Aux data for the rejected-signal experiments |

## API reference

| Endpoint | Returns |
|---|---|
| `GET /health` | Liveness |
| `GET /features` | Full registry — admitted and rejected, with mechanisms |
| `GET /forecast/volatility/{ticker}` | Expected move magnitude + interpretation |
| `GET /factors` | Admitted cross-sectional factors |
| `GET /factors/{name}/ranking` | Today's long/short book |
| `GET /factors/{name}/backtest` | Costed backtest report |

Requesting a rejected factor returns 404 by design.

---

## Project layout

```
src/luqman/
  data/          fetching, validation, parquet store, universe definition,
                 point-in-time joins, Fama-French factors, intraday sessions
  features/      feature construction + the registry (the admission gate)
  splits/        walk-forward splitting with embargo
  models/        baselines, ridge, gradient boosting, GARCH, LSTM+attention
  evaluation/    metrics, IC audits, the proof harness, cross-sectional tests
  backtest/      costed backtests: single-name, long-short, vol-sizing
  serving/       FastAPI surface
scripts/         proof runs — one per research wave, results recorded in docs
tests/           188 tests
.refactor/       the full research record (see below)
```

## The research record

Every claim in this README traces to a recorded run with numbers:

- **`.refactor/dataset-plan.md`** — the proof standard and every feature verdict, including full post-mortems on the rejections. The authoritative document.
- **`.refactor/plan.md`** — engineering phases and their outcomes.
- **`.refactor/ds-critique.md`** — the methodology audit of the original system that started the rebuild.
- **`.refactor/diagnosis.md`** — the software audit of the original system.
- **`experiments/`** — machine-readable run records (JSON + index CSV).

## Honest expectations

Short-horizon single-name price direction is among the hardest prediction targets in the field, and this project's own evidence says it did not crack it. Out of roughly two dozen candidate signals spanning technical, sentiment, fundamental, macro, geopolitical and cross-asset categories, **three cleared the bar**: one volatility feature and two cross-sectional momentum factors, of which one supersedes the other.

That ratio is not a disappointment; it is what the sober literature predicts, and real-time replications routinely find published factor performance substantially weaker out of sample. A tool that claimed more would be lying.
