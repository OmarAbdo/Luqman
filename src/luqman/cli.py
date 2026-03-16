import argparse
import sys

from .config import Settings


def cmd_fetch(args, settings: Settings) -> int:
    from .data.fetcher import fetch_daily
    from .data.store import ParquetStore
    from .data.validation import validate_daily_ohlcv

    ticker = args.ticker or settings.ticker
    print(f"Fetching daily adjusted OHLCV for {ticker} from {args.start}...")
    df = fetch_daily(ticker, start=args.start)
    report = validate_daily_ohlcv(df)
    print(report.summary())
    if not report.ok:
        print("Refusing to store invalid data.")
        return 1
    path = ParquetStore(settings.data_dir).save_raw(ticker, df)
    print(f"Stored {len(df)} rows -> {path}")
    return 0


def _load_aligned(
    ticker: str,
    horizon: int,
    settings: Settings,
    with_news: bool = False,
    with_fundamentals: bool = False,
    with_macro: bool = False,
):
    """Stored raw -> validation gate -> features + target, aligned and NaN-free."""
    from .data.store import ParquetStore
    from .data.validation import validate_daily_ohlcv
    from .features.features import build_features
    from .features.targets import forward_log_return

    store = ParquetStore(settings.data_dir)
    ohlcv = store.load_raw(ticker)
    report = validate_daily_ohlcv(ohlcv)
    print(report.summary())
    if not report.ok:
        raise SystemExit("Stored data failed validation; refetch before evaluating.")

    features = build_features(ohlcv)
    if with_news:
        from .features.news_features import build_news_features

        articles = store.load_news(ticker)
        if articles is None or articles.empty:
            raise SystemExit(
                f"No stored news for {ticker}. Run: luqman fetch-news --ticker {ticker}"
            )
        news = build_news_features(articles, ohlcv.index)
        features = features.join(news, how="inner")
        print(
            f"news features joined: coverage {articles['time_published'].min():%Y-%m-%d} "
            f"-> {articles['time_published'].max():%Y-%m-%d}, {len(articles)} articles"
        )
    if with_fundamentals:
        from .features.fundamental_features import build_fundamental_features

        reports = store.load_table(f"{ticker.upper()}_fundamentals")
        if reports is None or reports.empty:
            raise SystemExit(
                f"No stored fundamentals for {ticker}. Run: luqman fetch-fundamentals --ticker {ticker}"
            )
        periods_per_year = (
            4 if "period_type" in reports.columns and reports["period_type"].iloc[0] == "quarter" else 1
        )
        features = features.join(
            build_fundamental_features(reports, ohlcv.index, periods_per_year=periods_per_year),
            how="inner",
        )
        print(f"fundamental features joined: {len(reports)} reports ({periods_per_year}x/year YoY lag)")
    if with_macro:
        from .features.macro_features import build_macro_features

        macro = store.load_table("macro_us")
        if macro is None or macro.empty:
            raise SystemExit("No stored macro data. Run: luqman fetch-macro")
        features = features.join(build_macro_features(macro, ohlcv.index), how="inner")
        print(f"macro features joined: {len(macro)} observations")
    target = forward_log_return(ohlcv["close"], horizon=horizon)
    aligned = features.join(target, how="inner").dropna()
    X = aligned[features.columns]
    y = aligned[target.name]
    print(f"{ticker}: {len(X)} aligned rows, {X.shape[1]} features, horizon {horizon}d")
    return X, y


def _model_factories(settings: Settings, include_gbm: bool):
    from .evaluation.evaluate import ZERO_BASELINE
    from .models.baselines import HistoricalMeanBaseline, ZeroReturnBaseline, make_ridge

    factories = {
        ZERO_BASELINE: ZeroReturnBaseline,
        "hist_mean": HistoricalMeanBaseline,
        "ridge": make_ridge,
    }
    if include_gbm:
        from .models.boosted import make_hist_gbm

        factories["hist_gbm"] = lambda: make_hist_gbm(seed=settings.random_seed)
    return factories


def _print_table(table) -> None:
    print()
    print(table.to_string(float_format=lambda v: f"{v: .6f}"))
    print(f"\nup-move base rate on test rows: {table.attrs['up_base_rate']:.3f}")
    print("skill_vs_zero > 0 and mase < 1 mean beating persistence; anything else is no skill.")


def cmd_baseline(args, settings: Settings) -> int:
    from .evaluation.evaluate import walk_forward_evaluate
    from .splits.walk_forward import WalkForwardSplitter

    ticker = args.ticker or settings.ticker
    X, y = _load_aligned(ticker, args.horizon, settings)
    table = walk_forward_evaluate(
        X, y, _model_factories(settings, include_gbm=False),
        WalkForwardSplitter(embargo=args.horizon),
    )
    _print_table(table)
    return 0


def cmd_experiment(args, settings: Settings) -> int:
    from .experiments.tracker import run_experiment
    from .splits.walk_forward import WalkForwardSplitter

    ticker = args.ticker or settings.ticker
    X, y = _load_aligned(
        ticker,
        args.horizon,
        settings,
        with_news=args.with_news,
        with_fundamentals=args.with_fundamentals,
        with_macro=args.with_macro,
    )
    if args.start:
        X, y = X.loc[args.start:], y.loc[args.start:]
        print(f"window restricted to {X.index[0].date()} -> {X.index[-1].date()} ({len(X)} rows)")
    table, record_path = run_experiment(
        name=args.name,
        ticker=ticker,
        X=X,
        y=y,
        model_factories=_model_factories(settings, include_gbm=True),
        splitter=WalkForwardSplitter(
            n_splits=args.n_splits, test_size=args.test_size, embargo=args.horizon
        ),
        experiments_dir=settings.experiments_dir,
        seed=settings.random_seed,
        horizon=args.horizon,
    )
    _print_table(table)
    print(f"\nRecorded -> {record_path}")
    return 0


def cmd_fetch_news(args, settings: Settings) -> int:
    import pandas as pd

    from .data.news import AlphaVantageNewsFetcher
    from .data.store import ParquetStore

    if not settings.alpha_vantage_key:
        print("ALPHA_VANTAGE_API is not set in .env")
        return 1

    ticker = args.ticker or settings.ticker
    store = ParquetStore(settings.data_dir)
    existing = store.load_news(ticker)
    start = pd.Timestamp(args.start, tz="UTC")
    if existing is not None and not existing.empty:
        resume = existing["time_published"].max() + pd.Timedelta(minutes=1)
        start = max(start, resume)
        print(f"Resuming from {start:%Y-%m-%d %H:%M} ({len(existing)} articles stored)")

    end = pd.Timestamp.now(tz="UTC")
    fetcher = AlphaVantageNewsFetcher(settings.alpha_vantage_key, ticker)

    def persist_page(page):
        stored = store.load_news(ticker)
        frames = [f for f in (stored, page) if f is not None and not f.empty]
        store.save_news(ticker, pd.concat(frames, ignore_index=True))

    fetcher.fetch_range(start, end, on_page=persist_page)

    combined = store.load_news(ticker)
    if combined is None or combined.empty:
        print("No articles fetched." + (f" API said: {fetcher.stopped_early}" if fetcher.stopped_early else ""))
        return 1
    path = store.save_news(ticker, combined)
    print(
        f"Stored {len(combined)} articles "
        f"({combined['time_published'].min():%Y-%m-%d} -> "
        f"{combined['time_published'].max():%Y-%m-%d}) -> {path}"
    )
    if fetcher.stopped_early:
        print(f"Stopped early (quota): {fetcher.stopped_early[:160]}")
        print("Re-run later to resume from the last stored article.")
    return 0


def cmd_backtest(args, settings: Settings) -> int:
    import pandas as pd

    from .backtest.engine import backtest_signals
    from .evaluation.evaluate import walk_forward_predictions
    from .splits.walk_forward import WalkForwardSplitter

    ticker = args.ticker or settings.ticker
    X, y = _load_aligned(ticker, args.horizon, settings)
    splitter = WalkForwardSplitter(embargo=args.horizon)

    rows = {}
    for name, factory in _model_factories(settings, include_gbm=True).items():
        preds = walk_forward_predictions(X, y, factory, splitter)
        result = backtest_signals(
            y, preds, cost_bps=args.cost_bps, allow_short=args.short
        )
        rows[name] = result.metrics

    table = pd.DataFrame(rows).T.sort_values("total_return", ascending=False)
    print()
    print(f"Out-of-fold backtest | cost {args.cost_bps} bps/side | "
          f"{'long/short' if args.short else 'long/flat'}")
    print(table.to_string(float_format=lambda v: f"{v: .4f}"))
    print("\nAll returns are cumulative log returns over the pooled test windows only.")
    print("buy_hold_return is the same-window benchmark every strategy must beat.")
    return 0


def cmd_nn(args, settings: Settings) -> int:
    import pandas as pd

    from .evaluation.evaluate import ZERO_BASELINE, walk_forward_evaluate
    from .experiments.tracker import run_experiment
    from .models.baselines import HistoricalMeanBaseline, ZeroReturnBaseline
    from .models.sequences import make_sequences
    from .models.torch_lstm import make_lstm_attention
    from .splits.walk_forward import WalkForwardSplitter

    ticker = args.ticker or settings.ticker
    horizons = args.horizons

    all_rows = []
    for horizon in horizons:
        print(f"\n=== horizon {horizon}d ===")
        X, y = _load_aligned(
            ticker, horizon, settings,
            with_news=args.with_news, with_fundamentals=args.with_fundamentals,
            with_macro=args.with_macro,
        )
        X_seq, y_seq, index = make_sequences(X, y, seq_len=args.seq_len)
        y_seq = pd.Series(y_seq, index=index, name=y.name)
        print(f"windowed into {len(X_seq)} sequences of length {args.seq_len}, {X_seq.shape[2]} features")

        splitter = WalkForwardSplitter(
            n_splits=args.n_splits, test_size=args.test_size, embargo=horizon
        )
        factories = {
            ZERO_BASELINE: ZeroReturnBaseline,
            "hist_mean": HistoricalMeanBaseline,
            "lstm_attn": lambda: make_lstm_attention(n_features=X_seq.shape[2], seed=settings.random_seed),
        }

        if args.name:
            table, record_path = run_experiment(
                name=f"{args.name}-h{horizon}",
                ticker=ticker,
                X=X_seq,
                y=y_seq,
                model_factories=factories,
                splitter=splitter,
                experiments_dir=settings.experiments_dir,
                seed=settings.random_seed,
                horizon=horizon,
            )
            print(f"Recorded -> {record_path}")
        else:
            table = walk_forward_evaluate(X_seq, y_seq, factories, splitter)

        _print_table(table)
        table = table.copy()
        table.insert(0, "horizon", horizon)
        all_rows.append(table)

    if len(horizons) > 1:
        summary = pd.concat(all_rows)
        print("\n=== horizon sweep summary ===")
        print(summary[["horizon", "mase", "dir_acc", "skill_vs_zero"]].to_string(
            float_format=lambda v: f"{v: .4f}"
        ))
    return 0


def cmd_audit_features(args, settings: Settings) -> int:
    from .evaluation.feature_audit import audit_features

    ticker = args.ticker or settings.ticker
    X, y = _load_aligned(
        ticker, args.horizon, settings,
        with_news=args.with_news, with_fundamentals=args.with_fundamentals, with_macro=args.with_macro,
    )
    table = audit_features(X, y, n_splits=args.n_splits)
    print()
    print(table.to_string(float_format=lambda v: f"{v: .4f}"))
    print(
        "\nic_full: Spearman rank corr(feature, forward return) over the whole series.\n"
        "ic_mean_blocks/ic_std_blocks: mean/std of IC computed separately in "
        f"{args.n_splits} chronological blocks — a real signal is consistent across blocks,\n"
        "not just a lucky whole-sample fit.\n"
        "ic_sign_consistency: fraction of blocks agreeing with the mean sign (0.5 = coin flip, pure noise).\n"
        "Rule of thumb: |ic_full| > ~0.03-0.05 AND sign_consistency > 0.6+ is a candidate for real signal;\n"
        "anything smaller/inconsistent is noise a model will only ever overfit, never generalize from."
    )
    return 0


def cmd_forecast_vol(args, settings: Settings) -> int:
    from .data.store import ParquetStore
    from .data.validation import validate_daily_ohlcv
    from .evaluation.evaluate import walk_forward_evaluate
    from .features.registry import build_registered_features
    from .features.vol_targets import volatility_target
    from .models.baselines import make_ridge
    from .models.boosted import make_hist_gbm
    from .splits.walk_forward import WalkForwardSplitter

    ticker = args.ticker or settings.ticker
    ohlcv = ParquetStore(settings.data_dir).load_raw(ticker)
    report = validate_daily_ohlcv(ohlcv)
    print(report.summary())
    if not report.ok:
        raise SystemExit("Stored data failed validation; refetch before evaluating.")

    features = build_registered_features(ohlcv, target_kind="volatility")
    target = volatility_target(ohlcv["close"], horizon=args.horizon, window=args.window)
    aligned = features.join(target, how="inner").dropna()
    X, y = aligned[features.columns], aligned[target.name]
    print(f"{ticker}: {len(X)} aligned rows, {X.shape[1]} registered volatility feature(s)")

    class PersistVol:
        """Baseline: next period's realized vol = today's trailing realized vol
        (the standard naive volatility forecast)."""

        def fit(self, X, y):
            return self

        def predict(self, X):
            return X[:, 0]  # vol_cluster IS the trailing realized vol, unscaled

    splitter = WalkForwardSplitter(embargo=args.horizon)
    factories = {
        "persist_vol": PersistVol,
        "ridge": make_ridge,
        "hist_gbm": lambda: make_hist_gbm(seed=settings.random_seed),
    }
    table = walk_forward_evaluate(X, y, factories, splitter)
    table = table.drop(columns=["dir_acc"], errors="ignore")
    if "persist_vol" in table.index:
        table["skill_vs_persist"] = 1.0 - table["mae"] / table.loc["persist_vol", "mae"]

    print()
    print(table.to_string(float_format=lambda v: f"{v: .6f}"))
    print(
        "\nskill_vs_persist > 0 means beating 'tomorrow's vol = today's trailing vol', "
        "the standard naive volatility baseline. This is a magnitude (risk-sizing) "
        "forecast, not a direction call."
    )
    return 0


def cmd_backtest_vol_sizing(args, settings: Settings) -> int:
    import numpy as np
    import pandas as pd

    from .backtest.vol_sizing import backtest_vol_targeting
    from .data.store import ParquetStore
    from .data.validation import validate_daily_ohlcv
    from .evaluation.evaluate import walk_forward_predictions
    from .features.registry import build_registered_features
    from .features.targets import forward_log_return
    from .features.vol_targets import volatility_target
    from .models.baselines import make_ridge
    from .splits.walk_forward import WalkForwardSplitter

    ticker = args.ticker or settings.ticker
    ohlcv = ParquetStore(settings.data_dir).load_raw(ticker)
    report = validate_daily_ohlcv(ohlcv)
    if not report.ok:
        raise SystemExit("Stored data failed validation; refetch before evaluating.")

    features = build_registered_features(ohlcv, target_kind="volatility")
    vol_target = volatility_target(ohlcv["close"], horizon=args.horizon, window=args.window)
    aligned = features.join(vol_target, how="inner").dropna()
    X, y = aligned[features.columns], aligned[vol_target.name]

    splitter = WalkForwardSplitter(embargo=args.horizon)
    ridge_forecast = walk_forward_predictions(X, y, make_ridge, splitter).clip(lower=1e-6)
    naive_forecast = aligned.loc[ridge_forecast.index, "vol_cluster"]  # persist_vol equivalent

    # Realized forward return over the position-holding window, aligned to
    # the same day the sizing decision is made (the forecast's own index).
    realized = forward_log_return(ohlcv["close"], horizon=args.horizon).reindex(ridge_forecast.index)
    daily_target_vol = args.target_vol_annual / np.sqrt(252)

    ridge_result = backtest_vol_targeting(realized, ridge_forecast, daily_target_vol, cost_bps=args.cost_bps)
    naive_result = backtest_vol_targeting(realized, naive_forecast, daily_target_vol, cost_bps=args.cost_bps)

    table = pd.DataFrame({"ridge_forecast_sizing": ridge_result, "naive_persist_sizing": naive_result}).T
    print(f"\n{ticker}: vol-targeting backtest, target {args.target_vol_annual:.0%} annualized vol, "
          f"{args.cost_bps} bps/side, {len(realized)} out-of-fold periods")
    print(table.to_string(float_format=lambda v: f"{v: .4f}"))
    print(
        "\nsized_realized_vol_annualized closer to target_vol_annualized = better risk control.\n"
        "Compare ridge_forecast_sizing vs naive_persist_sizing: does the trained forecast size\n"
        "risk better than just using today's trailing vol as tomorrow's forecast?\n"
        "fixed_realized_vol_annualized (same both rows) is the no-sizing-at-all baseline."
    )
    return 0


def cmd_backtest_momentum(args, settings: Settings) -> int:
    import pandas as pd

    from .backtest.cross_sectional import backtest_long_short
    from .data.store import ParquetStore
    from .data.validation import validate_daily_ohlcv
    from .features.registry import CROSS_SECTIONAL_REGISTRY
    from .features.targets import forward_log_return

    store = ParquetStore(settings.data_dir)
    factor = CROSS_SECTIONAL_REGISTRY["momentum_12_1_xs"]

    ohlcv_by_ticker = {}
    for path in sorted(settings.data_dir.glob("raw/*_daily.parquet")):
        ticker = path.stem.replace("_daily", "")
        df = store.load_raw(ticker)
        if validate_daily_ohlcv(df).ok:
            ohlcv_by_ticker[ticker] = df
    print(f"Universe: {len(ohlcv_by_ticker)} tickers with valid stored data")

    factor_panel = factor.build(ohlcv_by_ticker)
    return_panel = pd.DataFrame(
        {t: forward_log_return(df["close"], horizon=factor.horizon) for t, df in ohlcv_by_ticker.items()}
    )

    result = backtest_long_short(
        factor_panel, return_panel, quantile=args.quantile, cost_bps=args.cost_bps
    )
    print(f"\n{factor.name} long-short backtest | quantile {args.quantile:.0%} | {args.cost_bps} bps/side turnover cost")
    print(f"mechanism: {factor.mechanism}")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:+.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\nzero_baseline_return: a market-neutral long-short book has no natural buy-and-hold benchmark;")
    print("compare total_return/sharpe against doing nothing (0), not against an index.")
    return 0


def cmd_fetch_fundamentals(args, settings: Settings) -> int:
    from .data.fundamentals import fetch_income_statements
    from .data.store import ParquetStore

    if not settings.fmp_key:
        print("FMP_API is not set in .env")
        return 1
    ticker = args.ticker or settings.ticker
    reports = fetch_income_statements(ticker, settings.fmp_key, period=args.period)
    reports["period_type"] = args.period
    path = ParquetStore(settings.data_dir).save_table(f"{ticker.upper()}_fundamentals", reports)
    print(
        f"Stored {len(reports)} {args.period} reports "
        f"({reports['period_end'].min():%Y-%m-%d} -> {reports['period_end'].max():%Y-%m-%d}) -> {path}"
    )
    return 0


def cmd_fetch_macro(args, settings: Settings) -> int:
    from .data.macro import AlphaVantageMacroFetcher
    from .data.store import ParquetStore

    if not settings.alpha_vantage_key:
        print("ALPHA_VANTAGE_API is not set in .env")
        return 1
    macro = AlphaVantageMacroFetcher(settings.alpha_vantage_key).fetch()
    path = ParquetStore(settings.data_dir).save_table("macro_us", macro)
    print(
        f"Stored macro series {list(macro.columns)} "
        f"({macro.index.min():%Y-%m-%d} -> {macro.index.max():%Y-%m-%d}, {len(macro)} rows) -> {path}"
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="luqman", description="Luqman forecasting engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch and store daily adjusted OHLCV")
    p_fetch.add_argument("--ticker", default=None)
    p_fetch.add_argument("--start", default="2000-01-01")
    p_fetch.set_defaults(func=cmd_fetch)

    p_base = sub.add_parser("baseline", help="Walk-forward evaluate baselines + ridge")
    p_base.add_argument("--ticker", default=None)
    p_base.add_argument("--horizon", type=int, default=1)
    p_base.set_defaults(func=cmd_baseline)

    p_exp = sub.add_parser(
        "experiment", help="Tracked walk-forward run: baselines + ridge + gradient boosting"
    )
    p_exp.add_argument("--name", required=True, help="Experiment name for the record")
    p_exp.add_argument("--ticker", default=None)
    p_exp.add_argument("--horizon", type=int, default=1)
    p_exp.add_argument("--with-news", action="store_true", help="Join daily news-sentiment features")
    p_exp.add_argument("--start", default=None, help="Restrict evaluation window (YYYY-MM-DD)")
    p_exp.add_argument("--with-fundamentals", action="store_true", help="Join report-date-aligned fundamental features")
    p_exp.add_argument("--with-macro", action="store_true", help="Join availability-lagged macro features")
    p_exp.add_argument("--n-splits", type=int, default=5)
    p_exp.add_argument("--test-size", type=int, default=63)
    p_exp.set_defaults(func=cmd_experiment)

    p_news = sub.add_parser("fetch-news", help="Fetch/resume historical news sentiment articles")
    p_news.add_argument("--ticker", default=None)
    p_news.add_argument("--start", default="2022-03-01", help="Earliest publication date (UTC)")
    p_news.set_defaults(func=cmd_fetch_news)

    p_fund = sub.add_parser("fetch-fundamentals", help="Fetch income statements (FMP)")
    p_fund.add_argument("--ticker", default=None)
    p_fund.add_argument("--period", choices=["annual", "quarter"], default="annual",
                        help="Free FMP tier caps history at 5 statements; annual maximizes YoY coverage")
    p_fund.set_defaults(func=cmd_fetch_fundamentals)

    p_macro = sub.add_parser("fetch-macro", help="Fetch daily macro series (Alpha Vantage)")
    p_macro.set_defaults(func=cmd_fetch_macro)

    p_bt = sub.add_parser("backtest", help="Costed backtest of out-of-fold predictions")
    p_bt.add_argument("--ticker", default=None)
    p_bt.add_argument("--horizon", type=int, default=1)
    p_bt.add_argument("--cost-bps", type=float, default=5.0)
    p_bt.add_argument("--short", action="store_true", help="Allow short positions")
    p_bt.set_defaults(func=cmd_backtest)

    p_nn = sub.add_parser(
        "nn", help="LSTM+attention sequence model, walk-forward evaluated (optionally swept over horizons)"
    )
    p_nn.add_argument("--ticker", default=None)
    p_nn.add_argument("--horizons", type=int, nargs="+", default=[1],
                       help="One or more horizons (days) to evaluate, e.g. --horizons 1 5 20")
    p_nn.add_argument("--seq-len", type=int, default=30)
    p_nn.add_argument("--with-news", action="store_true")
    p_nn.add_argument("--with-fundamentals", action="store_true")
    p_nn.add_argument("--with-macro", action="store_true")
    p_nn.add_argument("--n-splits", type=int, default=5)
    p_nn.add_argument("--test-size", type=int, default=63)
    p_nn.add_argument("--name", default=None, help="If set, records a tracked experiment per horizon")
    p_nn.set_defaults(func=cmd_nn)

    p_audit = sub.add_parser(
        "audit-features", help="Per-feature IC/mutual-information audit: does the data carry any signal?"
    )
    p_audit.add_argument("--ticker", default=None)
    p_audit.add_argument("--horizon", type=int, default=1)
    p_audit.add_argument("--with-news", action="store_true")
    p_audit.add_argument("--with-fundamentals", action="store_true")
    p_audit.add_argument("--with-macro", action="store_true")
    p_audit.add_argument("--n-splits", type=int, default=5)
    p_audit.set_defaults(func=cmd_audit_features)

    p_vol = sub.add_parser(
        "forecast-vol", help="Track A: walk-forward volatility forecast using registered proven features"
    )
    p_vol.add_argument("--ticker", default=None)
    p_vol.add_argument("--horizon", type=int, default=5)
    p_vol.add_argument("--window", type=int, default=5)
    p_vol.set_defaults(func=cmd_forecast_vol)

    p_vol_bt = sub.add_parser(
        "backtest-vol-sizing",
        help="Track A practical value: does the vol forecast improve position sizing vs. naive/fixed?",
    )
    p_vol_bt.add_argument("--ticker", default=None)
    p_vol_bt.add_argument("--horizon", type=int, default=5)
    p_vol_bt.add_argument("--window", type=int, default=5)
    p_vol_bt.add_argument("--target-vol-annual", type=float, default=0.15)
    p_vol_bt.add_argument("--cost-bps", type=float, default=5.0)
    p_vol_bt.set_defaults(func=cmd_backtest_vol_sizing)

    p_mom_bt = sub.add_parser(
        "backtest-momentum",
        help="Costed long-short momentum_12_1 backtest across all stored tickers",
    )
    p_mom_bt.add_argument("--quantile", type=float, default=0.2)
    p_mom_bt.add_argument("--cost-bps", type=float, default=10.0)
    p_mom_bt.set_defaults(func=cmd_backtest_momentum)

    args = parser.parse_args(argv)
    return args.func(args, Settings.load())


if __name__ == "__main__":
    sys.exit(main())
