"""The equity universe used for cross-sectional factor work.

Why this is a package module rather than a list inside a script: a
cross-sectional factor ranks names against each other, so WHICH names are in
the ranking is part of the method, not a detail of one experiment. The store
also holds non-equity instruments (SPY, GLD, USO, BTC-USD) that were fetched
for other tests; letting those fall into an equity long-short book by way of a
directory glob would silently corrupt the ranking. Keeping the definition here
means the CLI and the proof scripts rank exactly the same set.

Composition: 52 large-caps spanning all major GICS sectors, all liquid enough
to trade in practice.

Known bias, stated wherever results depend on it: these names were chosen from
the large-caps of 2026, so the deeper history is survivorship-biased. That
inflates absolute return levels; it does not plausibly explain relative
rankings within the set, since every member carries the same bias.
"""

EQUITY_UNIVERSE: list[str] = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO", "IBM",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT",
    # Consumer discretionary / staples
    "AMZN", "TSLA", "HD", "MCD", "NKE", "WMT", "KO", "PEP", "PG", "COST",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Industrials
    "CAT", "BA", "HON", "UPS", "GE",
    # Communication services
    "GOOGL", "META", "NFLX", "DIS", "CMCSA",
    # Utilities / materials (thinner representation, but present)
    "NEE", "LIN",
]

# Instruments held in the same store for other purposes (benchmarks, cross-asset
# tests). Never members of an equity cross-sectional ranking.
NON_EQUITY_INSTRUMENTS: frozenset[str] = frozenset({"SPY", "GLD", "USO", "BTC-USD"})
