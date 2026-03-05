from pathlib import Path

import pandas as pd


class ParquetStore:
    """One parquet file per ticker under <data_dir>/raw. The only place that knows storage paths."""

    def __init__(self, data_dir: Path):
        self.raw_dir = Path(data_dir) / "raw"

    def _path(self, ticker: str) -> Path:
        return self.raw_dir / f"{ticker.upper()}_daily.parquet"

    def save_raw(self, ticker: str, df: pd.DataFrame) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(ticker)
        df.to_parquet(path)
        return path

    def load_raw(self, ticker: str) -> pd.DataFrame:
        path = self._path(ticker)
        if not path.exists():
            raise FileNotFoundError(
                f"No stored data for {ticker!r}. Run: luqman fetch --ticker {ticker}"
            )
        return pd.read_parquet(path)

    def _news_path(self, ticker: str) -> Path:
        return self.raw_dir / f"{ticker.upper()}_news.parquet"

    def save_news(self, ticker: str, articles: pd.DataFrame) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self._news_path(ticker)
        deduped = (
            articles.drop_duplicates(subset="url")
            .sort_values("time_published")
            .reset_index(drop=True)
        )
        deduped.to_parquet(path)
        return path

    def load_news(self, ticker: str) -> pd.DataFrame | None:
        path = self._news_path(ticker)
        return pd.read_parquet(path) if path.exists() else None

    def save_table(self, name: str, df: pd.DataFrame) -> Path:
        """Generic auxiliary table (fundamentals, macro, ...)."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dir / f"{name}.parquet"
        df.to_parquet(path)
        return path

    def load_table(self, name: str) -> pd.DataFrame | None:
        path = self.raw_dir / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else None
