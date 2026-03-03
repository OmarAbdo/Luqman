import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    experiments_dir: Path
    ticker: str
    random_seed: int
    alpha_vantage_key: str | None = None
    fmp_key: str | None = None

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")
        return cls(
            alpha_vantage_key=os.getenv("ALPHA_VANTAGE_API"),
            fmp_key=os.getenv("FMP_API"),
            data_dir=Path(os.getenv("LUQMAN_DATA_DIR", str(PROJECT_ROOT / "data"))),
            experiments_dir=Path(
                os.getenv("LUQMAN_EXPERIMENTS_DIR", str(PROJECT_ROOT / "experiments"))
            ),
            ticker=os.getenv("TICKER", "AAPL"),
            random_seed=int(os.getenv("LUQMAN_SEED", "42")),
        )
