import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..evaluation.evaluate import walk_forward_evaluate
from ..splits.walk_forward import WalkForwardSplitter


def _git_sha(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def data_hash(aligned: pd.DataFrame) -> str:
    """Stable content hash of the exact rows/values a run trained and tested on."""
    return f"{int(pd.util.hash_pandas_object(aligned, index=True).sum()):016x}"


def _array_hash(X, y: pd.Series) -> str:
    """Content hash for a pre-windowed sequence array, paired with its targets."""
    digest = pd.util.hash_array(np.ascontiguousarray(X).ravel())
    combined = int(digest.sum()) ^ int(pd.util.hash_pandas_object(y, index=True).sum())
    return f"{combined & ((1 << 64) - 1):016x}"


def run_experiment(
    name: str,
    ticker: str,
    X,
    y: pd.Series,
    model_factories: dict[str, Callable[[], object]],
    splitter: WalkForwardSplitter,
    experiments_dir: Path,
    seed: int,
    horizon: int,
) -> tuple[pd.DataFrame, Path]:
    """Run a walk-forward evaluation and persist a full record of it.

    X may be a tabular DataFrame or a pre-windowed 3D sequence array (see
    evaluation.evaluate._select) — both are hashed and recorded identically.

    Writes <experiments_dir>/<timestamp>_<name>.json (complete config + metrics)
    and appends one row per model to <experiments_dir>/index.csv. An experiment
    that isn't recorded didn't happen — this is the replacement for 'manual tests.txt'.
    """
    table = walk_forward_evaluate(X, y, model_factories, splitter)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    is_tabular = isinstance(X, pd.DataFrame)
    n_features = int(X.shape[1]) if is_tabular else int(X.shape[2])
    content_hash = data_hash(X.assign(**{y.name: y})) if is_tabular else _array_hash(X, y)
    feature_names = list(X.columns) if is_tabular else None

    record = {
        "name": name,
        "timestamp": timestamp,
        "ticker": ticker,
        "git_sha": _git_sha(experiments_dir.parent),
        "seed": seed,
        "horizon": horizon,
        "data": {
            "n_rows": int(len(y)),
            "n_features": n_features,
            "hash": content_hash,
            "start": str(y.index[0].date()),
            "end": str(y.index[-1].date()),
        },
        "splitter": asdict(splitter),
        "features": feature_names,
        "up_base_rate": table.attrs["up_base_rate"],
        "metrics": {model: row.to_dict() for model, row in table.iterrows()},
    }

    experiments_dir.mkdir(parents=True, exist_ok=True)
    record_path = experiments_dir / f"{timestamp}_{name}.json"
    record_path.write_text(json.dumps(record, indent=2))

    index_rows = table.reset_index(names="model")
    index_rows.insert(0, "timestamp", timestamp)
    index_rows.insert(1, "name", name)
    index_rows.insert(2, "ticker", ticker)
    index_path = experiments_dir / "index.csv"
    index_rows.to_csv(index_path, mode="a", header=not index_path.exists(), index=False)

    return table, record_path
