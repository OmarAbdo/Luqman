import pandas as pd

from luqman.data.fetcher import OHLCV_COLUMNS, normalize_ohlcv
from luqman.data.store import ParquetStore


def test_normalize_multiindex_and_tz(ohlcv):
    raw = ohlcv.copy()
    raw.columns = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["TEST"]])
    raw.index = raw.index.tz_localize("America/New_York")
    out = normalize_ohlcv(raw)
    assert list(out.columns) == OHLCV_COLUMNS
    assert out.index.tz is None
    assert out.index.name == "date"
    assert out.index.is_monotonic_increasing


def test_normalize_dedupes_and_sorts(ohlcv):
    raw = pd.concat([ohlcv, ohlcv.iloc[:5]]).iloc[::-1]
    out = normalize_ohlcv(raw)
    assert not out.index.has_duplicates
    assert out.index.is_monotonic_increasing
    assert len(out) == len(ohlcv)


def test_store_round_trip(tmp_path, ohlcv):
    store = ParquetStore(tmp_path)
    store.save_raw("test", ohlcv)
    loaded = store.load_raw("TEST")
    pd.testing.assert_frame_equal(loaded, ohlcv, check_freq=False)


def test_store_missing_raises(tmp_path):
    store = ParquetStore(tmp_path)
    try:
        store.load_raw("NOPE")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
