"""Проверки формул индикаторов и сборки баров. Сеть не нужна."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import add_indicators, drop_open_bar, resample_candles
from .moex import MSK


def _candles(closes: list[float], start: str = "2026-11-02 10:00:00") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(closes), freq="min", tz=MSK)
    close = np.array(closes, dtype="float64")
    return pd.DataFrame(
        {
            "begin": index,
            "end": index + pd.Timedelta(seconds=59),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(len(close), dtype="int64"),
            "value": close * 10,
        }
    )


def test_sma_and_ema_seed() -> None:
    frame = _candles([1, 2, 3, 4, 5])
    out = add_indicators(frame, ["sma", "ema"], {"sma": [3], "ema": [3]})
    sma = out["sma_3"].to_numpy()
    assert np.isnan(sma[0]) and np.isnan(sma[1])
    assert sma[2:].tolist() == [2, 3, 4]
    ema = out["ema_3"].to_numpy()
    assert np.isnan(ema[0]) and np.isnan(ema[1])
    assert ema[2] == 2
    assert ema[3] == 0.5 * 4 + 0.5 * 2
    assert ema[4] == 0.5 * 5 + 0.5 * ema[3]


def test_rsi_all_gains_is_100() -> None:
    frame = _candles(list(range(1, 21)))
    out = add_indicators(frame, ["rsi"], {"rsi": 14})
    rsi = out["rsi_14"].to_numpy()
    assert np.isnan(rsi[:14]).all()
    assert np.allclose(rsi[14:], 100.0)


def test_rsi_and_macd_match_independent_recursion() -> None:
    closes = np.array(
        [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00],
        dtype="float64",
    )
    frame = _candles(closes.tolist())
    out = add_indicators(frame, ["rsi", "macd"], {"rsi": 14, "macd": {"fast": 3, "slow": 6, "signal": 3}})
    delta = np.diff(closes, prepend=np.nan)
    gain = np.where(np.isfinite(delta), np.maximum(delta, 0), np.nan)
    loss = np.where(np.isfinite(delta), np.maximum(-delta, 0), np.nan)
    avg_gain = gain[1:15].mean()
    avg_loss = loss[1:15].mean()
    for index in range(15, len(closes)):
        avg_gain = (avg_gain * 13 + gain[index]) / 14
        avg_loss = (avg_loss * 13 + loss[index]) / 14
    expected = 100 - 100 / (1 + avg_gain / avg_loss)
    assert abs(out["rsi_14"].iloc[-1] - expected) < 1e-9

    full = out["macd"].to_numpy()
    short = add_indicators(frame.iloc[:-1], ["macd"], {"macd": {"fast": 3, "slow": 6, "signal": 3}})
    assert np.allclose(full[:-1], short["macd"].to_numpy(), equal_nan=True)


def test_bollinger_and_atr() -> None:
    frame = _candles([10] * 25)
    out = add_indicators(frame, ["bb", "atr"], {"bb": {"length": 20, "std": 2}, "atr": 14})
    assert abs(out["bb_mid"].iloc[-1] - 10) < 1e-9
    assert abs(out["bb_upper"].iloc[-1] - 10) < 1e-9
    assert abs(out["bb_lower"].iloc[-1] - 10) < 1e-9
    # high-low is 1 on every bar, no gaps, so ATR settles at 1
    assert abs(out["atr_14"].iloc[-1] - 1) < 1e-9


def test_resample_and_open_bar() -> None:
    # Monday 2026-11-02 10:00 through the next Monday, one minute each hour start is enough
    # if we pass consecutive minutes covering two weekdays.
    monday = pd.Timestamp("2026-11-02 10:00:00", tz=MSK)
    stamps = [monday + pd.Timedelta(minutes=offset) for offset in range(5)]
    stamps += [monday + pd.Timedelta(days=7, minutes=offset) for offset in range(5)]
    close = np.arange(len(stamps), dtype="float64") + 1
    frame = pd.DataFrame(
        {
            "begin": stamps,
            "end": [stamp + pd.Timedelta(seconds=59) for stamp in stamps],
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.ones(len(stamps), dtype="int64"),
            "value": close,
        }
    )
    five = resample_candles(frame.iloc[:5], "5m")
    assert len(five) == 1
    assert five["open"].iloc[0] == 1
    assert five["close"].iloc[0] == 5
    assert five["volume"].iloc[0] == 5

    days = []
    for offset in range(10):
        begin = monday.normalize() + pd.Timedelta(days=offset, hours=10)
        days.append(begin)
    day_frame = pd.DataFrame(
        {
            "begin": days,
            "end": [stamp + pd.Timedelta(minutes=1) for stamp in days],
            "open": np.arange(len(days), dtype="float64"),
            "high": np.arange(len(days), dtype="float64"),
            "low": np.arange(len(days), dtype="float64"),
            "close": np.arange(len(days), dtype="float64"),
            "volume": np.ones(len(days), dtype="int64"),
            "value": np.ones(len(days)),
        }
    )
    weekly = resample_candles(day_frame, "1w")
    assert weekly["begin"].iloc[0] == monday.normalize()
    assert len(weekly) == 2

    now = pd.Timestamp("2026-11-02 10:03:00", tz=MSK)
    closed = drop_open_bar(five, "5m", now)
    assert closed.empty
    later = drop_open_bar(five, "5m", pd.Timestamp("2026-11-02 10:06:00", tz=MSK))
    assert len(later) == 1


def main() -> None:
    test_sma_and_ema_seed()
    test_rsi_all_gains_is_100()
    test_rsi_and_macd_match_independent_recursion()
    test_bollinger_and_atr()
    test_resample_and_open_bar()
    print("ok")


if __name__ == "__main__":
    main()
