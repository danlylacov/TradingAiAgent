"""Свечные таймфреймы и индикаторы без заглядывания вперёд.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .moex import MSK

FAMILIES = ("sma", "ema", "rsi", "macd", "bb", "atr", "stoch", "obv")

_RULES = {
    "1m": None,
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "1d": "1D",
    "1w": "W-MON",
}
CANDLE_SIZES = tuple(_RULES)
_ALIASES = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "60m": "1h",
    "1hour": "1h",
    "d": "1d",
    "day": "1d",
    "1day": "1d",
    "w": "1w",
    "week": "1w",
    "1week": "1w",
}
_OFFSETS = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(days=7),
}
_COLUMNS = ["begin", "end", "open", "high", "low", "close", "volume", "value"]


def normalize_timeframe(value: str) -> str:
    key = value.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _RULES:
        known = ", ".join(_RULES)
        raise ValueError(f"Unknown timeframe {value}. Use one of: {known}")
    return key


def resample_candles(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Собрать старший таймфрейм из минуток. Пустые интервалы не заполняются."""
    timeframe = normalize_timeframe(timeframe)
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values("begin").drop_duplicates("begin", keep="last")
    if timeframe == "1m":
        return ordered.reset_index(drop=True)
    indexed = ordered.set_index("begin")
    aggregated = indexed.resample(_RULES[timeframe], label="left", closed="left").agg(
        {
            "end": "max",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "value": "sum",
        }
    )
    aggregated = aggregated.dropna(subset=["open"]).reset_index()
    aggregated["volume"] = aggregated["volume"].round().astype("int64")
    aggregated["value"] = aggregated["value"].astype("float64").round(2)
    return aggregated.loc[:, _COLUMNS]


def drop_open_bar(frame: pd.DataFrame, timeframe: str, now: pd.Timestamp) -> pd.DataFrame:
    """Убрать последний бар, если его интервал ещё не закрылся."""
    if frame.empty:
        return frame
    timeframe = normalize_timeframe(timeframe)
    if now.tzinfo is None:
        now = now.tz_localize(MSK)
    else:
        now = now.tz_convert(MSK)
    last_begin = frame["begin"].iloc[-1]
    if last_begin + _OFFSETS[timeframe] > now:
        return frame.iloc[:-1].reset_index(drop=True)
    return frame.reset_index(drop=True)


def add_indicators(
    frame: pd.DataFrame,
    indicators: list[str] | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    """Добавить колонки индикаторов. Считаются только по уже переданным барам."""
    selected = [name.strip().lower() for name in (indicators or FAMILIES)]
    unknown = [name for name in selected if name not in FAMILIES]
    if unknown:
        raise ValueError(f"Unknown indicators: {', '.join(unknown)}")
    params = params or {}
    out = frame.copy()
    if out.empty:
        return out
    close = out["close"].astype("float64")
    if "sma" in selected:
        for period in _periods(params.get("sma"), (20, 50, 200)):
            out[f"sma_{period}"] = close.rolling(period, min_periods=period).mean()
    if "ema" in selected:
        for period in _periods(params.get("ema"), (12, 26)):
            out[f"ema_{period}"] = _ema(close.to_numpy(), period)
    if "rsi" in selected:
        period = _one(params.get("rsi"), 14)
        out[f"rsi_{period}"] = _rsi(close.to_numpy(), period)
    if "macd" in selected:
        fast, slow, signal = _macd_params(params.get("macd"))
        line, signal_line, hist = _macd(close.to_numpy(), fast, slow, signal)
        out["macd"] = line
        out["macd_signal"] = signal_line
        out["macd_hist"] = hist
    if "bb" in selected:
        length, std = _bb_params(params.get("bb"))
        middle = close.rolling(length, min_periods=length).mean()
        deviation = close.rolling(length, min_periods=length).std(ddof=0)
        out["bb_mid"] = middle
        out["bb_upper"] = middle + std * deviation
        out["bb_lower"] = middle - std * deviation
        out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / middle
    if "atr" in selected:
        period = _one(params.get("atr"), 14)
        out[f"atr_{period}"] = _atr(out, period)
    if "stoch" in selected:
        period = _one(params.get("stoch"), 14)
        out["stoch_k"], out["stoch_d"] = _stochastic(out, period)
    if "obv" in selected:
        out["obv"] = _obv(close.to_numpy(), out["volume"].to_numpy(dtype="float64"))
    return out


def _periods(value: object, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        periods = default
    elif isinstance(value, int):
        periods = (value,)
    else:
        periods = tuple(int(item) for item in value)  # type: ignore[arg-type]
    _check_periods(periods)
    return periods


def _one(value: object, default: int) -> int:
    period = default if value is None else int(value)  # type: ignore[arg-type]
    _check_periods((period,))
    return period


def _check_periods(periods: tuple[int, ...]) -> None:
    if not periods or any(period < 1 for period in periods):
        raise ValueError("Indicator periods must be positive integers")


def _macd_params(value: object) -> tuple[int, int, int]:
    if value is None:
        parsed = (12, 26, 9)
    elif isinstance(value, dict):
        parsed = (int(value["fast"]), int(value["slow"]), int(value["signal"]))
    else:
        fast, slow, signal = value  # type: ignore[misc]
        parsed = (int(fast), int(slow), int(signal))
    _check_periods(parsed)
    if parsed[0] >= parsed[1]:
        raise ValueError("MACD fast period must be shorter than slow period")
    return parsed


def _bb_params(value: object) -> tuple[int, float]:
    if value is None:
        length, std = 20, 2.0
    elif isinstance(value, dict):
        length, std = int(value.get("length", 20)), float(value.get("std", 2.0))
    else:
        length, std = value  # type: ignore[misc]
        length, std = int(length), float(std)
    _check_periods((length,))
    if std <= 0:
        raise ValueError("Bollinger deviation must be positive")
    return length, std


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    start = _first_finite(values)
    if start is None or len(values) - start < span:
        return out
    alpha = 2.0 / (span + 1.0)
    seed_at = start + span - 1
    out[seed_at] = np.mean(values[start : seed_at + 1])
    for index in range(seed_at + 1, len(values)):
        if not np.isfinite(values[index]) or not np.isfinite(out[index - 1]):
            continue
        out[index] = alpha * values[index] + (1.0 - alpha) * out[index - 1]
    return out


def _wilder(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    run = 0
    seed_at = None
    for index, value in enumerate(values):
        run = run + 1 if np.isfinite(value) else 0
        if run == period:
            seed_at = index
            break
    if seed_at is None:
        return out
    window = values[seed_at - period + 1 : seed_at + 1]
    out[seed_at] = np.mean(window)
    for index in range(seed_at + 1, len(values)):
        if not np.isfinite(values[index]) or not np.isfinite(out[index - 1]):
            continue
        out[index] = (out[index - 1] * (period - 1) + values[index]) / period
    return out


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(np.isfinite(delta), np.maximum(delta, 0.0), np.nan)
    loss = np.where(np.isfinite(delta), np.maximum(-delta, 0.0), np.nan)
    avg_gain = _wilder(gain, period)
    avg_loss = _wilder(loss, period)
    rsi = np.full(close.shape, np.nan)
    ready = np.isfinite(avg_gain) & np.isfinite(avg_loss)
    flat = ready & (avg_loss == 0) & (avg_gain == 0)
    rsi[flat] = np.nan
    up_only = ready & (avg_loss == 0) & (avg_gain > 0)
    rsi[up_only] = 100.0
    normal = ready & (avg_loss != 0)
    rsi[normal] = 100.0 - (100.0 / (1.0 + avg_gain[normal] / avg_loss[normal]))
    return rsi


def _macd(close: np.ndarray, fast: int, slow: int, signal: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(line, signal)
    hist = line - signal_line
    return line, signal_line, hist


def _atr(frame: pd.DataFrame, period: int) -> np.ndarray:
    high = frame["high"].to_numpy(dtype="float64")
    low = frame["low"].to_numpy(dtype="float64")
    close = frame["close"].to_numpy(dtype="float64")
    previous = np.roll(close, 1)
    previous[0] = np.nan
    true_range = np.maximum(high - low, np.maximum(np.abs(high - previous), np.abs(low - previous)))
    true_range[0] = np.nan
    return _wilder(true_range, period)


def _stochastic(frame: pd.DataFrame, period: int, smooth: int = 3) -> tuple[pd.Series, pd.Series]:
    low = frame["low"].astype("float64")
    high = frame["high"].astype("float64")
    close = frame["close"].astype("float64")
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    span = (highest - lowest).replace(0, np.nan)
    percent_k = (close - lowest) / span * 100.0
    percent_d = percent_k.rolling(smooth, min_periods=smooth).mean()
    return percent_k, percent_d


def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    delta = np.diff(close, prepend=np.nan)
    signed = np.where(delta > 0, volume, np.where(delta < 0, -volume, 0.0))
    signed = np.where(np.isfinite(delta), signed, 0.0)
    return np.cumsum(signed)


def _first_finite(values: np.ndarray) -> int | None:
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) == 0:
        return None
    return int(finite[0])
