"""Сжатый обзор длинной истории: свинги, зоны и объём."""

from __future__ import annotations

import statistics
from typing import Any

from data.api import get_indicators

_LOOKBACK = 1500
_WING = 3
_ZONE_ATR = 0.5


def get_structure(
    ticker: str,
    timeframe: str = "1d",
    limit: int = _LOOKBACK,
    end: str | None = None,
) -> dict[str, Any]:
    """Читает до 1500 закрытых баров и возвращает уровни, а не сами свечи."""
    size = min(max(int(limit or _LOOKBACK), 30), _LOOKBACK)
    bars = get_indicators(ticker, timeframe=timeframe, limit=size, end=end)
    symbol = ticker.strip().upper()
    if len(bars) < _WING * 2 + 1:
        raise ValueError(f"Not enough closed bars to build structure for {symbol}")
    atr = _median(bar.get("atr_14") for bar in bars)
    last = bars[-1]
    volumes = [float(bar["volume"]) for bar in bars if bar.get("volume") is not None]
    median_all = statistics.median(volumes) if volumes else None
    median_20 = statistics.median(volumes[-20:]) if volumes else None
    last_volume = float(last["volume"]) if last.get("volume") is not None else None
    highs, lows = _swings(bars)
    return {
        "ticker": symbol,
        "timeframe": timeframe,
        "bars": len(bars),
        "begin": bars[0]["begin"],
        "end": last["begin"],
        "last_close": float(last["close"]),
        "window_high": max(float(bar["high"]) for bar in bars),
        "window_low": min(float(bar["low"]) for bar in bars),
        "atr_14": None if atr is None else round(atr, 6),
        "volume_median_20": None if median_20 is None else round(median_20, 2),
        "volume_median_all": None if median_all is None else round(median_all, 2),
        "last_volume": last_volume,
        "last_bar_short_session": bool(
            median_20 and last_volume is not None and last_volume < median_20 / 3
        ),
        "swings": _recent(highs, 20) + _recent(lows, 20),
        "supports": _zones(lows, atr),
        "resistances": _zones(highs, atr),
    }


def _swings(bars: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    highs: list[dict[str, Any]] = []
    lows: list[dict[str, Any]] = []
    for index in range(_WING, len(bars) - _WING):
        window = bars[index - _WING : index + _WING + 1]
        bar = bars[index]
        neighbors = [item for item in window if item is not bar]
        if bar["high"] > max(item["high"] for item in neighbors):
            highs.append({"kind": "high", "begin": bar["begin"], "price": float(bar["high"])})
        if bar["low"] < min(item["low"] for item in neighbors):
            lows.append({"kind": "low", "begin": bar["begin"], "price": float(bar["low"])})
    return highs, lows


def _recent(swings: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return sorted(swings, key=lambda item: item["begin"])[-count:]


def _zones(swings: list[dict[str, Any]], atr: float | None) -> list[dict[str, Any]]:
    if not swings or atr is None or atr <= 0:
        return []
    ordered = sorted(swings, key=lambda item: item["price"])
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
    for swing in ordered[1:]:
        center = sum(item["price"] for item in groups[-1]) / len(groups[-1])
        if abs(swing["price"] - center) <= _ZONE_ATR * atr:
            groups[-1].append(swing)
        else:
            groups.append([swing])
    zones = []
    for group in groups:
        if len(group) < 2:
            continue
        dates = [item["begin"] for item in sorted(group, key=lambda item: item["begin"])]
        zones.append(
            {
                "price": round(sum(item["price"] for item in group) / len(group), 4),
                "touches": len(group),
                "dates": dates,
            }
        )
    zones.sort(key=lambda item: (-item["touches"], item["dates"][-1]))
    return zones[:12]


def _median(values) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return statistics.median(numbers)
