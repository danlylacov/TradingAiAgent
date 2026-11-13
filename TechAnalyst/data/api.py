"""Функции чтения данных и индикаторов.

Их можно вызывать напрямую и позже обернуть в tool calling. Сеть нужна только
`sync_latest`, `get_quote` и `get_market_snapshot`, и только если локального
среза ещё нет. Свечи и индикаторы читаются из Parquet.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from .indicators import add_indicators, drop_open_bar, normalize_timeframe, resample_candles
from .moex import MSK, MoexError, MoexIss, moscow_now
from .store import load_quotes, load_universe, read_candles, save_quotes, save_universe, tickers, write_day

__all__ = [
    "get_candles",
    "get_indicators",
    "get_market_snapshot",
    "get_quote",
    "get_universe",
    "sync_latest",
]


def get_universe() -> dict[str, Any]:
    """Сохранённый топ-30 IMOEX. При первом вызове забирает состав с биржи."""
    payload = ensure_universe()
    return {
        "as_of": payload.get("as_of"),
        "securities": [
            {
                "ticker": item["ticker"],
                "name": item["name"],
                "weight": item["weight"],
            }
            for item in payload["securities"]
        ],
    }


def get_candles(
    ticker: str,
    timeframe: str = "1d",
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Свечи из локальной истории. `limit` оставляет последние бары.

    Таймфреймы: 1m, 5m, 15m, 1h, 1d, 1w. Старшие собираются из минуток.
    Последний бар может быть ещё не закрыт.
    """
    symbol = _require_ticker(ticker)
    timeframe = normalize_timeframe(timeframe)
    start_ts = _parse_bound(start, end_of_day=False)
    end_ts = _parse_bound(end, end_of_day=True)
    frame = read_candles(
        symbol,
        None if start_ts is None else start_ts.date(),
        None if end_ts is None else end_ts.date(),
    )
    if frame.empty:
        raise ValueError(f"No local candles for {symbol}. Run python -m data.backfill")
    bars = resample_candles(frame, timeframe)
    bars = _clip(bars, start_ts, end_ts)
    bars = _tail(bars, limit)
    return _records(bars)


def get_indicators(
    ticker: str,
    timeframe: str = "1d",
    indicators: list[str] | None = None,
    limit: int | None = 200,
    start: str | None = None,
    end: str | None = None,
    params: dict | None = None,
) -> list[dict[str, Any]]:
    """Закрытые бары с индикаторами.

    По умолчанию: SMA 20/50/200, EMA 12/26, RSI 14, MACD 12/26/9,
    Bollinger 20±2 и ширина полос, ATR 14, стохастик %K/%D и OBV. `params` меняет периоды, например
    `{"rsi": 7, "sma": [10, 20], "macd": {"fast": 8, "slow": 21, "signal": 5}}`.
    Незакрытый последний бар в расчёт не входит.
    """
    symbol = _require_ticker(ticker)
    timeframe = normalize_timeframe(timeframe)
    start_ts = _parse_bound(start, end_of_day=False)
    end_ts = _parse_bound(end, end_of_day=True)
    frame = read_candles(symbol, None, None if end_ts is None else end_ts.date())
    if frame.empty:
        raise ValueError(f"No local candles for {symbol}. Run python -m data.backfill")
    bars = resample_candles(frame, timeframe)
    bars = _clip(bars, None, end_ts)
    bars = drop_open_bar(bars, timeframe, moscow_now())
    enriched = add_indicators(bars, indicators, params)
    enriched = _clip(enriched, start_ts, None)
    enriched = _tail(enriched, limit)
    return _records(enriched)


def get_quote(ticker: str) -> dict[str, Any]:
    """Последний сохранённый срез бумаги. Если среза нет, один запрос в ISS."""
    symbol = _require_ticker(ticker)
    payload = load_quotes()
    quote = payload.get("quotes", {}).get(symbol)
    fetched_at = payload.get("fetched_at")
    if quote is None:
        fresh = MoexIss().marketdata([symbol])
        saved = _merge_quotes(fresh)
        quote = saved["quotes"].get(symbol)
        fetched_at = saved["fetched_at"]
    if quote is None:
        raise ValueError(f"No quote for {symbol}")
    return {**quote, "fetched_at": fetched_at}


def get_market_snapshot() -> list[dict[str, Any]]:
    """Срез всех бумаг вселенной в порядке веса. Без файла котировок ходит в ISS."""
    universe = ensure_universe()
    payload = load_quotes()
    quotes = payload.get("quotes") or {}
    fetched_at = payload.get("fetched_at")
    wanted = [item["ticker"] for item in universe["securities"]]
    if any(ticker not in quotes for ticker in wanted):
        fresh = MoexIss().marketdata(wanted)
        saved = _merge_quotes(fresh)
        quotes = saved["quotes"]
        fetched_at = saved["fetched_at"]
    rows = []
    for item in universe["securities"]:
        quote = quotes.get(item["ticker"])
        if quote is None:
            continue
        rows.append(
            {
                **quote,
                "name": item["name"],
                "weight": item["weight"],
                "fetched_at": fetched_at,
            }
        )
    return rows


def sync_latest(client: MoexIss | None = None) -> dict[str, Any]:
    """Один проход: котировки всех бумаг и минутные свечи текущего дня."""
    client = client or MoexIss()
    universe = ensure_universe(client)
    names = [item["ticker"] for item in universe["securities"]]
    errors: list[str] = []
    quote_count = 0
    try:
        quote_count = len(save_quotes(client.marketdata(names))["quotes"])
    except MoexError as exc:
        errors.append(f"quotes: {exc}")
    today = moscow_now().date()
    candles: dict[str, int] = {}
    for ticker in names:
        try:
            rows = client.iter_candles(ticker, today, today)
            rows = [row for row in rows if row["begin"].date() == today]
            if not rows:
                candles[ticker] = 0
                continue
            candles[ticker] = write_day(ticker, today, pd.DataFrame(rows), overwrite=True)
        except MoexError as exc:
            errors.append(f"{ticker}: {exc}")
            candles[ticker] = 0
    return {
        "fetched_at": moscow_now().isoformat(),
        "quotes": quote_count,
        "candles": candles,
        "errors": errors,
    }


def ensure_universe(client: MoexIss | None = None) -> dict[str, Any]:
    current = load_universe()
    if current and current.get("securities"):
        return current
    payload = (client or MoexIss()).top_imoex(30)
    if len(payload["securities"]) < 30:
        raise MoexError(f"IMOEX returned {len(payload['securities'])} securities, expected 30")
    save_universe(payload)
    return payload


def _require_ticker(ticker: str) -> str:
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker is empty")
    known = tickers()
    if not known:
        ensure_universe()
        known = tickers()
    if symbol not in known:
        raise ValueError(f"Unknown ticker {symbol}")
    return symbol


def _parse_bound(value: str | None, *, end_of_day: bool) -> pd.Timestamp | None:
    if value is None:
        return None
    text = value.strip()
    stamp = pd.Timestamp(text)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(MSK)
    else:
        stamp = stamp.tz_convert(MSK)
    date_only = len(text) <= 10
    if date_only and end_of_day:
        stamp = stamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return stamp


def _clip(
    frame: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= frame["begin"] >= start
    if end is not None:
        mask &= frame["begin"] <= end
    return frame.loc[mask].reset_index(drop=True)


def _tail(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return frame
    if limit < 0:
        raise ValueError("limit must be >= 0")
    if limit == 0 or frame.empty:
        return frame.iloc[0:0].reset_index(drop=True)
    return frame.tail(limit).reset_index(drop=True)


def _merge_quotes(fresh: list[dict[str, Any]]) -> dict[str, Any]:
    current = load_quotes().get("quotes") or {}
    merged = dict(current)
    for item in fresh:
        if item.get("ticker"):
            merged[item["ticker"]] = item
    return save_quotes(list(merged.values()))


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = list(frame.columns)
    records = []
    for row in frame.itertuples(index=False, name=None):
        records.append({column: _scalar(value) for column, value in zip(columns, row)})
    return records


def _scalar(value: Any) -> Any:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return round(number, 6)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value
