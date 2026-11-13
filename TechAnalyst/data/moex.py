"""Публичный клиент MOEX ISS.

Онлайн-котировки без подписки могут отставать до 15 минут. Минутные свечи
отдаются без токена. Время свечей в ответе — московское, без часового пояса.
"""

from __future__ import annotations

import threading
import time
from datetime import date
from typing import Any

import pandas as pd
import requests

ISS_BASE = "https://iss.moex.com/iss"
MSK = "Europe/Moscow"
PAGE_SIZE = 500
USER_AGENT = "TradingAgent/1.0"


class MoexError(RuntimeError):
    """ISS не ответил или вернул неожиданное тело."""


def moscow_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=MSK)


def parse_moscow(value: str | None) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize(MSK)
    return stamp.tz_convert(MSK)


class MoexIss:
    """HTTP-клиент ISS с ограничением частоты и повторами."""

    def __init__(self, min_interval: float = 0.08, timeout: float = 30.0) -> None:
        self.min_interval = min_interval
        self.timeout = timeout
        self._next_at = 0.0
        self._pace = threading.Lock()
        self._local = threading.local()

    def top_imoex(self, limit: int = 30) -> dict[str, Any]:
        """Топ бумаг индекса МосБиржи по весу на последнюю дату состава."""
        rows: list[dict[str, Any]] = []
        start = 0
        as_of: str | None = None
        while True:
            payload = self._get(
                "/statistics/engines/stock/markets/index/analytics/IMOEX.json",
                {
                    "iss.meta": "off",
                    "iss.only": "analytics",
                    "analytics.columns": "ticker,shortnames,weight,tradedate",
                    "start": start,
                },
            )
            block = _block(payload, "analytics")
            page = block["data"]
            if not page:
                break
            columns = block["columns"]
            for raw in page:
                item = dict(zip(columns, raw))
                if as_of is None and item.get("tradedate"):
                    as_of = str(item["tradedate"])
                rows.append(
                    {
                        "ticker": str(item["ticker"]),
                        "name": str(item.get("shortnames") or item["ticker"]),
                        "weight": float(item["weight"]),
                    }
                )
            start += len(page)
            if len(page) < 20:
                break
        rows.sort(key=lambda item: (-item["weight"], item["ticker"]))
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in rows:
            if item["ticker"] in seen:
                continue
            seen.add(item["ticker"])
            deduped.append(item)
        return {"as_of": as_of, "securities": deduped[:limit]}

    def iter_candles(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: int = 1,
    ) -> list[dict[str, Any]]:
        """Все свечи диапазона. Страницы по 500, пока ответ не короче страницы."""
        rows: list[dict[str, Any]] = []
        offset = 0
        last_begin: pd.Timestamp | None = None
        while True:
            page = self._candle_page(ticker, start, end, offset, interval)
            if not page:
                break
            first = page[0]["begin"]
            if last_begin is not None and first <= last_begin:
                raise MoexError(
                    f"ISS candle page for {ticker} did not advance at offset {offset}"
                )
            rows.extend(page)
            last_begin = page[-1]["begin"]
            if len(page) < PAGE_SIZE:
                break
            offset += len(page)
        return rows

    def marketdata(self, tickers: list[str]) -> list[dict[str, Any]]:
        """Текущий срез TQBR по списку тикеров одним запросом."""
        if not tickers:
            return []
        payload = self._get(
            "/engines/stock/markets/shares/boards/TQBR/securities.json",
            {
                "iss.meta": "off",
                "iss.only": "marketdata",
                "securities": ",".join(tickers),
            },
        )
        block = _block(payload, "marketdata")
        columns = block["columns"]
        quotes = []
        for raw in block["data"]:
            row = dict(zip(columns, raw))
            quotes.append(_quote(row))
        return quotes

    def _candle_page(
        self,
        ticker: str,
        start: date,
        end: date,
        offset: int,
        interval: int,
    ) -> list[dict[str, Any]]:
        payload = self._get(
            f"/engines/stock/markets/shares/securities/{ticker}/candles.json",
            {
                "iss.meta": "off",
                "from": start.isoformat(),
                "till": end.isoformat(),
                "interval": interval,
                "start": offset,
            },
        )
        return _candles(payload)

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers["User-Agent"] = USER_AGENT
            self._local.session = session
        return session

    def _acquire(self) -> None:
        with self._pace:
            now = time.monotonic()
            if now < self._next_at:
                time.sleep(self._next_at - now)
            self._next_at = time.monotonic() + self.min_interval

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = ISS_BASE + path
        last_error: Exception | None = None
        for attempt in range(5):
            self._acquire()
            try:
                response = self._session().get(url, params=params, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    raise MoexError(f"ISS {response.status_code} for {path}")
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError, MoexError) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 16))
                continue
            if isinstance(payload, dict) and payload.get("error"):
                last_error = MoexError(str(payload["error"]))
                time.sleep(min(2**attempt, 16))
                continue
            return payload
        raise MoexError(f"ISS request failed: {path}") from last_error


def _block(payload: dict[str, Any], name: str) -> dict[str, Any]:
    block = payload.get(name)
    if not isinstance(block, dict) or "columns" not in block or "data" not in block:
        raise MoexError(f"ISS response has no {name} block")
    return block


def _candles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    block = _block(payload, "candles")
    columns = block["columns"]
    rows = []
    for raw in block["data"]:
        item = dict(zip(columns, raw))
        begin = parse_moscow(item.get("begin"))
        end = parse_moscow(item.get("end"))
        if begin is None or end is None:
            continue
        rows.append(
            {
                "begin": begin,
                "end": end,
                "open": float(item["open"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "close": float(item["close"]),
                "volume": int(round(float(item["volume"]))),
                "value": float(item["value"]),
            }
        )
    return rows


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _quote(row: dict[str, Any]) -> dict[str, Any]:
    systime = row.get("SYSTIME")
    parsed = parse_moscow(systime) if isinstance(systime, str) else None
    return {
        "ticker": row.get("SECID"),
        "board": row.get("BOARDID"),
        "last": _num(row.get("LAST")),
        "open": _num(row.get("OPEN")),
        "high": _num(row.get("HIGH")),
        "low": _num(row.get("LOW")),
        "bid": _num(row.get("BID")),
        "offer": _num(row.get("OFFER")),
        "change": _num(row.get("CHANGE")),
        "change_pct": _num(row.get("LASTCHANGEPRCNT")),
        "volume": None if row.get("VOLTODAY") is None else int(row["VOLTODAY"]),
        "value": _num(row.get("VALTODAY_RUR")),
        "num_trades": None if row.get("NUMTRADES") is None else int(row["NUMTRADES"]),
        "waprice": _num(row.get("WAPRICE")),
        "time": row.get("TIME"),
        "systime": None if parsed is None else parsed.isoformat(),
        "trading_status": row.get("TRADINGSTATUS"),
    }
