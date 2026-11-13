"""Загрузка минутных свечей за последний год с продолжением после обрыва."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any

import pandas as pd

from .api import ensure_universe
from .moex import MoexError, MoexIss, moscow_now
from .store import read_candles, stored_dates, write_day, write_status

log = logging.getLogger(__name__)
CHUNK_DAYS = 7


def backfill(
    tickers: list[str] | None = None,
    days: int = 365,
    workers: int = 4,
    client: MoexIss | None = None,
) -> dict[str, Any]:
    """Скачать или докачать минутные свечи. Уже закрытые дни не перекачиваются."""
    if days < 1:
        raise ValueError("days must be positive")
    client = client or MoexIss()
    universe = ensure_universe(client)
    selected = [item.upper() for item in tickers] if tickers else [item["ticker"] for item in universe["securities"]]
    known = {item["ticker"] for item in universe["securities"]}
    unknown = [ticker for ticker in selected if ticker not in known]
    if unknown:
        raise ValueError(f"Unknown tickers: {', '.join(unknown)}")
    if workers < 1:
        raise ValueError("workers must be positive")
    if len(selected) == 1 or workers == 1:
        return {ticker: _one(client, ticker, days) for ticker in selected}
    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as pool:
        futures = {pool.submit(_one, client, ticker, days): ticker for ticker in selected}
        for future in as_completed(futures):
            ticker = futures[future]
            results[ticker] = future.result()
    return results


def _one(client: MoexIss, ticker: str, days: int) -> dict[str, Any]:
    today = moscow_now().date()
    end = today
    start = end - timedelta(days=days)
    existing = stored_dates(ticker)
    if existing:
        last = existing[-1]
        start = today if last >= today else max(start, last + timedelta(days=1))
    write_status(ticker, {"status": "running", "from": start.isoformat(), "to": end.isoformat()})
    try:
        written_rows = 0
        written_days = 0
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), end)
            rows = client.iter_candles(ticker, cursor, chunk_end)
            days_saved, rows_saved = _store_chunk(ticker, rows, cursor, chunk_end, today)
            written_days += days_saved
            written_rows += rows_saved
            log.info(
                "%s %s..%s fetched=%s saved_days=%s saved_rows=%s",
                ticker,
                cursor.isoformat(),
                chunk_end.isoformat(),
                len(rows),
                days_saved,
                rows_saved,
            )
            cursor = chunk_end + timedelta(days=1)
        summary = _summary(ticker)
        summary["saved_rows"] = written_rows
        summary["saved_days"] = written_days
        write_status(ticker, {"status": "done", **summary})
        log.info("%s done rows=%s days=%s %s..%s", ticker, summary["rows"], summary["days"], summary["first"], summary["last"])
        return summary
    except (MoexError, OSError, ValueError) as exc:
        log.exception("backfill failed for %s", ticker)
        write_status(ticker, {"status": "error", "error": str(exc)})
        return {"ticker": ticker, "error": str(exc)}


def _store_chunk(
    ticker: str,
    rows: list[dict[str, Any]],
    start: date,
    end: date,
    today: date,
) -> tuple[int, int]:
    if not rows:
        return 0, 0
    frame = pd.DataFrame(rows)
    frame = frame[(frame["begin"].dt.date >= start) & (frame["begin"].dt.date <= end)]
    saved_days = 0
    saved_rows = 0
    for day, part in frame.groupby(frame["begin"].dt.date, sort=True):
        count = write_day(ticker, day, part, overwrite=day == today)
        if count:
            saved_days += 1
            saved_rows += count
    return saved_days, saved_rows


def _summary(ticker: str) -> dict[str, Any]:
    frame = read_candles(ticker)
    if frame.empty:
        return {"ticker": ticker, "rows": 0, "days": 0, "first": None, "last": None}
    return {
        "ticker": ticker,
        "rows": int(len(frame)),
        "days": int(frame["begin"].dt.date.nunique()),
        "first": frame["begin"].iloc[0].isoformat(),
        "last": frame["begin"].iloc[-1].isoformat(),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download one year of 1-minute MOEX candles")
    parser.add_argument("--ticker", action="append", help="Limit the download to these tickers")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    backfill(args.ticker, days=args.days, workers=args.workers)


if __name__ == "__main__":
    main()
