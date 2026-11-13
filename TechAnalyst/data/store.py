"""Свечи, состав индекса и котировки в локальной MongoDB.

База `techanalyst`. Адрес берётся из MONGO_URI, по умолчанию mongodb://127.0.0.1:27017.
Минутка хранится документом
с уникальным ключом (ticker, begin). Закрытый день не перезаписывается,
текущий день заменяется целиком.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError

from .moex import MSK, moscow_now

ROOT = Path(__file__).resolve().parent
CANDLES_ROOT = ROOT / "candles" / "1m"
UNIVERSE_PATH = ROOT / "universe.json"
QUOTES_PATH = ROOT / "quotes.json"
STATUS_PATH = ROOT / "backfill_status.json"

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017")
DB_NAME = "techanalyst"
CANDLE_COLUMNS = ["begin", "end", "open", "high", "low", "close", "volume", "value"]
_MSK = ZoneInfo(MSK)

_lock = threading.Lock()
_client: MongoClient | None = None


def database() -> pymongo.database.Database:
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000, tz_aware=True, tzinfo=timezone.utc)
        _client.admin.command("ping")
        db = _client[DB_NAME]
        db.candles.create_index([("ticker", 1), ("begin", 1)], unique=True)
        db.candles.create_index([("ticker", 1), ("day", 1)])
    return _client[DB_NAME]


def save_universe(payload: dict[str, Any]) -> None:
    database().universe.replace_one(
        {"_id": "imoex"},
        {"_id": "imoex", "as_of": payload.get("as_of"), "securities": payload.get("securities", [])},
        upsert=True,
    )


def load_universe() -> dict[str, Any] | None:
    document = database().universe.find_one({"_id": "imoex"})
    if not document:
        return None
    return {"as_of": document.get("as_of"), "securities": document.get("securities", [])}


def tickers() -> list[str]:
    payload = load_universe()
    if not payload:
        return []
    return [item["ticker"] for item in payload.get("securities", [])]


def save_quotes(quotes: list[dict[str, Any]], fetched_at: str | None = None) -> dict[str, Any]:
    payload = {
        "fetched_at": fetched_at or moscow_now().isoformat(),
        "quotes": {item["ticker"]: item for item in quotes if item.get("ticker")},
    }
    with _lock:
        database().quotes.replace_one({"_id": "latest"}, {"_id": "latest", **payload}, upsert=True)
    return payload


def load_quotes() -> dict[str, Any]:
    document = database().quotes.find_one({"_id": "latest"})
    if not document:
        return {"fetched_at": None, "quotes": {}}
    return {"fetched_at": document.get("fetched_at"), "quotes": document.get("quotes") or {}}


def write_status(ticker: str, status: dict[str, Any]) -> None:
    with _lock:
        database().backfill.replace_one(
            {"_id": ticker},
            {"_id": ticker, **status, "updated_at": moscow_now().isoformat()},
            upsert=True,
        )


def stored_dates(ticker: str) -> list[date]:
    days = database().candles.distinct("day", {"ticker": ticker})
    return sorted(date.fromisoformat(day) for day in days)


def write_day(ticker: str, day: date, frame: pd.DataFrame, *, overwrite: bool) -> int:
    """Записать свечи одного дня. Возвращает число вставленных документов."""
    candles = database().candles
    day_key = day.isoformat()
    with _lock:
        if not overwrite and candles.count_documents({"ticker": ticker, "day": day_key}, limit=1):
            return 0
        documents = _documents(ticker, frame, day)
        if overwrite:
            candles.delete_many({"ticker": ticker, "day": day_key})
        if not documents:
            return 0
        _insert_many(candles, documents)
        return len(documents)


def read_candles(
    ticker: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    query: dict[str, Any] = {"ticker": ticker}
    bounds: dict[str, datetime] = {}
    if start is not None:
        bounds["$gte"] = datetime(start.year, start.month, start.day, tzinfo=_MSK)
    if end is not None:
        next_day = end + timedelta(days=1)
        bounds["$lt"] = datetime(next_day.year, next_day.month, next_day.day, tzinfo=_MSK)
    if bounds:
        query["begin"] = bounds
    cursor = database().candles.find(query, {"_id": 0, "ticker": 0, "day": 0}).sort("begin", 1)
    rows = list(cursor)
    if not rows:
        return _empty()
    return _normalize(pd.DataFrame(rows))


def import_parquet() -> dict[str, int]:
    """Перенести уже скачанные Parquet и JSON в MongoDB. Повторный запуск дописывает только новое."""
    _import_json()
    candles = database().candles
    inserted = 0
    tickers_loaded = 0
    if not CANDLES_ROOT.exists():
        return {"tickers": 0, "inserted": 0}
    for folder in sorted(path for path in CANDLES_ROOT.iterdir() if path.is_dir()):
        batch: list[dict[str, Any]] = []
        files = 0
        for path in sorted(folder.glob("*.parquet")):
            frame = pd.read_parquet(path)
            batch.extend(_documents(folder.name, frame))
            files += 1
            if len(batch) >= 10_000:
                inserted += _insert_many(candles, batch)
                batch = []
        if batch:
            inserted += _insert_many(candles, batch)
        tickers_loaded += 1
        print(f"{folder.name} files={files} inserted_total={inserted}", flush=True)
    return {"tickers": tickers_loaded, "inserted": inserted}


def _import_json() -> None:
    if UNIVERSE_PATH.exists():
        save_universe(json.loads(UNIVERSE_PATH.read_text(encoding="utf-8")))
    if QUOTES_PATH.exists():
        payload = json.loads(QUOTES_PATH.read_text(encoding="utf-8"))
        quotes = list((payload.get("quotes") or {}).values())
        if quotes:
            save_quotes(quotes, payload.get("fetched_at"))
    if STATUS_PATH.exists():
        payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        for ticker, status in (payload.get("tickers") or {}).items():
            write_status(ticker, status)


def _documents(ticker: str, frame: pd.DataFrame, only_day: date | None = None) -> list[dict[str, Any]]:
    normalized = _normalize(frame)
    if only_day is not None and not normalized.empty:
        normalized = normalized.loc[normalized["begin"].dt.date == only_day]
    documents = []
    for row in normalized.itertuples(index=False):
        begin = row.begin.to_pydatetime()
        documents.append(
            {
                "ticker": ticker,
                "day": begin.strftime("%Y-%m-%d"),
                "begin": begin,
                "end": row.end.to_pydatetime(),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": int(row.volume),
                "value": float(row.value),
            }
        )
    return documents


def _insert_many(candles: Collection, documents: list[dict[str, Any]]) -> int:
    try:
        result = candles.insert_many(documents, ordered=False)
    except BulkWriteError as exc:
        errors = exc.details.get("writeErrors", [])
        if any(error.get("code") != 11000 for error in errors):
            raise
        return int(exc.details.get("nInserted", 0))
    return len(result.inserted_ids)


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty()
    out = frame.loc[:, CANDLE_COLUMNS].copy()
    out["begin"] = _as_moscow(out["begin"])
    out["end"] = _as_moscow(out["end"])
    for column in ("open", "high", "low", "close"):
        out[column] = out[column].astype("float64").round(6)
    out["volume"] = out["volume"].round().astype("int64")
    out["value"] = out["value"].astype("float64").round(2)
    out = out.dropna(subset=["begin", "open", "high", "low", "close"])
    out = out.drop_duplicates(subset=["begin"], keep="last")
    out = out.sort_values("begin").reset_index(drop=True)
    return out


def _as_moscow(values: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(values, utc=False)
    if getattr(stamps.dt, "tz", None) is None:
        return stamps.dt.tz_localize(MSK)
    return stamps.dt.tz_convert(MSK)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "begin": pd.Series(dtype=f"datetime64[ns, {MSK}]"),
            "end": pd.Series(dtype=f"datetime64[ns, {MSK}]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="int64"),
            "value": pd.Series(dtype="float64"),
        }
    )
