"""Загрузка локальных Parquet в MongoDB.

Запуск из каталога TechAnalyst:

    python -m data.load_mongo
"""

from __future__ import annotations

from .store import database, import_parquet


def main() -> None:
    summary = import_parquet()
    count = database().candles.count_documents({})
    print(f"imported tickers={summary['tickers']} new_documents={summary['inserted']} candles={count}", flush=True)


if __name__ == "__main__":
    main()
