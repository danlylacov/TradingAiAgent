"""Живое обновление котировок и минуток текущего дня.

Запуск из каталога TechAnalyst:

    python -m data.realtime
"""

from __future__ import annotations

import argparse
import logging
import time

from .api import sync_latest
from .moex import MoexIss

log = logging.getLogger(__name__)


def run(interval: float = 20.0, client: MoexIss | None = None) -> None:
    """Крутить `sync_latest`, пока процесс не остановят."""
    interval = max(float(interval), 5.0)
    client = client or MoexIss()
    while True:
        started = time.monotonic()
        try:
            result = sync_latest(client)
            log.info(
                "synced quotes=%s candle_tickers=%s errors=%s",
                result["quotes"],
                sum(1 for count in result["candles"].values() if count),
                len(result["errors"]),
            )
            for error in result["errors"]:
                log.warning("sync error: %s", error)
        except Exception:
            log.exception("realtime sync failed")
        time.sleep(max(0.0, interval - (time.monotonic() - started)))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Poll MOEX quotes and today's minute candles")
    parser.add_argument("--interval", type=float, default=20.0, help="Seconds between passes")
    args = parser.parse_args()
    run(args.interval)


if __name__ == "__main__":
    main()
