"""Запуск: python -m agent --ticker SBER --timeframe 1d"""

from __future__ import annotations

import argparse
import json

from .loop import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Технический анализ через DeepSeek")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--until", default=None, help="Дата включительно, YYYY-MM-DD")
    args = parser.parse_args()
    report = run_analysis(args.ticker, args.timeframe, args.until)
    print(json.dumps({"recommendation": report["recommendation"], "score": report["score"], "report_dir": report["report_dir"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
