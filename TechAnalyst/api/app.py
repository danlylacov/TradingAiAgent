"""Один запрос анализа возвращает zip: report.md и chart.png."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

from agent.loop import run_analysis
from data.indicators import CANDLE_SIZES

CandleSize = Enum("CandleSize", {f"s{size}": size for size in CANDLE_SIZES}, type=str)

app = FastAPI()


@app.get("/candles")
def candles() -> list[str]:
    return list(CANDLE_SIZES)


@app.post("/analyze")
def analyze(
    ticker: str,
    candle: CandleSize = CandleSize.s1d,
    until: date | None = Query(default=None, description="Смотреть данные до этой даты включительно"),
) -> Response:
    if not ticker.strip():
        raise HTTPException(status_code=400, detail="ticker is required")
    try:
        report = run_analysis(ticker, candle.value, None if until is None else until.isoformat())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    payload, filename = _archive(report)
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _archive(report: dict) -> tuple[bytes, str]:
    charts = report.get("charts") or []
    if not charts:
        raise HTTPException(status_code=500, detail="analysis finished without a chart")
    chart_path = Path(charts[0])
    if not chart_path.is_file():
        raise HTTPException(status_code=500, detail="chart file is missing")
    markdown = report.get("text") or ""
    if not markdown:
        md_path = chart_path.with_name("report.md")
        if not md_path.is_file():
            raise HTTPException(status_code=500, detail="markdown report is missing")
        markdown = md_path.read_text(encoding="utf-8")
    filename = f"{chart_path.parent.name}.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.md", markdown)
        archive.write(chart_path, "chart.png")
    return buffer.getvalue(), filename
