"""График анализа: цена, объём, RSI и MACD."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import pandas as pd

from data.api import get_indicators

REPORTS = Path(__file__).resolve().parent / "reports"

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_analysis(
    ticker: str,
    timeframe: str = "1d",
    limit: int = 400,
    entry: float | None = None,
    stop: float | None = None,
    targets: list[float] | None = None,
    supports: list[dict[str, Any]] | None = None,
    resistances: list[dict[str, Any]] | None = None,
    patterns: list[dict[str, Any]] | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Рисует закрытые бары и сохраняет PNG рядом с будущим отчётом."""
    symbol = ticker.strip().upper()
    as_of = end
    bars = get_indicators(symbol, timeframe=timeframe, limit=min(max(int(limit), 30), 400), end=as_of)
    frame = pd.DataFrame(bars)
    if frame.empty:
        raise ValueError(f"No closed bars to plot for {symbol}")
    frame["begin"] = pd.to_datetime(frame["begin"])
    targets = [float(level) for level in (targets or [])]
    entry_value = None if entry is None else float(entry)
    stop_value = None if stop is None else float(stop)

    fig, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(14.5, 10.6),
        gridspec_kw={"height_ratios": [3.5, 1.05, 0.95, 1.1]},
    )
    fig.patch.set_facecolor("white")
    price, volume, rsi, macd = axes
    for axis in axes:
        axis.set_facecolor("#fcfcfc")
        axis.grid(True, color="#e2e2e2", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8.5, colors="#333333")
        axis.yaxis.label.set_size(9)
        axis.yaxis.label.set_color("#333333")

    dates = frame["begin"]
    price.plot(dates, frame["close"], color="#1f4e79", linewidth=1.7, label="цена", zorder=3)
    for column, color, name in (
        ("sma_20", "#e07a3d", "SMA 20"),
        ("sma_50", "#2a9d8f", "SMA 50"),
        ("sma_200", "#6c757d", "SMA 200"),
    ):
        _plot_numeric(price, dates, frame, column, color=color, linewidth=1.15, label=name)
    _bands(price, dates, frame)

    marks: list[tuple[float, str, str]] = []
    _trade_lines(price, entry_value, stop_value, targets, marks)
    drawn_supports = _zones(price, supports, "#1b7f4e", (0, (4, 2.2)), "S", marks)
    drawn_resistances = _zones(price, resistances, "#9b2335", (0, (4, 2.2)), "R", marks)
    drawn_patterns = _patterns(price, patterns)
    _fit_price(price, [mark[0] for mark in marks])
    side_labels = _merge_marks(marks)

    start, last = dates.iloc[0], dates.iloc[-1]
    price.set_xlim(start, last)
    title = f"{symbol}  {timeframe}" + (f"  до {as_of[:10]}" if as_of else "")
    fig.suptitle(title, fontsize=15, color="#1a1a1a", x=0.43, ha="center")
    price.set_ylabel("цена")
    price.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: _fmt_price(value)))
    price.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=6,
        fontsize=8.5,
        frameon=False,
        borderaxespad=0,
        handlelength=1.8,
        columnspacing=1.1,
    )

    colors = [
        "#2a9d8f" if close >= open_ else "#c44536"
        for close, open_ in zip(frame["close"], frame["open"], strict=True)
    ]
    span = (dates.iloc[-1] - dates.iloc[0]).total_seconds() / max(len(dates) - 1, 1)
    bar_width = max(span * 0.62 / 86400, 0.002)
    volume.bar(dates, frame["volume"], width=bar_width, color=colors)
    peak = float(pd.to_numeric(frame["volume"], errors="coerce").max() or 0)
    volume.set_ylim(0, peak * 1.18 if peak else 1)
    volume.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_volume))
    volume.yaxis.get_offset_text().set_visible(False)
    volume.set_ylabel("объём")

    _plot_numeric(rsi, dates, frame, "rsi_14", color="#1f4e79", linewidth=1.15, legend=False)
    rsi.axhline(70, color="#c44536", linewidth=0.7, linestyle="--")
    rsi.axhline(30, color="#2a9d8f", linewidth=0.7, linestyle="--")
    rsi.set_ylim(0, 100)
    rsi.set_ylabel("RSI")

    _plot_numeric(macd, dates, frame, "macd", color="#1f4e79", linewidth=1.15, label="MACD")
    _plot_numeric(macd, dates, frame, "macd_signal", color="#e07a3d", linewidth=1.15, label="сигнал")
    if "macd_hist" in frame:
        hist = pd.to_numeric(frame["macd_hist"], errors="coerce").fillna(0)
        hist_colors = ["#2a9d8f" if value >= 0 else "#c44536" for value in hist]
        macd.bar(dates, hist, width=bar_width, color=hist_colors, alpha=0.55, zorder=1)
    macd.axhline(0, color="#888888", linewidth=0.6)
    macd.set_ylabel("MACD")
    macd.legend(loc="upper left", fontsize=8, frameon=False)
    span_days = (dates.iloc[-1] - dates.iloc[0]).days
    if span_days > 120:
        macd.xaxis.set_major_locator(mdates.MonthLocator())
        macd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=7)
        macd.xaxis.set_major_locator(locator)
        macd.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    macd.tick_params(axis="x", rotation=0, labelsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.76, top=0.88, bottom=0.06, hspace=0.14)
    _place_side_labels(price, side_labels)

    folder = REPORTS / (f"{symbol}_{timeframe}" + (f"_{as_of[:10]}" if as_of else ""))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "chart.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    last_bar = frame.iloc[-1]
    return {
        "path": str(path),
        "ticker": symbol,
        "timeframe": timeframe,
        "bars": int(len(frame)),
        "last_begin": str(last_bar["begin"]),
        "last_close": float(last_bar["close"]),
        "entry": entry_value,
        "stop": stop_value,
        "targets": targets,
        "supports": drawn_supports,
        "resistances": drawn_resistances,
        "patterns": drawn_patterns,
    }


def _plot_numeric(
    axis,
    dates,
    frame: pd.DataFrame,
    column: str,
    *,
    color: str,
    linewidth: float,
    legend: bool = True,
    label: str | None = None,
) -> None:
    if column not in frame:
        return
    series = pd.to_numeric(frame[column], errors="coerce")
    if not series.notna().any():
        return
    axis.plot(dates, series, color=color, linewidth=linewidth, label=(label or column) if legend else None, zorder=2)


def _bands(axis, dates, frame: pd.DataFrame) -> None:
    if not {"bb_upper", "bb_lower"}.issubset(frame.columns):
        return
    upper = pd.to_numeric(frame["bb_upper"], errors="coerce")
    lower = pd.to_numeric(frame["bb_lower"], errors="coerce")
    if upper.notna().any():
        axis.plot(dates, upper, color="#9b8ec4", linewidth=0.8, label="полосы")
    if lower.notna().any():
        axis.plot(dates, lower, color="#9b8ec4", linewidth=0.8, label="_nolegend_")
    if upper.notna().any() and lower.notna().any():
        axis.fill_between(dates, lower, upper, color="#9b8ec4", alpha=0.13, zorder=0)


def _trade_lines(axis, entry: float | None, stop: float | None, targets: list[float], marks: list) -> None:
    if entry is not None:
        axis.axhline(entry, color="#1b7f4e", linewidth=1.15, zorder=2)
        marks.append((entry, "вход", "#1b7f4e"))
    if stop is not None:
        axis.axhline(stop, color="#c44536", linewidth=1.15, linestyle="--", zorder=2)
        marks.append((stop, "стоп", "#c44536"))
    for index, level in enumerate(targets, start=1):
        axis.axhline(level, color="#1d4e89", linewidth=1.0, linestyle=(0, (1.2, 1.4)), zorder=2)
        marks.append((float(level), f"цель {index}", "#1d4e89"))


def _zones(axis, levels: list[dict[str, Any]] | None, color: str, linestyle, prefix: str, marks: list) -> list[dict[str, Any]]:
    drawn = []
    for index, item in enumerate(levels or [], start=1):
        if not isinstance(item, dict) or item.get("price") is None:
            continue
        price = float(item["price"])
        label = str(item.get("label") or "").strip()
        axis.axhline(price, color=color, linewidth=0.95, linestyle=linestyle, alpha=0.95, zorder=1)
        marks.append((price, _tag(label, f"{prefix}{index}"), color))
        drawn.append({"price": price, "label": label})
    return drawn


def _patterns(axis, patterns: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    drawn = []
    drawn_names: list[str] = []
    for pattern in patterns or []:
        if not isinstance(pattern, dict):
            continue
        points = []
        for point in pattern.get("points") or []:
            if not isinstance(point, dict) or point.get("time") is None or point.get("price") is None:
                continue
            when = pd.to_datetime(point["time"], errors="coerce")
            if pd.isna(when):
                continue
            points.append({"time": str(point["time"]), "price": float(point["price"]), "when": when})
        if len(points) < 2:
            continue
        axis.plot(
            [point["when"] for point in points],
            [point["price"] for point in points],
            color="#6d4c41",
            linewidth=1.25,
            marker="o",
            markersize=4,
            zorder=4,
        )
        name = str(pattern.get("name") or pattern.get("label") or "").strip()
        if len(name) > 32:
            name = name[:31].rstrip() + "…"
        if name:
            drawn_names.append(name)
        drawn.append(
            {
                "name": str(pattern.get("name") or ""),
                "label": str(pattern.get("label") or ""),
                "points": [{"time": point["time"], "price": point["price"]} for point in points],
            }
        )
    if drawn_names:
        axis.text(
            0.012,
            0.975,
            "\n".join(drawn_names),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#4e342e",
            bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#d7ccc8", "alpha": 0.94},
            zorder=5,
        )
    return drawn


def _fit_price(axis, levels: list[float]) -> None:
    low, high = axis.get_ylim()
    span = high - low or 1
    nearby = [level for level in levels if low - span * 0.2 <= level <= high + span * 0.2]
    if nearby:
        low = min(low, min(nearby))
        high = max(high, max(nearby))
        span = high - low or 1
    axis.set_ylim(low - span * 0.07, high + span * 0.1)


def _merge_marks(marks: list[tuple[float, str, str]]) -> list[tuple[float, str, str]]:
    groups: list[dict[str, Any]] = []
    for price, role, color in marks:
        for group in groups:
            if abs(group["price"] - price) <= max(0.08, abs(price) * 0.0018):
                group["roles"].append(role)
                group["colors"].append(color)
                break
        else:
            groups.append({"price": price, "roles": [role], "colors": [color]})
    merged = []
    for group in groups:
        color = next((item for item in group["colors"] if item in {"#9b2335", "#1b7f4e"}), group["colors"][0])
        roles = sorted(dict.fromkeys(group["roles"]), key=_role_order)
        merged.append((group["price"], f"{' · '.join(roles)}  {_fmt_price(group['price'])}", color))
    return merged


def _role_order(role: str) -> int:
    if re.fullmatch(r"[RS]\d{1,2}", role):
        return 0
    if role == "вход":
        return 1
    if role == "стоп":
        return 2
    return 3


def _place_side_labels(axis, marks: list[tuple[float, str, str]]) -> None:
    low, high = axis.get_ylim()
    visible = [mark for mark in marks if low <= mark[0] <= high]
    if not visible:
        return
    figure = axis.figure
    figure.canvas.draw()
    pixels = [axis.transData.transform((0, mark[0]))[1] for mark in visible]
    order = sorted(range(len(visible)), key=lambda index: pixels[index])
    gap = 8.5 * figure.dpi / 72 * 1.45
    placed = pixels[:]
    for previous, index in zip(order, order[1:], strict=False):
        if placed[index] < placed[previous] + gap:
            placed[index] = placed[previous] + gap
    ceiling = axis.transData.transform((0, high))[1] - 3
    floor = axis.transData.transform((0, low))[1] + 3
    if placed[order[-1]] > ceiling:
        placed = [value - (placed[order[-1]] - ceiling) for value in placed]
    if placed[order[0]] < floor:
        room = max(ceiling - floor, 1)
        step = min(gap, room / max(len(visible) - 1, 1))
        for rank, index in enumerate(order):
            placed[index] = floor + rank * step
    inverse = axis.transData.inverted()
    transform = blended_transform_factory(axis.transAxes, axis.transData)
    for (level, text, color), pixel in zip(visible, placed, strict=True):
        y = inverse.transform((0, pixel))[1]
        level_pixel = axis.transData.transform((0, level))[1]
        if abs(pixel - level_pixel) > 4:
            axis.plot(
                [1.0, 1.012],
                [level, y],
                transform=transform,
                color=color,
                linewidth=0.6,
                clip_on=False,
                solid_capstyle="round",
                zorder=4,
            )
        axis.text(
            1.02,
            y,
            text,
            transform=transform,
            color=color,
            fontsize=8.5,
            va="center",
            ha="left",
            clip_on=False,
            zorder=5,
        )


def _tag(label: str, fallback: str) -> str:
    head = re.split(r"[\s:–—\-]", label.strip(), maxsplit=1)[0]
    if re.fullmatch(r"[RSrs]\d{1,2}", head):
        return head.upper()
    return fallback


def _fmt_price(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1000:
        digits = 0 if abs(value - round(value)) < 0.05 else 1
        return f"{value:,.{digits}f}".replace(",", " ")
    return f"{value:.2f}"


def _fmt_volume(value: float, _pos: object) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f} млрд"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f} млн"
    if absolute >= 1_000:
        return f"{value / 1_000:.0f} тыс"
    return f"{value:.0f}"
