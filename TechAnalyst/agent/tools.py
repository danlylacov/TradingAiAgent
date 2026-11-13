"""Инструменты агента: данные, база знаний и график."""

from __future__ import annotations

import json
from contextvars import ContextVar, Token
from typing import Any, Callable

from data.api import get_candles, get_indicators, get_quote, get_universe
from knowledge.api import get_topic, search

from .charts import plot_analysis
from .structure import get_structure

_MAX_BARS = 400
_UNTIL: ContextVar[str | None] = ContextVar("analysis_until", default=None)
_TIMEFRAME: ContextVar[str] = ContextVar("analysis_timeframe", default="1d")

SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_universe",
            "description": "Список доступных тикеров голубых фишек и их вес.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_candles",
            "description": "Последние свечи OHLCV. Ответ ограничен 400 барами.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "timeframe": {"type": "string", "description": "1m, 5m, 15m, 1h, 1d или 1w"},
                    "limit": {"type": "integer"},
                },
                "required": ["ticker", "timeframe"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_indicators",
            "description": "Закрытые бары со SMA, EMA, RSI, MACD, полосами Боллинджера, ATR, стохастиком и OBV. Не больше 400 баров.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "timeframe": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["ticker", "timeframe"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Последний сохранённый рыночный срез бумаги.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Поиск статей по техническому анализу.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_topic",
            "description": "Полная статья: правило, чтение чисел, ошибка и источник.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_id": {
                        "type": "string",
                        "description": "trend_levels, market_structure, chart_patterns, moving_averages, rsi, macd, bollinger, atr_stop, volume, score_rubric",
                    }
                },
                "required": ["topic_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_structure",
            "description": "Сжатый обзор до 1500 закрытых баров: свинги, зоны поддержки и сопротивления, медианы объёма.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "timeframe": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["ticker", "timeframe"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_analysis",
            "description": "Строит PNG цены, индикаторов, поддержки, сопротивления и фигур. Уровни и точки бери из данных.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "timeframe": {"type": "string"},
                    "limit": {"type": "integer"},
                    "entry": {"type": "number"},
                    "stop": {"type": "number"},
                    "targets": {"type": "array", "items": {"type": "number"}},
                    "supports": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"price": {"type": "number"}, "label": {"type": "string"}},
                            "required": ["price"],
                        },
                    },
                    "resistances": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"price": {"type": "number"}, "label": {"type": "string"}},
                            "required": ["price"],
                        },
                    },
                    "patterns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "label": {"type": "string"},
                                "points": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {"time": {"type": "string"}, "price": {"type": "number"}},
                                        "required": ["time", "price"],
                                    },
                                },
                            },
                            "required": ["name", "points"],
                        },
                    },
                },
                "required": ["ticker", "timeframe"],
                "additionalProperties": False,
            },
        },
    },
]

def bind_analysis(timeframe: str, until: str | None) -> tuple[Token[str], Token[str | None]]:
    """Фиксирует размер свечи и дату, дальше которой инструменты не читают."""
    return _TIMEFRAME.set(timeframe), _UNTIL.set(until)


def reset_analysis(tokens: tuple[Token[str], Token[str | None]]) -> None:
    _TIMEFRAME.reset(tokens[0])
    _UNTIL.reset(tokens[1])


_HANDLERS: dict[str, Callable[..., Any]] = {
    "get_universe": lambda **_: get_universe(),
    "get_candles": lambda **args: get_candles(
        args["ticker"],
        timeframe=args.get("timeframe", "1d"),
        limit=_limit(args.get("limit")),
        end=_UNTIL.get(),
    ),
    "get_indicators": lambda **args: get_indicators(
        args["ticker"],
        timeframe=args.get("timeframe", "1d"),
        limit=_limit(args.get("limit")),
        end=_UNTIL.get(),
    ),
    "get_quote": lambda **args: _quote(args["ticker"]),
    "search_knowledge": lambda **args: search(args["query"]),
    "get_topic": lambda **args: get_topic(args["topic_id"]),
    "get_structure": lambda **args: get_structure(
        args["ticker"],
        timeframe=args.get("timeframe", "1d"),
        limit=args.get("limit") or 1500,
        end=_UNTIL.get(),
    ),
    "plot_analysis": lambda **args: plot_analysis(
        args["ticker"],
        timeframe=args.get("timeframe", "1d"),
        limit=_limit(args.get("limit")),
        entry=args.get("entry"),
        stop=args.get("stop"),
        targets=args.get("targets"),
        supports=args.get("supports"),
        resistances=args.get("resistances"),
        patterns=args.get("patterns"),
        end=_UNTIL.get(),
    ),
}


def _quote(ticker: str) -> dict[str, Any]:
    until = _UNTIL.get()
    if not until:
        return get_quote(ticker)
    bars = get_indicators(ticker, timeframe=_TIMEFRAME.get() or "1d", end=until, limit=1)
    if not bars:
        raise ValueError(f"No closed bars for {ticker} on or before {until}")
    last = bars[-1]
    return {
        "ticker": ticker.strip().upper(),
        "last": last["close"],
        "begin": last["begin"],
        "volume": last["volume"],
        "until": until,
    }


def dispatch(name: str, arguments: dict[str, Any] | str) -> str:
    """Исполняет инструмент и возвращает компактный JSON."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Bad arguments: {exc}"}, ensure_ascii=False)
    handler = _HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool {name}"}, ensure_ascii=False)
    try:
        payload = _compact(handler(**arguments))
    except Exception as exc:
        payload = {"error": str(exc)}
    return json.dumps(payload, ensure_ascii=False)


def _limit(value: Any) -> int:
    if value is None:
        return _MAX_BARS
    return min(max(int(value), 1), _MAX_BARS)


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value
