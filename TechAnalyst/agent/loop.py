"""Цикл tool calling DeepSeek и проверка отчёта."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from .charts import REPORTS
from .tools import SCHEMAS, bind_analysis, dispatch, reset_analysis

logger = logging.getLogger("agent")

_MAX_STEPS = 24
_SYSTEM = """Ты аналитик российского рынка акций. Отвечай по-русски. Отчёт строй по правилам технического аналитика.
Порядок работы:
1. Проверь тикер через get_universe.
2. Сними старший фон: get_structure и get_indicators по 1w. Потом средний фон: get_structure и get_indicators по 1d. window_high и window_low дневной структуры — диапазон периода.
3. Если выбранная свеча не 1d и не 1w, дополнительно прочитай её get_structure и последние get_candles с get_indicators. Вход решается по выбранной свече, фон — по неделе и дню.
4. Вызови get_topic score_rubric. Правила табеля важнее старых пяти слагаемых.
5. Числа sma20, sma50, sma200, rsi, macd, macd_signal, macd_hist, stoch_k, stoch_d, atr, bb_width и price копируй с последнего бара выбранной свечи. Если поля нет, пиши null и в табеле ставь этой строке нейтральный сигнал. Не выдумывай SMA200, сигнал MACD и неделю, которой не было в ответе. volume_vs_ma — отношение объёма последнего бара к volume_median_20, например 1.16 или 0.40, не проценты. macd_state: бычий только когда macd выше сигнала, иначе медвежий.
6. week52.low и week52.high равны window_low и window_high дневной структуры. Тип «экстремум окна» ставь только на эти две цены. R1 не может быть high последнего бара, если этот high не повторяет более ранний свинг.
7. Три поддержки ниже цены и три сопротивления выше. S1 и R1 — ближайшие. Если цена ниже SMA20, дневной тренд не называй восходящим вместе с согласованием «совпадает».
8. Фигуру ищи только в последних 60 барах и передай её points в JSON и в plot_analysis. Нет фигуры — name «нет ясной фигуры», trigger и target равны null. Два экстремума двойной вершины или двойного дна должны лежать в пределах 0.5 ATR, иначе фигуры нет. Пока пробоя нет, цель фигуры не ставь выше window_high и ниже window_low.
9. Стоп = середина зоны входа ∓ 1.5 ATR. Если согласование «конфликт» или веса равны, bias нейтральный, recommendation hold, score не выше 6. Если согласование «совпадает», бычий вес больше и неделя не нисходящая — только бычий и buy. Если согласование «совпадает», медвежий вес больше и неделя не восходящая — только медвежий и sell. В rationale называй ровно те веса, которые сложились в табеле, вторую сумму не выдумывай. Широту рынка не выдумывай.
10. Верни один JSON без текста вокруг и без переносов внутри строк:
{"recommendation":"buy|sell|hold","score":7,"bias":"бычий|нейтральный|медвежий","as_of":"дата","price":0,"week52":{"low":0,"high":0},"trend":{"weekly":"восходящий|нисходящий|боковик","weekly_alignment":"бычий|медвежий|смешанный","daily":"восходящий|нисходящий|боковик","alignment":"совпадает|конфликт","sma20":0,"sma50":0,"sma200":0,"cross":"текст"},"momentum":{"rsi":0,"rsi_state":"перекупленность|нейтрально|перепроданность","macd":0,"macd_signal":0,"macd_hist":0,"macd_state":"бычий|медвежий","stoch_k":0,"stoch_d":0,"stoch_state":"текст"},"volatility":{"bb_zone":"верх|середина|низ","bb_width":0,"atr":0,"volume_vs_ma":0,"volume_state":"выше среднего|ниже среднего","obv":"растёт|падает|плоско","obv_state":"накопление|распределение|нет сигнала"},"levels":{"resistances":[{"price":0,"kind":"тип"},{"price":0,"kind":"тип"},{"price":0,"kind":"тип"}],"supports":[{"price":0,"kind":"тип"},{"price":0,"kind":"тип"},{"price":0,"kind":"тип"}]},"pattern":{"name":"текст","text":"текст","trigger":null,"target":null,"points":[{"time":"дата","price":0}]},"scorecard":{"rows":[{"name":"цена/SMA20","value":"текст","signal":"бычий|медвежий|нейтральный","weight":3}]},"setup":{"entry_low":0,"entry_high":0,"stop":0,"target1":0,"target2":0,"horizon":"текст","action":"текст"},"breadth":"нет данных по широте рынка, старший фон — недельный тренд","entries":[{"price":0,"side":"buy|sell","reason":"..."}],"stop":0,"targets":[0,0],"rationale":"связный разбор всех разделов","charts":["путь"],"evidence":{}}
score — целое 1–10: явный перевес и совпадение недели с днём 7–10, конфликт не выше 6, перевес около нуля 4–5. recommendation совпадает с bias. В scorecard ровно 9 строк с весами 3,3,4,4,3,3,2,2,2. rationale — связный текст всех разделов, не короче абзаца. charts — путь из plot_analysis. evidence — числа последнего закрытого бара выбранной свечи."""


def run_analysis(ticker: str, timeframe: str = "1d", until: str | None = None) -> dict[str, Any]:
    """Ведёт диалог с моделью и записывает report.json и report.md.

    `until` — дата включительно. Инструменты не отдают бары и котировку позже неё.
    """
    _configure_logging()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    symbol = ticker.strip().upper()
    until_text = until.strip() if until else None
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    steps: list[dict[str, Any]] = []
    _emit(steps, {"event": "start", "ticker": symbol, "timeframe": timeframe, "until": until_text, "model": model})
    task = f"Сделай технический анализ {symbol} на интервале {timeframe}."
    if until_text:
        task += f" Смотри данные только до {until_text} включительно: инструменты уже обрезаны по этой дате."
    task += " В финале верни один json-отчёт."
    messages: list[Any] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": task},
    ]
    report: dict[str, Any] | None = None
    repair_used = False
    tokens = bind_analysis(timeframe, until_text)
    try:
        for _ in range(_MAX_STEPS):
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=SCHEMAS,
                extra_body={"thinking": {"type": "disabled"}},
            )
            message = completion.choices[0].message
            messages.append(message)
            if message.tool_calls:
                for call in message.tool_calls:
                    result = dispatch(call.function.name, call.function.arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result,
                        }
                    )
                    _emit(
                        steps,
                        {
                            "event": "tool",
                            "name": call.function.name,
                            "arguments": _arguments(call.function.arguments),
                            "result": _preview(result),
                        },
                    )
                continue
            try:
                report = _accept(message.content or "", symbol, timeframe, until_text)
                break
            except ValueError as exc:
                _emit(steps, {"event": "repair", "error": str(exc)})
                if repair_used:
                    raise
                repair_used = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"JSON не прошёл проверку: {exc}. "
                            "Верни один исправленный json без новых инструментов."
                        ),
                    }
                )
                repair = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                fixed = repair.choices[0].message
                messages.append(fixed)
                report = _accept(fixed.content or "", symbol, timeframe, until_text)
                break
    finally:
        reset_analysis(tokens)
    if report is None:
        raise RuntimeError("Model did not finish a report within 24 steps")
    folder_name = f"{symbol}_{timeframe}" + (f"_{until_text[:10]}" if until_text else "")
    folder = REPORTS / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    report["text"] = _markdown(symbol, timeframe, report)
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "report.md").write_text(report["text"], encoding="utf-8")
    report["report_dir"] = str(folder)
    _emit(
        steps,
        {
            "event": "report",
            "recommendation": report["recommendation"],
            "score": report["score"],
            "charts": report["charts"],
        },
    )
    report["logs"] = steps
    return report


def _configure_logging() -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _emit(steps: list[dict[str, Any]], event: dict[str, Any]) -> None:
    record = {"step": len(steps) + 1, **event}
    steps.append(record)
    logger.info("%s", json.dumps(record, ensure_ascii=False))


def _arguments(raw: str | None) -> Any:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return (raw or "")[:300]


def _preview(raw: str) -> Any:
    try:
        return _shrink(json.loads(raw))
    except json.JSONDecodeError:
        return raw[:400]


def _shrink(data: Any) -> Any:
    if isinstance(data, list):
        return {"count": len(data), "last": _shrink(data[-1]) if data else None}
    if isinstance(data, dict):
        shrunk: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 180:
                shrunk[key] = value[:180] + "…"
            elif isinstance(value, (dict, list)):
                shrunk[key] = _shrink(value)
            else:
                shrunk[key] = value
        return shrunk
    if isinstance(data, str) and len(data) > 180:
        return data[:180] + "…"
    return data


_TRENDS = {"восходящий", "нисходящий", "боковик"}
_ALIGNMENTS = {"бычий", "медвежий", "смешанный"}
_CONFLICTS = {"совпадает", "конфликт"}
_SIGNALS = {"бычий", "медвежий", "нейтральный"}
_BIASES = {"бычий": "buy", "нейтральный": "hold", "медвежий": "sell"}
_WEIGHTS = (3, 3, 4, 4, 3, 3, 2, 2, 2)


def validate_report(raw: str) -> dict[str, Any]:
    """Проверяет отчёт по разделам правил технического аналитика."""
    try:
        payload = json.loads(_extract_json(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("report must be an object")
    recommendation = str(payload.get("recommendation", "")).lower()
    if recommendation not in {"buy", "sell", "hold"}:
        raise ValueError("recommendation must be buy, sell or hold")
    bias = str(payload.get("bias", "")).strip().lower()
    if bias not in _BIASES:
        raise ValueError("bias must be бычий, нейтральный or медвежий")
    if _BIASES[bias] != recommendation:
        raise ValueError("recommendation must match bias")
    score = _whole(payload.get("score"), "score", 1, 10)
    price = _number(payload.get("price"), "price")
    trend = _section(payload, "trend")
    weekly = _choice(trend.get("weekly"), _TRENDS, "trend.weekly")
    daily = _choice(trend.get("daily"), _TRENDS, "trend.daily")
    if bias == "бычий" and weekly == "нисходящий":
        raise ValueError("bullish bias conflicts with the weekly downtrend")
    if bias == "медвежий" and weekly == "восходящий":
        raise ValueError("bearish bias conflicts with the weekly uptrend")
    momentum = _section(payload, "momentum")
    volatility = _section(payload, "volatility")
    levels = _section(payload, "levels")
    supports = _level_map(levels.get("supports"), price, below=True)
    resistances = _level_map(levels.get("resistances"), price, below=False)
    rows = _scorecard(payload.get("scorecard"))
    weighted_bull = sum(row["weight"] for row in rows if row["signal"] == "бычий")
    weighted_bear = sum(row["weight"] for row in rows if row["signal"] == "медвежий")
    alignment = _choice(trend.get("alignment"), _CONFLICTS, "trend.alignment")
    _check_bias(bias, score, weekly, alignment, weighted_bull, weighted_bear)
    sma20 = _optional_number(trend.get("sma20"))
    if sma20 is not None and price < sma20 and daily == "восходящий" and alignment == "совпадает":
        raise ValueError("price is below SMA20; an aligned daily uptrend is not allowed")
    pattern = _section(payload, "pattern")
    name = str(pattern.get("name", "")).strip()
    if not name:
        raise ValueError("pattern.name is required")
    setup = _section(payload, "setup")
    entry_low = _number(setup.get("entry_low"), "setup.entry_low")
    entry_high = _number(setup.get("entry_high"), "setup.entry_high")
    if entry_low > entry_high:
        entry_low, entry_high = entry_high, entry_low
    stop_value = _number(setup.get("stop"), "setup.stop")
    target1 = _number(setup.get("target1"), "setup.target1")
    target2 = _number(setup.get("target2"), "setup.target2")
    horizon = str(setup.get("horizon", "")).strip()
    action = str(setup.get("action", "")).strip()
    if not horizon or not action:
        raise ValueError("setup.horizon and setup.action are required")
    breadth = str(payload.get("breadth", "")).strip()
    if not breadth:
        raise ValueError("breadth is required")
    entries = []
    for item in payload.get("entries") or []:
        if not isinstance(item, dict):
            raise ValueError("each entry must be an object")
        side = str(item.get("side", "")).lower()
        if side not in {"buy", "sell"}:
            raise ValueError("entry side must be buy or sell")
        entries.append(
            {
                "price": _number(item.get("price"), "entry.price"),
                "side": side,
                "reason": str(item.get("reason", "")).strip(),
            }
        )
    if not entries:
        raise ValueError("entries must contain at least one price")
    rationale = str(payload.get("rationale", "")).strip()
    if len(rationale) < 200:
        raise ValueError("rationale is too short")
    _check_rationale_weights(rationale, weighted_bull, weighted_bear)
    macd_value = _optional_number(momentum.get("macd"))
    macd_signal = _optional_number(momentum.get("macd_signal"))
    macd_state = str(momentum.get("macd_state", "")).strip()
    _check_macd_state(macd_value, macd_signal, macd_state)
    pattern_points = _pattern_points(pattern.get("points"))
    pattern_target = _optional_number(pattern.get("target"))
    if "нет ясной" in name.lower():
        if pattern_target is not None or _optional_number(pattern.get("trigger")) is not None:
            raise ValueError("a missing pattern cannot have a trigger or a target")
    elif len(pattern_points) < 2:
        raise ValueError("a named pattern needs at least two points from real bars")
    charts = [str(path) for path in payload.get("charts") or []]
    if not charts or any(not Path(path).is_file() for path in charts):
        raise ValueError("charts must point to files created by plot_analysis")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("evidence must contain indicator numbers")
    week = payload.get("week52") if isinstance(payload.get("week52"), dict) else {}
    return {
        "recommendation": recommendation,
        "score": score,
        "bias": bias,
        "as_of": str(payload.get("as_of", "")).strip(),
        "price": price,
        "week52": {
            "low": _number(week.get("low"), "week52.low"),
            "high": _number(week.get("high"), "week52.high"),
        },
        "trend": {
            "weekly": weekly,
            "weekly_alignment": _choice(trend.get("weekly_alignment"), _ALIGNMENTS, "trend.weekly_alignment"),
            "daily": daily,
            "alignment": alignment,
            "sma20": sma20,
            "sma50": _optional_number(trend.get("sma50")),
            "sma200": _optional_number(trend.get("sma200")),
            "cross": str(trend.get("cross", "")).strip(),
        },
        "momentum": {
            "rsi": _optional_number(momentum.get("rsi")),
            "rsi_state": str(momentum.get("rsi_state", "")).strip(),
            "macd": macd_value,
            "macd_signal": macd_signal,
            "macd_hist": _optional_number(momentum.get("macd_hist")),
            "macd_state": macd_state,
            "stoch_k": _optional_number(momentum.get("stoch_k")),
            "stoch_d": _optional_number(momentum.get("stoch_d")),
            "stoch_state": str(momentum.get("stoch_state", "")).strip(),
        },
        "volatility": {
            "bb_zone": str(volatility.get("bb_zone", "")).strip(),
            "bb_width": _optional_number(volatility.get("bb_width")),
            "atr": _optional_number(volatility.get("atr")),
            "volume_vs_ma": _number(volatility.get("volume_vs_ma"), "volatility.volume_vs_ma"),
            "volume_state": str(volatility.get("volume_state", "")).strip(),
            "obv": str(volatility.get("obv", "")).strip(),
            "obv_state": str(volatility.get("obv_state", "")).strip(),
        },
        "levels": {"supports": supports, "resistances": resistances},
        "pattern": {
            "name": name,
            "text": str(pattern.get("text", "")).strip(),
            "trigger": _optional_number(pattern.get("trigger")),
            "target": pattern_target,
            "points": pattern_points,
        },
        "scorecard": {
            "bullish": sum(row["signal"] == "бычий" for row in rows),
            "bearish": sum(row["signal"] == "медвежий" for row in rows),
            "neutral": sum(row["signal"] == "нейтральный" for row in rows),
            "weighted_bull": weighted_bull,
            "weighted_bear": weighted_bear,
            "rows": rows,
        },
        "setup": {
            "entry_low": entry_low,
            "entry_high": entry_high,
            "stop": stop_value,
            "target1": target1,
            "target2": target2,
            "rr1": _reward_risk(recommendation, entry_low, entry_high, stop_value, target1),
            "rr2": _reward_risk(recommendation, entry_low, entry_high, stop_value, target2),
            "horizon": horizon,
            "action": action,
        },
        "breadth": breadth,
        "entries": entries,
        "stop": stop_value,
        "targets": [target1, target2],
        "rationale": rationale,
        "charts": charts,
        "evidence": evidence,
    }


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _choice(value: Any, allowed: set[str], label: str) -> str:
    text = str(value or "").strip().lower()
    if text not in allowed:
        raise ValueError(f"{label} must be one of: {', '.join(sorted(allowed))}")
    return text


def _whole(value: Any, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"{label} must be an integer from {low} to {high}")
    number = int(value)
    if not low <= number <= high:
        raise ValueError(f"{label} must be from {low} to {high}")
    return number


def _number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if number != number:
        raise ValueError(f"{label} must be a number")
    return number


def _optional_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return _number(value, "optional number")


def _accept(raw: str, ticker: str, timeframe: str, until: str | None) -> dict[str, Any]:
    return _audit_report(validate_report(raw), ticker, timeframe, until)


def _check_bias(bias: str, score: int, weekly: str, alignment: str, bull: int, bear: int) -> None:
    if alignment == "конфликт" or bull == bear:
        if bias != "нейтральный":
            raise ValueError("a timeframe conflict or an even tally requires hold")
        if score > 6:
            raise ValueError("score must be at most 6 when timeframes conflict")
        return
    if bull > bear and weekly != "нисходящий":
        if bias != "бычий":
            raise ValueError("aligned bullish lead requires buy")
        return
    if bear > bull and weekly != "восходящий":
        if bias != "медвежий":
            raise ValueError("aligned bearish lead requires sell")
        return
    if bias != "нейтральный":
        raise ValueError("the weekly trend opposes the indicator lead, so the bias is neutral")


def _check_macd_state(macd: float | None, signal: float | None, state: str) -> None:
    if macd is None or signal is None:
        return
    expected = "бычий" if macd > signal else "медвежий"
    text = state.lower()
    actual = "медвежий" if "медвеж" in text else "бычий" if "быч" in text else ""
    if actual != expected:
        raise ValueError(f"macd_state must be {expected}")


def _check_rationale_weights(rationale: str, bull: int, bear: int) -> None:
    mentioned = [int(item) for item in re.findall(r"весов\D{0,16}(\d+)", rationale)]
    mentioned += [int(item) for item in re.findall(r"(\d+)\s*/\s*26", rationale)]
    foreign = [item for item in mentioned if item not in {bull, bear, 26}]
    if foreign:
        raise ValueError(f"rationale quotes weights {foreign}, but the scorecard is {bull} against {bear}")


def _pattern_points(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if not isinstance(value, list):
        raise ValueError("pattern.points must be a list")
    points = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each pattern point must be an object")
        points.append({"time": str(item.get("time", "")), "price": _number(item.get("price"), "pattern.point")})
    return points


def _audit_report(report: dict[str, Any], ticker: str, timeframe: str, until: str | None) -> dict[str, Any]:
    from data.api import get_indicators

    from .structure import get_structure

    bar = get_indicators(ticker, timeframe, end=until, limit=1)[-1]
    _match_indicator(report["price"], bar.get("close"), "price")
    _match_indicator(report["trend"]["sma20"], bar.get("sma_20"), "sma20")
    _match_indicator(report["trend"]["sma50"], bar.get("sma_50"), "sma50")
    _match_indicator(report["trend"]["sma200"], bar.get("sma_200"), "sma200")
    _match_indicator(report["momentum"]["rsi"], bar.get("rsi_14"), "rsi")
    _match_indicator(report["momentum"]["macd"], bar.get("macd"), "macd")
    _match_indicator(report["momentum"]["macd_signal"], bar.get("macd_signal"), "macd_signal")
    _match_indicator(report["momentum"]["macd_hist"], bar.get("macd_hist"), "macd_hist")
    _match_indicator(report["momentum"]["stoch_k"], bar.get("stoch_k"), "stoch_k")
    _match_indicator(report["momentum"]["stoch_d"], bar.get("stoch_d"), "stoch_d")
    _match_indicator(report["volatility"]["atr"], bar.get("atr_14"), "atr")
    _match_indicator(report["volatility"]["bb_width"], bar.get("bb_width"), "bb_width")
    _missing_average_rows(report["scorecard"]["rows"], bar.get("sma_50"), bar.get("sma_200"))
    structure = _safe_structure(get_structure, ticker, "1d", until)
    if structure is not None:
        _match_indicator(report["week52"]["low"], structure["window_low"], "week52.low")
        _match_indicator(report["week52"]["high"], structure["window_high"], "week52.high")
        _check_window_extreme(report["levels"], structure["window_low"], structure["window_high"])
        _check_nearest_resistance(report["levels"]["resistances"][0]["price"], bar, structure)
        _check_pattern_target(report, structure["window_low"], structure["window_high"])
    chosen = structure if timeframe == "1d" else _safe_structure(get_structure, ticker, timeframe, until)
    if chosen is not None and chosen.get("volume_median_20"):
        ratio = float(chosen["last_volume"]) / float(chosen["volume_median_20"])
        _match_indicator(report["volatility"]["volume_vs_ma"], ratio, "volume_vs_ma")
    _check_double_extreme(report["pattern"], bar.get("atr_14"))
    return report


def _safe_structure(loader, ticker: str, timeframe: str, until: str | None) -> dict[str, Any] | None:
    try:
        return loader(ticker, timeframe, end=until)
    except ValueError:
        return None


def _match_indicator(got: float | None, expected: Any, label: str) -> None:
    if expected is None:
        if got is not None:
            raise ValueError(f"{label} is absent in the data; write null, not {got}")
        return
    if got is None:
        raise ValueError(f"{label} must be {expected}")
    expected_value = float(expected)
    tolerance = max(0.01, abs(expected_value) * 0.02) if abs(expected_value) < 1 else max(0.05, abs(expected_value) * 0.015)
    if abs(got - expected_value) > tolerance:
        raise ValueError(f"{label} {got} does not match the data value {expected_value}")


def _missing_average_rows(rows: list[dict[str, Any]], sma50: Any, sma200: Any) -> None:
    for row in rows:
        name = row["name"].lower()
        if sma200 is None and "200" in name and row["signal"] != "нейтральный":
            raise ValueError("SMA200 is missing; its scorecard row must be neutral")
        if (sma50 is None or sma200 is None) and "крест" in name and row["signal"] != "нейтральный":
            raise ValueError("the cross row must be neutral when SMA50 or SMA200 is missing")


def _check_window_extreme(levels: dict[str, Any], low: float, high: float) -> None:
    for level in levels["supports"] + levels["resistances"]:
        if "экстремум" in level["kind"].lower() and "окн" in level["kind"].lower():
            if not _near(level["price"], low) and not _near(level["price"], high):
                raise ValueError("экстремум окна must be the window high or the window low")


def _check_nearest_resistance(price: float, bar: dict[str, Any], structure: dict[str, Any]) -> None:
    if not _near(price, bar.get("high")):
        return
    earlier = [
        swing
        for swing in structure.get("swings") or []
        if swing.get("kind") == "high" and str(swing.get("begin")) < str(bar.get("begin"))
    ]
    if any(_near(swing["price"], price, 0.008) for swing in earlier):
        return
    raise ValueError("R1 is only the high of the last bar; use an earlier swing")


def _check_pattern_target(report: dict[str, Any], low: float, high: float) -> None:
    target = report["pattern"]["target"]
    if target is None:
        return
    levels = [level["price"] for level in report["levels"]["supports"] + report["levels"]["resistances"]]
    if _near(target, low) or _near(target, high) or any(_near(target, level) for level in levels):
        return
    if low <= target <= high:
        return
    raise ValueError("pattern target must lie inside the window or on a reported level")


def _check_double_extreme(pattern: dict[str, Any], atr: Any) -> None:
    if atr is None:
        return
    name = pattern["name"].lower()
    if "двойн" not in name:
        return
    prices = sorted(point["price"] for point in pattern.get("points") or [])
    if len(prices) < 2:
        return
    pair = prices[:2] if "дно" in name else prices[-2:]
    if abs(pair[0] - pair[1]) > 0.5 * float(atr):
        raise ValueError("the two extremes of a double top or bottom must lie within 0.5 ATR")


def _near(left: Any, right: Any, fraction: float = 0.004) -> bool:
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) <= max(0.02, abs(float(right)) * fraction)


def _level_map(value: Any, price: float, *, below: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("need exactly 3 supports and 3 resistances")
    levels = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each level must be an object")
        kind = str(item.get("kind") or item.get("type") or "").strip()
        if not kind:
            raise ValueError("each level needs a kind")
        levels.append({"price": _number(item.get("price"), "level.price"), "kind": kind})
    if below:
        if any(level["price"] >= price for level in levels):
            raise ValueError("supports must be below the current price")
        levels.sort(key=lambda level: -level["price"])
    else:
        if any(level["price"] <= price for level in levels):
            raise ValueError("resistances must be above the current price")
        levels.sort(key=lambda level: level["price"])
    return levels


def _scorecard(value: Any) -> list[dict[str, Any]]:
    rows_raw = value.get("rows") if isinstance(value, dict) else None
    if not isinstance(rows_raw, list) or len(rows_raw) != 9:
        raise ValueError("scorecard.rows must contain 9 signals")
    rows = []
    for item in rows_raw:
        if not isinstance(item, dict):
            raise ValueError("each scorecard row must be an object")
        weight = _whole(item.get("weight"), "scorecard.weight", 2, 4)
        rows.append(
            {
                "name": str(item.get("name", "")).strip(),
                "value": str(item.get("value", "")).strip(),
                "signal": _choice(item.get("signal"), _SIGNALS, "scorecard.signal"),
                "weight": weight,
            }
        )
    if tuple(sorted(row["weight"] for row in rows)) != tuple(sorted(_WEIGHTS)):
        raise ValueError("scorecard weights must be 3, 3, 4, 4, 3, 3, 2, 2, 2")
    return rows


def _reward_risk(recommendation: str, entry_low: float, entry_high: float, stop: float, target: float) -> float | None:
    entry = (entry_low + entry_high) / 2
    if recommendation == "sell":
        risk = stop - entry
        reward = entry - target
    else:
        risk = entry - stop
        reward = target - entry
    if risk <= 0:
        return None
    return round(reward / risk, 2)


def _extract_json(raw: str) -> str:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    raise ValueError("response has no JSON object")


def _markdown(ticker: str, timeframe: str, report: dict[str, Any]) -> str:
    trend = report["trend"]
    momentum = report["momentum"]
    volatility = report["volatility"]
    levels = report["levels"]
    pattern = report["pattern"]
    card = report["scorecard"]
    setup = report["setup"]
    price = report["price"]
    lines = [
        f"ТЕХНИЧЕСКИЙ АНАЛИЗ  —  {ticker} {timeframe}  ({report['as_of']})",
        f"Цена: {price}  |  Диапазон окна: {report['week52']['low']} – {report['week52']['high']}",
        "",
        "СТРУКТУРА ТРЕНДА",
        f"  Недельный тренд: {trend['weekly']}  (порядок средних: {trend['weekly_alignment']})",
        f"  Дневной тренд:   {trend['daily']}",
        f"  Согласование:    {trend['alignment']}",
        "  Средние:",
        f"    SMA 20:  {_shown(trend['sma20'])}   цена {_distance(price, trend['sma20'])}",
        f"    SMA 50:  {_shown(trend['sma50'])}   цена {_distance(price, trend['sma50'])}",
        f"    SMA 200: {_shown(trend['sma200'])}   цена {_distance(price, trend['sma200'])}",
        f"  Крест: {trend['cross']}",
        "",
        "ИМПУЛЬС",
        f"  RSI (14):     {_shown(momentum['rsi'])}  →  {momentum['rsi_state']}",
        (
            f"  MACD:         {_shown(momentum['macd'])}  сигнал: {_shown(momentum['macd_signal'])}  "
            f"гистограмма: {_shown(momentum['macd_hist'])}  →  {momentum['macd_state']}"
        ),
        f"  Стохастик %K/%D: {momentum['stoch_k']}/{momentum['stoch_d']}  →  {momentum['stoch_state']}",
        "",
        "ВОЛАТИЛЬНОСТЬ И ОБЪЁМ",
        f"  Полосы Боллинджера: {volatility['bb_zone']}  (ширина: {volatility['bb_width']})",
        f"  ATR (14):           {_shown(volatility['atr'])}  ({_atr_share(volatility['atr'], price)})",
        f"  Объём к средней 20: {volatility['volume_vs_ma']}× средней  →  {volatility['volume_state']}",
        f"  OBV:                {volatility['obv']}  →  {volatility['obv_state']}",
        "",
        "КАРТА УРОВНЕЙ",
    ]
    for index, level in enumerate(reversed(levels["resistances"]), start=1):
        number = 4 - index
        mark = "  ← ближайший потолок" if number == 1 else ""
        lines.append(f"  Сопротивление {number}: {level['price']}  [{level['kind']}]{mark}")
    lines.append(f"  ► ТЕКУЩАЯ ЦЕНА: {price} ◄")
    for index, level in enumerate(levels["supports"], start=1):
        mark = "  ← ближайший пол" if index == 1 else ""
        lines.append(f"  Поддержка {index}:     {level['price']}  [{level['kind']}]{mark}")
    lines.extend(
        [
            "",
            f"ФИГУРА: {pattern['name']}",
            f"  {pattern['text']}",
            f"  Пуск: {pattern['trigger'] if pattern['trigger'] is not None else 'нет'}",
            f"  Цель фигуры: {pattern['target'] if pattern['target'] is not None else 'нет'}",
            "",
            "ТАБЕЛЬ СИГНАЛОВ",
        ]
    )
    for row in card["rows"]:
        lines.append(f"  {row['name']}: {row['value']} → {row['signal']} (вес {row['weight']})")
    lines.extend(
        [
            f"  Бычьих: {card['bullish']}/9  |  Медвежьих: {card['bearish']}/9  |  Нейтральных: {card['neutral']}/9",
            f"  Взвешенно: бычьи {card['weighted_bull']}/26  против  медвежьих {card['weighted_bear']}/26",
            "",
            "СДЕЛКА",
            f"  Уклон:        {report['bias']}  ({report['recommendation']}, {report['score']}/10)",
            f"  Зона входа:   {setup['entry_low']} – {setup['entry_high']}",
            f"  Стоп:         {setup['stop']}  (1.5 ATR)",
            f"  Цель 1:       {setup['target1']}  (прибыль/риск {setup['rr1']})",
            f"  Цель 2:       {setup['target2']}  (прибыль/риск {setup['rr2']})",
            f"  Горизонт:     {setup['horizon']}",
            f"  Действие:     {setup['action']}",
            "",
            "КОНТЕКСТ",
            f"  {report['breadth']}",
            "",
            "РАЗБОР",
            report["rationale"],
            "",
            "ГРАФИК",
        ]
    )
    for chart in report["charts"]:
        lines.append(f"![анализ]({Path(chart).name})")
    lines.append("")
    return "\n".join(lines)


def _shown(value: Any) -> str:
    return "нет данных" if value is None else str(value)


def _distance(price: float, average: float | None) -> str:
    if average is None:
        return "нет в данных"
    if average == 0:
        return "без расстояния до средней"
    share = (price - average) / average * 100
    side = "выше" if share >= 0 else "ниже"
    return f"{side} на {abs(share):.2f}%"


def _atr_share(atr: float | None, price: float) -> str:
    if atr is None or price == 0:
        return "нет доли от цены"
    share = atr / price * 100
    if share < 1.5:
        regime = "низкая"
    elif share < 3:
        regime = "средняя"
    else:
        regime = "высокая"
    return f"{share:.2f}% цены, {regime} волатильность"
