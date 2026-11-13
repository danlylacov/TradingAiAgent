"""Чтение базы знаний по техническому анализу."""

from __future__ import annotations

import re
from typing import Any

from .articles import ARTICLES

_BY_ID = {article["id"]: article for article in ARTICLES}


def list_topics() -> list[dict[str, str]]:
    """Краткий каталог: id, заголовок и одно предложение."""
    return [
        {"id": article["id"], "title": article["title"], "summary": article["summary"]}
        for article in ARTICLES
    ]


def get_topic(topic_id: str) -> dict[str, Any]:
    """Полная статья: правило, как читать числа, ошибка и источник."""
    key = topic_id.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "sma": "moving_averages",
        "ema": "moving_averages",
        "averages": "moving_averages",
        "bollinger_bands": "bollinger",
        "bb": "bollinger",
        "atr": "atr_stop",
        "stop": "atr_stop",
        "levels": "trend_levels",
        "trend": "trend_levels",
        "score": "score_rubric",
        "rubric": "score_rubric",
        "structure": "market_structure",
        "market": "market_structure",
        "market_structure": "market_structure",
        "patterns": "chart_patterns",
        "pattern": "chart_patterns",
        "chart_pattern": "chart_patterns",
        "double_top": "chart_patterns",
        "head_shoulders": "chart_patterns",
    }
    key = aliases.get(key, key)
    article = _BY_ID.get(key)
    if article is None:
        known = ", ".join(_BY_ID)
        raise ValueError(f"Unknown topic {topic_id}. Use one of: {known}")
    return dict(article)


def search(query: str) -> list[dict[str, Any]]:
    """До пяти статей, в которых встречаются слова запроса."""
    tokens = [token for token in re.split(r"[^\wа-яА-ЯёЁ]+", query.lower()) if len(token) > 1]
    if not tokens:
        return []
    ranked: list[tuple[int, dict[str, Any]]] = []
    for article in ARTICLES:
        haystack = " ".join(
            [
                article["id"],
                article["title"],
                article["summary"],
                " ".join(article["tags"]),
                article["rule"],
                article["how_to_read"],
            ]
        ).lower()
        score = sum(haystack.count(token) for token in tokens)
        if score:
            ranked.append((score, article))
    ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
    return [dict(article) for _, article in ranked[:5]]
