"""Layout partitioning into typed elements (title, key/value, list item, narrative).

Uses ``unstructured.partition.text`` when the optional ``heavy`` extra is installed
and enabled; otherwise a deterministic rule-based partitioner that covers the
same element categories the downstream heuristics rely on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from loguru import logger

ElementCategory = Literal["Title", "KeyValue", "ListItem", "NarrativeText", "Table", "Other"]

_KV_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 .#/&()_'-]{0,60}?)\s*[:：]\s*(.+?)\s*$")
_LIST_RE = re.compile(r"^\s*(?:[-*•▪]|\d+[.)]|[a-z][.)])\s+\S")
_TABLE_ROW_RE = re.compile(r"\S(?:\s{2,}|\t|\s\|\s)\S")


@dataclass(frozen=True)
class Element:
    """One layout element with its category and text."""

    category: ElementCategory
    text: str
    key: str | None = None
    value: str | None = None


def classify_line(line: str) -> Element:
    """Assign a layout category to a single line.

    Args:
        line: One line of document text.

    Returns:
        The typed element (``KeyValue`` carries ``key`` and ``value``).
    """
    stripped = line.strip()
    if not stripped:
        return Element("Other", "")
    kv = _KV_RE.match(stripped)
    if kv and len(kv.group(2)) <= 200:
        return Element("KeyValue", stripped, key=kv.group(1).strip(), value=kv.group(2).strip())
    if _LIST_RE.match(stripped):
        return Element("ListItem", stripped)
    if _TABLE_ROW_RE.search(stripped) and len(stripped.split()) >= 3:
        return Element("Table", stripped)
    words = stripped.split()
    if len(words) <= 8 and (
        stripped.isupper() or (stripped.istitle() and not stripped.endswith("."))
    ):
        return Element("Title", stripped)
    return Element("NarrativeText", stripped)


def partition_heuristic(text: str) -> list[Element]:
    """Rule-based partitioner: one element per non-empty line."""
    return [e for e in (classify_line(line) for line in text.splitlines()) if e.text]


def partition_unstructured(text: str) -> list[Element]:
    """Partition with ``unstructured`` (optional dependency).

    Raises:
        ImportError: If ``unstructured`` is not installed.
        LookupError: If its NLTK models are missing and cannot be downloaded.
    """
    from unstructured.partition.text import partition_text  # lazy optional import

    out: list[Element] = []
    for el in partition_text(text=text):
        cat = type(el).__name__
        body = str(el)
        if cat == "Title":
            out.append(Element("Title", body))
        elif cat == "ListItem":
            out.append(Element("ListItem", body))
        elif cat == "Table":
            out.append(Element("Table", body))
        else:
            kv = _KV_RE.match(body)
            if kv:
                out.append(
                    Element("KeyValue", body, key=kv.group(1).strip(), value=kv.group(2).strip())
                )
            else:
                out.append(Element("NarrativeText", body))
    return out


def partition(text: str, use_unstructured: bool = False) -> list[Element]:
    """Partition document text into typed elements.

    Args:
        text: Full document text.
        use_unstructured: Try the ``unstructured`` library first.

    Returns:
        Elements in reading order. Falls back to the heuristic partitioner (and
        logs why) when ``unstructured`` is unavailable.
    """
    if use_unstructured:
        try:
            return partition_unstructured(text)
        except (ImportError, LookupError, OSError) as exc:
            logger.warning("unstructured unavailable ({!r}); using heuristic partitioner", exc)
    return partition_heuristic(text)


def key_value_pairs(elements: list[Element]) -> list[tuple[str, str]]:
    """Return the ``(key, value)`` pairs from ``KeyValue`` elements, in order."""
    return [(e.key, e.value) for e in elements if e.category == "KeyValue" and e.key and e.value]
