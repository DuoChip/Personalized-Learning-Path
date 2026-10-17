# File: src/moocradar/translation.py
# Purpose: Create a resumable, offline Chinese-to-English metadata glossary.

import csv
import json
import logging
import re
from pathlib import Path

from .preparation import _detail
from .reading import iter_jsonl

CHINESE = re.compile(r"[\u3400-\u9fff]")
LOG = logging.getLogger(__name__)


def _save_cache(path: Path, strings: set[str], previous: dict[str, str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, ["source_zh", "english"])
        writer.writeheader()
        for source in sorted(strings | previous.keys()):
            writer.writerow({"source_zh": source, "english": previous.get(source, "")})
    temporary.replace(path)


def extract(raw_problem_path: Path, cache_path: Path, limit: int | None = None,
            batch_size: int = 1000) -> int:
    """Collect unique translatable strings while preserving existing translations."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    previous = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8", newline="") as file:
            previous = {row["source_zh"]: row["english"] for row in csv.DictReader(file)}
    strings = set()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(iter_jsonl(raw_problem_path)):
        if limit is not None and i >= limit:
            break
        detail = _detail(item)
        values = list(item.get("concepts") or [])
        values += [detail.get("title"), detail.get("content"), detail.get("typetext")]
        options = detail.get("option") or {}
        if isinstance(options, dict):
            values.extend(options.values())
        strings.update(value for value in values if isinstance(value, str) and CHINESE.search(value))
        if (i + 1) % batch_size == 0:
            _save_cache(cache_path, strings, previous)
            LOG.info("Extracted %s problems, %s unique Chinese strings", i + 1, len(strings))
    _save_cache(cache_path, strings, previous)
    return len(strings)


def apply(raw_problem_path: Path, cache_path: Path, output_path: Path) -> dict:
    """Join vetted translations by exact string, leaving all source IDs unchanged."""
    with cache_path.open(encoding="utf-8", newline="") as file:
        translations = {row["source_zh"]: row["english"] for row in csv.DictReader(file) if row["english"].strip()}
    count = translated = 0
    seen_ids = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, ["problem_id", "exercise_id", "course_id", "title_en", "content_en", "options_en_json", "concepts_en_json"])
        writer.writeheader()
        for item in iter_jsonl(raw_problem_path):
            if item["problem_id"] in seen_ids:
                raise ValueError("Duplicate problem_id in translation source")
            seen_ids.add(item["problem_id"])
            detail = _detail(item)
            options = detail.get("option") or {}
            if not isinstance(options, dict):
                options = {}
            concept_names = item.get("concepts") or []
            if not isinstance(concept_names, list):
                concept_names = []
            row = {
                "problem_id": item["problem_id"], "exercise_id": item.get("exercise_id", ""),
                "course_id": item.get("course_id", ""),
                "title_en": translations.get(detail.get("title"), ""),
                "content_en": translations.get(detail.get("content"), ""),
                "options_en_json": json.dumps({key: translations.get(value, "") for key, value in options.items()}, ensure_ascii=False),
                "concepts_en_json": json.dumps([translations.get(value, "") for value in concept_names], ensure_ascii=False),
            }
            writer.writerow(row)
            count += 1
            translated += bool(row["title_en"] or row["content_en"] or any(translations.get(value) for value in concept_names))
            if count % 1000 == 0:
                LOG.info("Applied translations to %s problems", count)
    return {"problem_rows": count, "rows_with_english": translated,
            "unique_translated_strings": len(translations), "ids_preserved": True}
