# File: src/moocradar/reading.py
# Purpose: Stream MOOC-Radar JSON records without loading large arrays into RAM.

import json
from pathlib import Path
from typing import Iterator


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict]:
    """Yield top-level objects from a JSON array with bounded working memory."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as source:
        buffer = ""
        pos = 0
        eof = False

        def refill() -> None:
            nonlocal buffer, pos, eof
            chunk = source.read(chunk_size)
            buffer = buffer[pos:] + chunk
            pos = 0
            eof = not chunk

        refill()
        while True:
            if pos >= len(buffer):
                if eof:
                    raise ValueError(f"Empty or incomplete JSON array: {path}")
                refill()
            elif buffer[pos].isspace():
                pos += 1
            else:
                break
        if buffer[pos] != "[":
            raise ValueError(f"Expected JSON array: {path}")
        pos += 1

        first = True
        while True:
            while True:
                if pos >= len(buffer):
                    if eof:
                        raise ValueError(f"Truncated JSON array: {path}")
                    refill()
                    continue
                if buffer[pos].isspace():
                    pos += 1
                    continue
                break
            if buffer[pos] == "]":
                if not first:
                    raise ValueError(f"Trailing comma in JSON array: {path}")
                pos += 1
                if buffer[pos:].strip() or source.read().strip():
                    raise ValueError(f"Trailing content after JSON array: {path}")
                return
            while True:
                try:
                    item, end = decoder.raw_decode(buffer, pos)
                    pos = end
                    if not isinstance(item, dict):
                        raise ValueError("Expected objects inside JSON array")
                    yield item
                    break
                except json.JSONDecodeError as error:
                    if eof:
                        raise ValueError(f"Invalid JSON in {path}: {error}") from error
                    refill()
            first = False
            while True:
                if pos >= len(buffer):
                    if eof:
                        raise ValueError(f"Truncated JSON array: {path}")
                    refill()
                    continue
                if buffer[pos].isspace():
                    pos += 1
                    continue
                break
            if buffer[pos] == ",":
                pos += 1
                continue
            if buffer[pos] == "]":
                pos += 1
                if buffer[pos:].strip() or source.read().strip():
                    raise ValueError(f"Trailing content after JSON array: {path}")
                return
            raise ValueError(f"Missing separator in JSON array: {path}")


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield one JSON object per nonempty line."""
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                yield item
