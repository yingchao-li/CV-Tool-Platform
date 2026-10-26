"""Small IO helpers shared across CVKit modules."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def save_json(path: str | Path, payload: Mapping | Sequence) -> Path:
    """Write ``payload`` as pretty-printed JSON, creating parent folders."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output


def load_json(path: str | Path):
    """Read a JSON document."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(
    path: str | Path, rows: Iterable[Mapping], fieldnames: Sequence[str] | None = None
) -> Path:
    """Write an iterable of mappings as CSV with a header row."""
    rows = list(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output.write_text("", encoding="utf-8")
        return output

    fields = list(fieldnames) if fieldnames else list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output
