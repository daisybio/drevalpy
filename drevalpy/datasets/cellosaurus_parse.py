"""Streaming parser for Cellosaurus reference text files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _CellosaurusRecord:
    ids: list[str] = field(default_factory=list)
    name: str | None = None
    site: str | None = None
    disease: str | None = None

    def clear(self) -> None:
        self.ids = []
        self.name = None
        self.site = None
        self.disease = None


def _parse_id_line(line: str, record: _CellosaurusRecord) -> None:
    record.ids = [s.strip() for s in line[5:].split(";") if s.strip()]


def _parse_name_line(line: str, record: _CellosaurusRecord) -> None:
    record.name = line.strip().split("   ")[1]


def _parse_site_line(line: str, record: _CellosaurusRecord) -> None:
    parts = line.strip().split(":", 1)[1].split(";")
    if len(parts) >= 2:
        record.site = parts[1].strip()


def _parse_disease_line(line: str, record: _CellosaurusRecord) -> None:
    if not record.ids:
        return
    parts = line[5:].split(";")
    if len(parts) >= 3:
        record.disease = parts[2].strip()


def _flush_cellosaurus_record(
    record: _CellosaurusRecord,
    id_to_name: dict[str, str],
    id_to_site: dict[str, str],
    id_to_disease: dict[str, str],
) -> None:
    for cid in record.ids:
        if record.name:
            id_to_name[cid] = record.name
        if record.site:
            id_to_site[cid] = record.site
        if record.disease:
            id_to_disease[cid] = record.disease
    record.clear()


def parse_cellosaurus(cellosaurus_path: str | Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Parse Cellosaurus file and return mappings from cellosaurus ID to name, site, and disease.

    :param cellosaurus_path: Path to the Cellosaurus text file
    :return: Tuple of dictionaries (id_to_name, id_to_site, id_to_disease)
    """
    id_to_name: dict[str, str] = {}
    id_to_site: dict[str, str] = {}
    id_to_disease: dict[str, str] = {}
    record = _CellosaurusRecord()

    with open(cellosaurus_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ID   "):
                _parse_name_line(line, record)
            elif line.startswith("AC   "):
                _parse_id_line(line, record)
            elif line.startswith("CC   Derived from site:"):
                _parse_site_line(line, record)
            elif line.startswith("DI   "):
                _parse_disease_line(line, record)
            elif line.strip() == "//":
                _flush_cellosaurus_record(record, id_to_name, id_to_site, id_to_disease)

    return id_to_name, id_to_site, id_to_disease
