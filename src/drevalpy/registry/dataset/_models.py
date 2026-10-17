"""Pydantic models for drevalpy user configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SourceEntry(BaseModel):
    """A dataset source with a base URL and optional fsspec storage options."""

    url: str
    storage_options: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: str | dict[str, Any]) -> SourceEntry:
        """Parse either a plain URL string or a {url, storage_options} dict."""
        if isinstance(raw, str):
            return cls(url=raw)
        return cls.model_validate(raw)

    def to_raw(self) -> str | dict[str, Any]:
        """Serialize back: plain string if no storage_options, else dict."""
        if not self.storage_options:
            return self.url
        return {"url": self.url, "storage_options": self.storage_options}


class DatasetEntry(BaseModel):
    """A registered dataset pointing to a source and filename."""

    source: str
    file: str


class DrevalConfig(BaseModel):
    """Dataset registry config file schema.

    The file contains only sources and datasets at the root level.
    """

    model_config = {"extra": "forbid"}

    sources: dict[str, SourceEntry] = Field(default_factory=dict)
    datasets: dict[str, DatasetEntry] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> DrevalConfig:
        """Parse from a raw JSON dict."""
        sources = {name: SourceEntry.from_raw(val) for name, val in raw.get("sources", {}).items()}
        datasets = {name: DatasetEntry.model_validate(val) for name, val in raw.get("datasets", {}).items()}
        return cls(sources=sources, datasets=datasets)

    def to_raw(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "sources": {name: entry.to_raw() for name, entry in self.sources.items()},
            "datasets": {name: entry.model_dump() for name, entry in self.datasets.items()},
        }
