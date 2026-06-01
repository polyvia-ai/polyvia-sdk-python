"""Pydantic response models for the Polyvia API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class IngestResult(BaseModel):
    """Returned by a single-file ingest call."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str
    task_id: str
    status: str  # always "pending" immediately after upload


class BatchIngestItem(BaseModel):
    """One item in a batch ingest response — either success or error."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: Optional[str] = None
    task_id: Optional[str] = None
    status: Optional[str] = None
    file: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.document_id is not None


class BatchIngestResult(BaseModel):
    results: List[BatchIngestItem]
    errors: Optional[List[Dict[str, str]]] = None

    # Behave like a sequence of items so ``for item in batch``, ``batch[0]`` and
    # ``len(batch)`` work as expected. Without these, Pydantic v2's default
    # ``__iter__`` yields ``(field_name, value)`` tuples, so ``for item in batch:
    # item.task_id`` raised ``'tuple' object has no attribute 'task_id'``.
    # ``batch.results`` / ``batch.errors`` keep working unchanged.
    def __iter__(self):  # type: ignore[override]
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> "BatchIngestItem":
        return self.results[index]


class IngestionStatus(BaseModel):
    """Returned by ingest.status() and ingest.wait()."""

    task_id: str
    document_id: Optional[str] = None
    status: str  # pending | parsing | completed | failed
    error: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed")


class Document(BaseModel):
    """Document metadata."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    title: str
    status: str  # uploading | parsing | completed | failed
    file_type: Optional[str] = None
    file_url: Optional[str] = None
    summary: Optional[str] = None
    created_at: Optional[int] = None
    group_id: Optional[str] = None


class Group(BaseModel):
    """Document group."""

    id: str
    name: str
    color: Optional[str] = None
    created_at: Optional[int] = None


class QueryResult(BaseModel):
    """Answer returned by query()."""

    answer: str
    document_id: Optional[str] = None
    group_ids: Optional[List[str]] = None


class UsagePeriod(BaseModel):
    start: str
    end: str


class UsageCounters(BaseModel):
    period: int
    total: int


class UsageStats(BaseModel):
    requests: UsageCounters
    ingests: UsageCounters
    queries: UsageCounters
    # Workspace-scoped page count (sum across completed ingests in the
    # workspace this key belongs to). `requests`/`ingests`/`queries` are
    # per-key; `pages` and `audio_seconds` are per-workspace.
    pages: UsageCounters
    # Workspace-scoped audio seconds processed (whole seconds — convert to
    # minutes for display if needed).
    audio_seconds: UsageCounters
    documents_stored: int


class APIKeyInfo(BaseModel):
    name: Optional[str] = None
    prefix: Optional[str] = None
    created_at: Optional[int] = None
    last_used_at: Optional[int] = None


class Usage(BaseModel):
    api_key: APIKeyInfo
    period: UsagePeriod
    usage: UsageStats


class RateLimitWindow(BaseModel):
    minute: str
    month: str


class RateLimits(BaseModel):
    limits: Dict[str, Any]
    current: Dict[str, Any]
    resets_at: RateLimitWindow
