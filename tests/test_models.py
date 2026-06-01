"""Model behaviour tests — regression coverage for batch-ingest iteration."""

from polyvia._models import BatchIngestItem, BatchIngestResult


def _batch():
    return BatchIngestResult(
        results=[
            BatchIngestItem(document_id="doc_1", task_id="task_1", status="pending"),
            BatchIngestItem(document_id="doc_2", task_id="task_2", status="pending"),
        ]
    )


def test_iterating_batch_yields_items_not_field_tuples():
    """Regression: ``for item in batch`` used to yield Pydantic ``(field, value)``
    tuples, so ``item.task_id`` raised ``'tuple' object has no attribute ...``."""
    batch = _batch()
    task_ids = [item.task_id for item in batch]
    assert task_ids == ["task_1", "task_2"]
    for item in batch:
        assert isinstance(item, BatchIngestItem)


def test_batch_len_and_indexing():
    batch = _batch()
    assert len(batch) == 2
    assert batch[0].task_id == "task_1"
    assert batch[1].document_id == "doc_2"


def test_batch_results_and_errors_still_accessible():
    batch = _batch()
    assert [i.task_id for i in batch.results] == ["task_1", "task_2"]
    assert batch.errors is None


def test_batch_serialization_unaffected_by_custom_iter():
    """Overriding __iter__ must not break Pydantic serialization."""
    batch = _batch()
    dumped = batch.model_dump()
    assert dumped["results"][1]["task_id"] == "task_2"
    assert BatchIngestResult.model_validate(dumped).results[0].document_id == "doc_1"


def test_batch_item_ok_flag():
    ok = BatchIngestItem(document_id="d", task_id="t", status="pending")
    err = BatchIngestItem(file="x.pdf", error="boom")
    assert ok.ok is True
    assert err.ok is False
