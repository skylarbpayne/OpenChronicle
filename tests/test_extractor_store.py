import pytest

from openchronicle.extractors import store as extractor_store
from openchronicle.store import fts


def _record(**overrides):
    base = {
        "id": "commitment-send-bob-deck",
        "extractor_id": "core",
        "kind": "commitment",
        "status": "active",
        "confidence": 0.82,
        "summary": "Send Bob the deck by Friday.",
        "payload": {"what": "Send Bob the deck", "when_text": "Friday", "action_by": "user"},
        "source_refs": [{"event_path": "event-2026-05-12.md", "entry_id": "e1", "quote": "I'll send Bob the deck by Friday"}],
        "links": ["person-bob"],
        "created_at": "2026-05-12T21:00:00-07:00",
        "updated_at": "2026-05-12T21:00:00-07:00",
    }
    base.update(overrides)
    return extractor_store.ExtractorRecord(**base)


def test_upsert_and_list_extractor_records(ac_root) -> None:
    with fts.cursor() as conn:
        record = _record()
        extractor_store.upsert_record(conn, record)

        rows = extractor_store.list_records(conn, kind="commitment", status="active")

    assert [row.id for row in rows] == ["commitment-send-bob-deck"]
    assert rows[0].payload["when_text"] == "Friday"
    assert rows[0].source_refs[0]["entry_id"] == "e1"


def test_record_requires_source_refs(ac_root) -> None:
    with pytest.raises(ValueError, match="source_refs"):
        _record(source_refs=[])


def test_supersede_marks_old_record(ac_root) -> None:
    with fts.cursor() as conn:
        old = _record()
        new = _record(
            id="commitment-send-bob-deck-v2",
            summary="Send Bob the revised deck by Monday.",
            payload={"what": "Send Bob the revised deck", "when_text": "Monday", "action_by": "user"},
            updated_at="2026-05-12T22:00:00-07:00",
        )
        extractor_store.upsert_record(conn, old)
        extractor_store.supersede_record(conn, old.id, new)

        old_row = extractor_store.get_record(conn, old.id)
        active = extractor_store.list_records(conn, kind="commitment", status="active")

    assert old_row is not None
    assert old_row.status == "superseded"
    assert old_row.superseded_by == "commitment-send-bob-deck-v2"
    assert [row.id for row in active] == ["commitment-send-bob-deck-v2"]
