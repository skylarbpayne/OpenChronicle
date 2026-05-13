import pytest

from openchronicle.extractors import store as extractor_store
from openchronicle.store import fts


def _record(**overrides):
    base = {
        "id": "commitment-send-bob-deck",
        "extractor_id": "core",
        "kind": "commitment",
        "status": "active",
        "confidence": 0.9,
        "summary": "Send Bob the deck by Friday.",
        "payload": {"what": "Send Bob the deck", "when_text": "Friday"},
        "source_refs": [{"event_path": "event-2026-05-12.md", "entry_id": "e1", "quote": "I'll send Bob the deck by Friday"}],
        "links": ["person-bob"],
        "created_at": "2026-05-12T21:00:00-07:00",
        "updated_at": "2026-05-12T21:00:00-07:00",
    }
    base.update(overrides)
    return extractor_store.ExtractorRecord(**base)


def test_upsert_and_filter_extractor_records(ac_root) -> None:
    with fts.cursor() as conn:
        extractor_store.upsert_record(conn, _record())
        extractor_store.upsert_record(
            conn,
            _record(
                id="person-bob",
                kind="person_signal",
                summary="Bob is connected to the deck follow-up.",
                payload={"name": "Bob"},
            ),
        )

        commitments = extractor_store.list_records(conn, kind="commitment", status="active")
        people = extractor_store.list_records(conn, kind="person_signal")

    assert [r.id for r in commitments] == ["commitment-send-bob-deck"]
    assert [r.id for r in people] == ["person-bob"]
    assert commitments[0].payload["when_text"] == "Friday"


def test_supersede_record(ac_root) -> None:
    with fts.cursor() as conn:
        extractor_store.upsert_record(conn, _record())
        extractor_store.supersede_record(
            conn,
            "commitment-send-bob-deck",
            _record(
                id="commitment-send-bob-deck-done",
                status="done",
                summary="Sent Bob the deck.",
                updated_at="2026-05-13T09:00:00-07:00",
            ),
        )
        old = extractor_store.get_record(conn, "commitment-send-bob-deck")
        new = extractor_store.get_record(conn, "commitment-send-bob-deck-done")

    assert old.status == "superseded"
    assert old.superseded_by == "commitment-send-bob-deck-done"
    assert new.status == "done"


def test_reject_record_without_source_refs() -> None:
    with pytest.raises(ValueError):
        _record(source_refs=[])


def test_search_records_across_summary_payload_and_source_refs(ac_root) -> None:
    with fts.cursor() as conn:
        extractor_store.upsert_record(conn, _record())
        extractor_store.upsert_record(
            conn,
            _record(
                id="decision-use-r2-mailbox",
                kind="decision",
                summary="Use R2 as the MacBook context mailbox.",
                payload={"decision": "Use R2 mailbox", "rationale": "travel-proof store-and-forward"},
                source_refs=[{"event_path": "event-2026-05-12.md", "entry_id": "e2", "quote": "R2 mailbox"}],
                links=[],
                updated_at="2026-05-12T21:05:00-07:00",
            ),
        )
        decisions = extractor_store.search_records(conn, query="mailbox", kind="decision")
        all_hits = extractor_store.search_records(conn, query="deck")

    assert [r.id for r in decisions] == ["decision-use-r2-mailbox"]
    assert [r.id for r in all_hits] == ["commitment-send-bob-deck"]
