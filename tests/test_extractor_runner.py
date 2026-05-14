import json
from types import SimpleNamespace

from openchronicle import config
from openchronicle.extractors import runner, store as extractor_store
from openchronicle.store import fts
from openchronicle.writer import llm as llm_mod


def test_variadic_extractor_stores_multiple_record_kinds_in_one_llm_pass(ac_root, monkeypatch) -> None:
    calls = []

    def fake_call_llm(cfg, stage, *, messages, tools=None, json_mode=False):
        calls.append({"stage": stage, "messages": messages, "json_mode": json_mode})
        payload = {
            "records": [
                {
                    "id": "commitment-send-bob-deck",
                    "kind": "commitment",
                    "status": "active",
                    "confidence": 0.9,
                    "summary": "Send Bob the deck by Friday.",
                    "payload": {"what": "Send Bob the deck", "when_text": "Friday", "action_by": "user"},
                    "source_refs": [{"event_path": "event-2026-05-12.md", "entry_id": "e1", "quote": "I'll send Bob the deck by Friday"}],
                    "links": ["person-bob"],
                },
                {
                    "id": "person-bob",
                    "kind": "person_signal",
                    "status": "active",
                    "confidence": 0.7,
                    "summary": "Bob is tied to the deck follow-up thread.",
                    "payload": {"name": "Bob", "interaction_summary": "Deck follow-up"},
                    "source_refs": [{"event_path": "event-2026-05-12.md", "entry_id": "e1", "quote": "Bob"}],
                    "links": [],
                },
            ]
        }
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])

    monkeypatch.setattr(llm_mod, "call_llm", fake_call_llm)
    cfg = config.load(ac_root / "config.toml")

    with fts.cursor() as conn:
        result = runner.run_extractors_for_context(
            cfg,
            conn,
            run_on="classified_window",
            session_id="sess_123",
            event_daily_path="event-2026-05-12.md",
            context="User wrote: I'll send Bob the deck by Friday.",
            now="2026-05-12T21:00:00-07:00",
        )
        commitments = extractor_store.list_records(conn, kind="commitment")
        people = extractor_store.list_records(conn, kind="person_signal")

    assert len(calls) == 1
    assert calls[0]["stage"] == "classifier"
    assert calls[0]["json_mode"] is True
    assert result.written_count == 2
    assert [r.id for r in commitments] == ["commitment-send-bob-deck"]
    assert [r.id for r in people] == ["person-bob"]


def test_variadic_extractor_accepts_fenced_json(ac_root, monkeypatch) -> None:
    def fake_call_llm(cfg, stage, *, messages, tools=None, json_mode=False):
        payload = {
            "records": [
                {
                    "id": "activity-python-project-cursor",
                    "kind": "activity_signal",
                    "status": "active",
                    "confidence": 0.8,
                    "summary": "Worked in Cursor on a Python project.",
                    "payload": {"project": "Python project", "tool": "Cursor"},
                    "source_refs": [{"event_path": "event-2026-05-12.md", "entry_id": "e2", "quote": "Cursor configuring"}],
                    "links": [],
                }
            ]
        }
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="```json\n" + json.dumps(payload) + "\n```"))])

    monkeypatch.setattr(llm_mod, "call_llm", fake_call_llm)
    cfg = config.load(ac_root / "config.toml")

    with fts.cursor() as conn:
        result = runner.run_extractors_for_context(
            cfg,
            conn,
            run_on="classified_window",
            session_id="sess_456",
            event_daily_path="event-2026-05-12.md",
            context="Activity: Cursor configuring Python project.",
            now="2026-05-12T21:00:00-07:00",
        )
        activities = extractor_store.list_records(conn, kind="activity_signal")

    assert result.errors == []
    assert result.written_count == 1
    assert [r.id for r in activities] == ["activity-python-project-cursor"]


def test_variadic_extractor_accepts_wrapped_json(ac_root, monkeypatch) -> None:
    def fake_call_llm(cfg, stage, *, messages, tools=None, json_mode=False):
        payload = {
            "records": [
                {
                    "id": "risk-bridge-fragility",
                    "kind": "risk",
                    "status": "active",
                    "confidence": 0.82,
                    "summary": "Bridge sync is fragile.",
                    "payload": {"risk": "Bridge sync fragility"},
                    "source_refs": [{"event_path": "event-2026-05-12.md", "entry_id": "e3", "quote": "fragile"}],
                    "links": [],
                }
            ]
        }
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Here is the JSON:\n" + json.dumps(payload)))])

    monkeypatch.setattr(llm_mod, "call_llm", fake_call_llm)
    cfg = config.load(ac_root / "config.toml")

    with fts.cursor() as conn:
        result = runner.run_extractors_for_context(
            cfg,
            conn,
            run_on="classified_window",
            session_id="sess_789",
            event_daily_path="event-2026-05-12.md",
            context="Risk: fragile bridge.",
            now="2026-05-12T21:00:00-07:00",
        )
        risks = extractor_store.list_records(conn, kind="risk")

    assert result.errors == []
    assert result.written_count == 1
    assert [r.id for r in risks] == ["risk-bridge-fragility"]


def test_variadic_extractor_filters_disabled_kinds_and_low_confidence(ac_root, monkeypatch) -> None:
    def fake_call_llm(cfg, stage, *, messages, tools=None, json_mode=False):
        payload = {
            "records": [
                {
                    "id": "commitment-too-weak",
                    "kind": "commitment",
                    "status": "active",
                    "confidence": 0.1,
                    "summary": "Weak maybe.",
                    "payload": {"what": "Maybe follow up"},
                    "source_refs": [{"event_path": "event-2026-05-12.md", "entry_id": "e1", "quote": "maybe"}],
                    "links": [],
                },
                {
                    "id": "calendar-event-nope",
                    "kind": "calendar_event",
                    "status": "active",
                    "confidence": 0.99,
                    "summary": "Not an allowed kind.",
                    "payload": {},
                    "source_refs": [{"event_path": "event-2026-05-12.md", "entry_id": "e1", "quote": "nope"}],
                    "links": [],
                },
            ]
        }
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])

    monkeypatch.setattr(llm_mod, "call_llm", fake_call_llm)
    cfg = config.load(ac_root / "config.toml")

    with fts.cursor() as conn:
        result = runner.run_extractors_for_context(
            cfg,
            conn,
            run_on="classified_window",
            session_id="sess_123",
            event_daily_path="event-2026-05-12.md",
            context="Maybe follow up.",
            now="2026-05-12T21:00:00-07:00",
        )
        rows = extractor_store.list_records(conn)

    assert result.written_count == 0
    assert rows == []
    assert result.skipped_count == 2
