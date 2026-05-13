# Configurable Extractors Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Turn OpenChronicle’s one hard-coded classifier into a configurable extraction system that can extract commitments, people, decisions, risks, open loops, and future user-defined record types from the same grounded session context.

**Architecture:** Keep the existing reducer/event-daily pipeline as the source of truth, then run one or more enabled extractors over each classified window. Each extractor has a typed spec, a prompt contract, a JSON schema for emitted records, and storage/query behavior. Built-ins ship with the fork; user config can enable/disable or add extractors without changing Python code.

**Tech Stack:** Python dataclasses/Pydantic-like validation via stdlib + `jsonschema` if added, existing LiteLLM wrapper, existing SQLite/FTS store, TOML config, MCP FastMCP tools.

---

## Why this beats “just add `commitment-`”

Issue #7 is right, but `commitment-` as another hard-coded classifier prefix is the small version of the idea. The real product shape is:

- `commitments`: promises, deadlines, meetings, follow-ups
- `people`: entities, aliases, relationships, roles, open threads
- `decisions`: durable choices and rationale
- `risks`: blockers, fragile assumptions, warnings
- `open_loops`: unresolved tasks/questions that may not have a clear deadline
- later: custom user/team extractors

A fixed classifier prompt will become a junk drawer. Configurable extractors keep each signal crisp, testable, queryable, and replaceable.

## Product principle

OpenChronicle should not only remember prose. It should expose **typed context objects** that agents can query operationally:

```text
what did I commit to this week?
who have I been discussing OpenChronicle with?
what decisions did I make about the MacBook bridge?
what unresolved loops are waiting on me?
```

That requires structure, not just better Markdown.

## Proposed model

### Existing layers

1. Raw captures: literal screen/context buffer.
2. Timeline blocks: short, verbatim-preserving slices.
3. Event-daily: reducer-written narrative of what happened.
4. Durable memories: current classifier-written Markdown entity files.

### New layer

5. Extracted records: typed JSON records grounded in event/timeline source refs.

Markdown memory can still exist, but typed records become the primary API for operational queries.

## Extractor spec

Add a config shape like:

```toml
[[extractors]]
id = "commitments"
enabled = true
kind = "commitment"
prompt = "extractors/commitments.md"
schema = "extractors/commitment.schema.json"
model_stage = "classifier"
run_on = "classified_window"       # classified_window | session_end | daily
max_records = 20
min_confidence = 0.55
materialize_prefix = "commitment-" # optional Markdown projection, not primary store

[[extractors]]
id = "people"
enabled = true
kind = "person_signal"
prompt = "extractors/people.md"
schema = "extractors/person_signal.schema.json"
model_stage = "classifier"
run_on = "classified_window"
max_records = 30
min_confidence = 0.45
materialize_prefix = "person-"     # optional projection into existing person files
```

### `ExtractorSpec`

Create `src/openchronicle/extractors/spec.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RunMode = Literal["classified_window", "session_end", "daily"]

@dataclass
class ExtractorSpec:
    id: str
    kind: str
    enabled: bool = True
    prompt: str = ""
    schema: str = ""
    model_stage: str = "classifier"
    run_on: RunMode = "classified_window"
    max_records: int = 20
    min_confidence: float = 0.5
    materialize_prefix: str = ""
    tags: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
```

## Common record envelope

Every extractor emits records with a shared envelope and a type-specific `payload`:

```json
{
  "id": "commitment-2026-05-12-bob-q3-roadmap",
  "extractor_id": "commitments",
  "kind": "commitment",
  "status": "active",
  "confidence": 0.74,
  "summary": "1:1 with Bob about Q3 roadmap",
  "payload": {
    "what": "1:1 with Bob about Q3 roadmap",
    "when": "2026-05-15T15:00:00-07:00",
    "when_text": "Friday at 3",
    "with": ["person-bob"],
    "action_by": "user"
  },
  "source_refs": [
    {
      "event_path": "event-2026-05-12.md",
      "entry_id": "e-abc123",
      "timeline_start": "2026-05-12T14:20:00-07:00",
      "timeline_end": "2026-05-12T14:25:00-07:00",
      "quote": "I can take that by Friday"
    }
  ],
  "links": ["person-bob", "project-q3-roadmap"],
  "created_at": "2026-05-12T21:30:00-07:00",
  "updated_at": "2026-05-12T21:30:00-07:00",
  "superseded_by": null
}
```

Hard rules:

- Every record needs at least one `source_ref`.
- `confidence < min_confidence` is stored only if the extractor config says to keep low-confidence records; default is skip.
- No extractor may perform side effects outside local OpenChronicle memory.
- Calendar sync, task creation, sending messages, etc. remain companion tools that consume extracted records.

## Storage

Create a new SQLite table rather than forcing structured data into Markdown:

```sql
CREATE TABLE IF NOT EXISTS extractor_records (
  id TEXT PRIMARY KEY,
  extractor_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  confidence REAL NOT NULL,
  summary TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  source_refs_json TEXT NOT NULL,
  links_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  superseded_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_extractor_records_kind_status
  ON extractor_records(kind, status);

CREATE INDEX IF NOT EXISTS idx_extractor_records_extractor_updated
  ON extractor_records(extractor_id, updated_at);
```

Keep Markdown projection optional. For commitments, the query API should not parse Markdown to answer date/status questions.

## Runner shape

Create `src/openchronicle/extractors/runner.py`:

```python
def run_extractors_for_window(
    cfg: Config,
    *,
    session_id: str,
    event_daily_path: str,
    start: datetime,
    end: datetime,
    context: str,
) -> ExtractorRunResult:
    specs = enabled_specs(cfg, run_on="classified_window")
    for spec in specs:
        run_extractor(cfg, spec, context=..., source=...)
```

The current `classifier.classify_window(...)` should eventually become one extractor or call the extractor runner after its existing durable-memory pass.

Recommended transition:

1. Add extractor store + config + runner without touching current classifier behavior.
2. Add built-in `commitments` extractor behind `extractors.commitments.enabled = false` or fork default true.
3. Add MCP read APIs.
4. Once stable, migrate durable-fact classifier into an extractor-style spec.

## LLM interaction

Use structured JSON first, not tool calls, for extractor records.

Why: extractors should be cheap, bounded, and easy to validate. Tool loops are good for read/append/supersede workflows; extraction should return a record list and let deterministic Python dedup/store.

Call shape:

```python
resp = llm_mod.call_llm(
    cfg,
    spec.model_stage,
    messages=[
        {"role": "system", "content": extractor_prompt},
        {"role": "user", "content": rendered_context},
    ],
    response_format={"type": "json_object"},
)
```

The `chatgpt/` Responses adapter already added to the fork must support JSON output cleanly. If LiteLLM Responses JSON mode is flaky, fall back to prompt-enforced JSON and strict parser repair only inside the runner.

## Built-in extractor: commitments

Schema fields:

```json
{
  "what": "string",
  "when": "string|null",
  "when_text": "string|null",
  "action_by": "user|other|unknown",
  "with": ["string"],
  "source_quote": "string",
  "status": "pending|done|cancelled|superseded|unknown"
}
```

Rules:

- Capture explicit promises, deadlines, appointments, accepted meetings, interviews, and follow-ups.
- Require `what` plus either `when` or `when_text`, unless the extractor is configured to keep open loops.
- Preserve ambiguous temporal language in `when_text`; set `when = null` when resolution is unsafe.
- Do not write to calendar or reminders.
- Do not infer commitments from vague browsing.

## Built-in extractor: people

This should not spam `person-*` files with every mention.

Schema fields:

```json
{
  "name": "string",
  "aliases": ["string"],
  "role_or_affiliation": "string|null",
  "relationship_to_user": "string|null",
  "interaction_summary": "string",
  "open_thread": "string|null",
  "source_quote": "string"
}
```

Rules:

- Capture durable identity/relationship facts and recurring interaction threads.
- Skip one-off mentions with no lasting context.
- Prefer `person_signal` records first; project into `person-*.md` only after dedup or repeated signal.

## MCP API

Add generic tools:

```text
list_extractor_records(kind?, status?, since?, until?, limit?)
read_extractor_record(id)
search_extractor_records(query, kind?, status?, top_k?)
```

Do **not** add convenience tools per built-in kind yet. Keep entity specialization in `kind` filters so the API does not sprawl as extractors grow.

The generic tools keep the system extensible and still let agents ask narrow questions, e.g. `search_extractor_records(query="deck", kind="commitment")`.

## Config defaults for Skylar’s fork

In Skylar’s fork, default to:

```toml
[extractors]
enabled = true

[[extractors.items]]
id = "commitments"
enabled = true
kind = "commitment"
prompt = "extractors/commitments.md"
schema = "extractors/commitment.schema.json"
model_stage = "classifier"
run_on = "classified_window"
max_records = 20
min_confidence = 0.55

[[extractors.items]]
id = "people"
enabled = true
kind = "person_signal"
prompt = "extractors/people.md"
schema = "extractors/person_signal.schema.json"
model_stage = "classifier"
run_on = "classified_window"
max_records = 30
min_confidence = 0.45
```

Upstream-friendly default could be `enabled = false` until maintainers buy in.

## Implementation tasks

### Task 1: Add extractor config parsing

**Objective:** Load enabled extractor specs from TOML with sane built-in defaults.

**Files:**
- Create: `src/openchronicle/extractors/spec.py`
- Modify: `src/openchronicle/config.py`
- Test: `tests/test_extractor_config.py`

**Test first:** verify default config contains commitments/people in Skylar fork and custom TOML can disable one extractor.

Run:

```bash
uv run pytest tests/test_extractor_config.py -q
```

### Task 2: Add extractor record store

**Objective:** Store typed extracted records in SQLite with deterministic upsert/supersede behavior.

**Files:**
- Create: `src/openchronicle/extractors/store.py`
- Modify: `src/openchronicle/store/fts.py` or migration init path
- Test: `tests/test_extractor_store.py`

**Test first:** insert, list by kind/status, update/supersede, reject missing source refs.

Run:

```bash
uv run pytest tests/test_extractor_store.py -q
```

### Task 3: Add commitment extractor prompt + schema

**Objective:** Define grounded extraction contract for issue #7.

**Files:**
- Create: `src/openchronicle/prompts/extractors/commitments.md`
- Create: `src/openchronicle/prompts/extractors/commitment.schema.json`
- Test: `tests/test_extractor_specs.py`

**Test first:** schema validates a real pending commitment and rejects records without source refs.

### Task 4: Add generic extractor runner

**Objective:** Run enabled extractors against the existing classifier window context and store validated records.

**Files:**
- Create: `src/openchronicle/extractors/runner.py`
- Modify: `src/openchronicle/session/tick.py` or `writer/classifier.py`
- Test: `tests/test_extractor_runner.py`

**Test first:** monkeypatch `llm_mod.call_llm` to return JSON with one commitment; assert store receives one record with source refs and no Markdown write occurs.

### Task 5: Wire runner after classifier window

**Objective:** Run extractors after reducer/classifier has assembled the grounded event/timeline context.

**Files:**
- Modify: `src/openchronicle/writer/classifier.py`
- Test: `tests/test_classifier_extractors_integration.py`

**Design note:** avoid duplicating context assembly. Expose a helper like `build_classification_context(...)` or return context from `classify_window` path.

### Task 6: Add MCP read tools

**Objective:** Make extracted records usable by agents.

**Files:**
- Modify: `src/openchronicle/mcp/server.py`
- Test: `tests/test_mcp_extractor_records.py`

**Tools:**

```text
list_extractor_records(kind?, status?, since?, until?, limit?)
read_extractor_record(id)
search_extractor_records(query, kind?, status?, since?, until?, limit?)
```

**Rule:** avoid specialized MCP tools such as `list_commitments` until proven necessary; filter by `kind` instead.

### Task 7: Add people extractor as second proof

**Objective:** Prove configurability is real, not commitment-specific.

**Files:**
- Create: `src/openchronicle/prompts/extractors/people.md`
- Create: `src/openchronicle/prompts/extractors/person_signal.schema.json`
- Test: `tests/test_people_extractor.py`

**Acceptance:** people extractor shares runner/store/MCP plumbing with commitments and only differs by spec/prompt/schema.

### Task 8: Optional Markdown projection

**Objective:** Materialize selected record kinds into existing memory files without making Markdown the primary database.

**Files:**
- Create: `src/openchronicle/extractors/projector.py`
- Test: `tests/test_extractor_projection.py`

**Rule:** projection is derived/rebuildable. The SQLite record remains canonical.

## Acceptance criteria

- `uv run pytest -q` passes.
- With extractors enabled, a fake session containing “I’ll send Bob the deck by Friday” produces one `commitment` record.
- `search_extractor_records(query="deck", kind="commitment")` returns that record.
- A fake session mentioning “Bob” without durable relationship/context does not produce a person record.
- Disabling `commitments` in config prevents commitment extraction without changing code.
- No calendar/task side effects exist in OpenChronicle.

## Open decisions

1. Should Skylar’s fork default commitments/people to enabled immediately? Recommendation: yes, because this fork is explicitly for owned experience.
2. Should records be exported by the MacBook bridge? Recommendation: yes — add `extractor_records.sqlite` or JSON export to the R2 bundle once records exist.
3. Should the existing durable Markdown classifier become an extractor? Recommendation: eventually yes, but not in v1; keep behavior stable while adding typed records.
4. Should custom extractors support arbitrary Python plugins? Recommendation: not yet. Start with prompt/schema/config extractors. Python plugins are a bigger security and maintenance surface.

## Recommendation

Build commitments first, but design it as the first built-in extractor, not as a special case. Then add people immediately as the second extractor to force the abstraction to be real.
