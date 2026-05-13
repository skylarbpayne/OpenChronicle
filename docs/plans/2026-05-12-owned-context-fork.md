# Owned OpenChronicle Experience Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Turn Skylar’s OpenChronicle fork into the owned MacBook context layer Palmer can trust: subscription-backed model support, queryable commitments, and a clean bridge to the Mac mini mirror.

**Architecture:** Keep OpenChronicle local-first and read-only from Palmer’s Mac mini. The MacBook fork owns capture, session reduction, classifier output, and model-provider compatibility. The existing `macbook-context-bridge-r2` repo remains transport-only: encrypted export/import + read-only MCP mirror, no OpenChronicle monkey-patching.

**Tech Stack:** Python 3.11+, Typer, LiteLLM, OpenAI Responses API / ChatGPT provider, Markdown memory files, SQLite/FTS, MCP, pytest.

---

## Current Diagnosis

### 1. ChatGPT/Codex subscription model path

OpenChronicle’s LLM wrapper currently assumes every provider can be called through:

```python
litellm.completion(...)
```

and normalized into:

```python
response.choices[0].message.content
response.choices[0].message.tool_calls
```

That is reasonable for normal Chat Completions-compatible providers. It fails for LiteLLM’s `chatgpt/...` provider in our setup because LiteLLM routes through the Responses API and its Chat Completions bridge throws:

```text
Unknown items in responses API response: []
```

So the fork needs a proper internal response abstraction, not a site-packages monkey patch.

### 2. Time-bound commitments, upstream issue #7

Upstream issue: https://github.com/Einsia/OpenChronicle/issues/7

The issue is right: time-bound commitments currently land as prose in `event-YYYY-MM-DD.md` and are not queryable as structured future obligations. The classifier prompt explicitly rejects one-off events/deadlines even though the schema and MCP examples imply users should be able to ask for upcoming commitments.

We want first-class commitment memory so Palmer can answer:

- “What did I commit to this week?”
- “What deadlines came out of Slack/Linear/email today?”
- “What do I owe Jacqueline / Motion / a vendor?”
- “What needs calendar scheduling vs just follow-up?”

## Non-goals

- Do not expose OpenChronicle’s unauthenticated MCP server over LAN/public internet.
- Do not make OpenChronicle write to Calendar/EventKit directly.
- Do not make the Mac mini perform capture/writer work.
- Do not patch installed `site-packages` from bridge scripts.
- Do not send screenshots/text to cloud models without explicit config/approval.

---

## Phase 0: Clean Ownership Boundaries

### Task 0.1: Keep bridge repo transport-only

**Objective:** Ensure `macbook-context-bridge-r2` never edits installed OpenChronicle source.

**Files:**
- Already modified in bridge repo: `scripts/configure-openchronicle-chatgpt-models.sh`
- Already modified in bridge repo: `tests/test_openchronicle_chatgpt_config.py`

**Verification:**

```bash
cd ~/macbook-context-bridge-r2
uv run pytest tests/test_openchronicle_chatgpt_config.py -q && scripts/smoke-test.sh
```

Expected: pass, and the script exits non-zero rather than monkey-patching OpenChronicle.

---

## Phase 1: First-class LLM Provider Adapter

### Task 1.1: Add an internal LLM response interface

**Objective:** Decouple OpenChronicle internals from LiteLLM’s Chat Completions object shape.

**Files:**
- Modify: `src/openchronicle/writer/llm.py`
- Test: `tests/test_writer_llm_responses.py`

**Implementation shape:**

Create lightweight dataclasses:

```python
@dataclass
class ToolCall:
    id: str | None
    name: str
    arguments: dict[str, Any]

@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
```

Then update `extract_text()` and `extract_tool_calls()` to support both old raw LiteLLM responses and the new `LLMResponse`.

**Verification:**

```bash
uv run pytest tests/test_writer_llm_responses.py tests/test_classifier.py -q
```

### Task 1.2: Add provider routing for `chatgpt/` models

**Objective:** When `model_cfg.model.startswith("chatgpt/")`, use `litellm.responses(...)` directly rather than `litellm.completion(...)`.

**Files:**
- Modify: `src/openchronicle/writer/llm.py`
- Test: `tests/test_writer_llm_responses.py`

**Key behavior:**

- Convert OpenAI-style chat messages into Responses `input` items.
- Convert OpenAI-style function tools into Responses function tools.
- Convert Responses `message` output into `LLMResponse.text`.
- Convert Responses `function_call` output into `LLMResponse.tool_calls`.
- Preserve `api_base`, `api_key`, `max_tokens` where configured.
- For JSON mode, pass Responses text format if supported; otherwise keep prompt-level JSON constraints.

**Verification:**

Mock `litellm.responses` and assert it is called for `chatgpt/...`, while `litellm.completion` remains the default for non-ChatGPT models.

### Task 1.3: Fix model health ping for reasoning/Responses models

**Objective:** Stop false-negative `openchronicle status` checks caused by tiny `max_tokens=4` pings.

**Files:**
- Modify: `src/openchronicle/writer/llm.py`
- Test: `tests/test_writer_llm_ping.py`

**Expected behavior:**

- Use `max_tokens`/`max_output_tokens` of at least 64 for health pings.
- Route pings through the same provider adapter as normal calls.
- Never let ping failures crash `status`.

**Verification:**

```bash
uv run pytest tests/test_writer_llm_ping.py -q
```

---

## Phase 2: Commitment Memory Prefix

### Task 2.1: Add `commitment-` as a valid memory prefix

**Objective:** Permit classifier-created files like `commitment-bob-q3-roadmap.md`.

**Files:**
- Modify: `src/openchronicle/store/files.py`
- Modify: `src/openchronicle/prompts/schema.md`
- Test: `tests/test_store.py`

**Rules:**

A commitment is a time-bound obligation, appointment, deadline, interview, follow-up, or promise with:

- explicit or relative `when`
- concrete `what`
- optional `with` links to person/org/project files
- source grounding
- status: `pending | done | cancelled | superseded`
- confidence

### Task 2.2: Add commitment tool schema helpers

**Objective:** Give the classifier a safe way to create/update commitments without free-form YAML guessing.

**Files:**
- Modify: `src/openchronicle/writer/tools.py`
- Test: `tests/test_writer_tools.py`

**Tool options:**

Prefer a dedicated tool over just `append`:

```python
record_commitment(
    path: str,
    when: str | None,
    when_text: str,
    what: str,
    with_: list[str],
    source: str,
    status: str = "pending",
    confidence: float,
)
```

This keeps commitment entries consistently shaped.

### Task 2.3: Update classifier prompt policy

**Objective:** Reverse the current “single-occurrence deadlines don’t qualify” rule for grounded commitments.

**Files:**
- Modify: `src/openchronicle/prompts/classifier.md`
- Test: `tests/test_classifier.py`

**Prompt rule:**

Do not write every event. Only promote commitments when the session contains a grounded time-bound obligation with enough source evidence. Ambiguous dates are allowed only with `when = null`, verbatim `when_text`, and lower confidence.

### Task 2.4: Add classifier tests for commitment creation

**Objective:** Prove the classifier can create a commitment file from an event-daily entry.

**Files:**
- Modify: `tests/test_classifier.py`

**Test scenario:**

Given event text:

```text
Slack: Bob confirmed 1:1 next Friday at 3pm about Q3 roadmap; I said I’d prep notes beforehand.
```

The mocked classifier calls `record_commitment(...)`; test asserts:

- `commitment-bob-q3-roadmap.md` exists
- entry includes `when_text`, `what`, `with`, `source`, `status`, `confidence`
- FTS indexes it with prefix `commitment`

---

## Phase 3: Generic Typed Record MCP Surface

### Task 3.1: Add generic typed record MCP tools

**Objective:** Let agents query commitments and other extracted entities without raw FTS spelunking or per-kind tool sprawl.

**Files:**
- Modify: `src/openchronicle/mcp/server.py`
- Test: `tests/test_mcp_tools.py`

**Tool contract:**

```python
list_extractor_records(
    kind: str | None = None,    # e.g. "commitment", "person_signal", "decision"
    status: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 100,
)

search_extractor_records(
    query: str,
    kind: str | None = None,
    status: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 100,
)
```

Avoid specialized MCP tools such as `list_commitments` unless generic search/list proves insufficient.

---

## Phase 4: MacBook Context Bridge Compatibility

### Task 4.1: Ensure exported memory includes commitment files

**Objective:** Verify bridge exports include `commitment-*.md` automatically.

**Files:**
- In bridge repo: `src/macbook_context_bridge/exporter.py`
- Test: bridge smoke fixture

Likely no code change needed because memory file export should already include all `.md` memory files.

### Task 4.2: Add Palmer mirror generic extractor tools if needed

**Objective:** If OpenChronicle’s own MCP tools are not mirrored directly, add read-only mirror-side generic extractor tools in `macbook-context-bridge-r2`.

**Files:**
- In bridge repo: `src/macbook_context_bridge/mcp_server.py`

---

## Phase 5: Installation Path for Skylar’s MacBook

### Task 5.1: Add fork install script

**Objective:** Install Skylar’s fork into `~/.openchronicle/venv` without patching `site-packages` by hand.

**Files:**
- Create: `scripts/install-skylar-fork.sh` in the bridge repo or fork support docs

**Command:**

```bash
uv pip install --python ~/.openchronicle/venv/bin/python -e ~/code/OpenChronicle
```

### Task 5.2: Add config template for ChatGPT subscription models

**Objective:** Provide a safe config template for `chatgpt/...` once the fork supports it.

**Files:**
- Add docs/template in fork or bridge repo

**Verification:**

```bash
openchronicle status
openchronicle timeline tick
openchronicle writer run
```

Expected: model pings pass, timeline/reducer/classifier no longer show LiteLLM bridge errors.

---

## Recommended Implementation Order

1. LLM adapter abstraction and tests.
2. `chatgpt/` Responses routing.
3. Install fork editable on MacBook and verify one real writer run.
4. Configurable typed records + generic MCP read/search tools.
5. Bridge mirror support only if OpenChronicle memory export is insufficient.

## Acceptance Criteria

- No setup script monkey-patches installed packages.
- `openchronicle status` passes for ChatGPT/Codex-subscription models.
- Writer produces memory/timeline output from real captures.
- Typed extractor records are queryable through generic MCP tools with `kind` filters.
- Mac mini Palmer mirror receives commitment memory through the existing encrypted R2 bridge.
