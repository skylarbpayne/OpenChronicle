"""Run configured extractors over grounded classifier context."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..logger import get
from ..prompts import load as load_prompt
from ..writer import llm as llm_mod
from .spec import ExtractorSpec
from .store import ExtractorRecord, upsert_record

logger = get("openchronicle.extractors")


@dataclass
class ExtractorRunResult:
    ran: list[str] = field(default_factory=list)
    written_count: int = 0
    skipped_count: int = 0
    errors: list[str] = field(default_factory=list)


def run_extractors_for_context(
    cfg: Config,
    conn: sqlite3.Connection,
    *,
    run_on: str,
    session_id: str,
    event_daily_path: str,
    context: str,
    now: str,
) -> ExtractorRunResult:
    result = ExtractorRunResult()
    for spec in cfg.extractors.enabled_specs(run_on=run_on):
        result.ran.append(spec.id)
        try:
            written, skipped = _run_one(
                cfg,
                conn,
                spec,
                session_id=session_id,
                event_daily_path=event_daily_path,
                context=context,
                now=now,
            )
            result.written_count += written
            result.skipped_count += skipped
            logger.info(
                "extractor %s: wrote=%d skipped=%d",
                spec.id, written, skipped,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("extractor %s failed: %s", spec.id, exc)
            result.errors.append(f"{spec.id}: {exc}")
    return result


def _run_one(
    cfg: Config,
    conn: sqlite3.Connection,
    spec: ExtractorSpec,
    *,
    session_id: str,
    event_daily_path: str,
    context: str,
    now: str,
) -> tuple[int, int]:
    system = _load_prompt(spec)
    user_msg = _render_user_message(spec, session_id=session_id, event_daily_path=event_daily_path, context=context)
    resp = llm_mod.call_llm(
        cfg,
        spec.model_stage,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        json_mode=True,
    )
    payload = _parse_json_response(llm_mod.extract_text(resp))
    raw_records = payload.get("records", []) or []
    if not raw_records:
        raw_records = _fallback_activity_records(
            spec,
            session_id=session_id,
            event_daily_path=event_daily_path,
            context=context,
        )
    written = 0
    skipped = 0
    for raw in raw_records:
        record = _record_from_payload(spec, raw, now=now)
        if record is None:
            skipped += 1
            continue
        upsert_record(conn, record)
        written += 1
    return written, skipped


def _fallback_activity_records(
    spec: ExtractorSpec,
    *,
    session_id: str,
    event_daily_path: str,
    context: str,
) -> list[dict[str, Any]]:
    """Create a conservative activity_signal when the LLM extracts nothing.

    Reducer entries are already grounded session summaries. For OpenChronicle's
    agent-use case, losing every activity-only session makes the extractor table
    useless, so preserve one coarse, source-backed activity record per session.
    """
    if not spec.allows_kind("activity_signal"):
        return []
    quote = _best_activity_quote(context)
    if not quote:
        return []
    summary = _activity_summary_from_quote(quote)
    digest = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:10]
    return [{
        "id": f"activity-{digest}",
        "kind": "activity_signal",
        "status": "active",
        "confidence": 0.62,
        "summary": summary,
        "payload": {
            "session_id": session_id,
            "event_path": event_daily_path,
            "derived_by": "fallback_activity_signal",
        },
        "source_refs": [{
            "event_path": event_daily_path,
            "quote": quote,
        }],
        "links": [],
    }]


def _best_activity_quote(context: str) -> str:
    lines = [ln.strip() for ln in context.splitlines()]
    candidates: list[str] = []
    priority_prefixes = ("The user ", "- [")
    for line in lines:
        if not line or line.startswith("#"):
            continue
        if line.startswith(priority_prefixes):
            candidates.append(line)
    if not candidates:
        for line in lines:
            if not line or line.startswith("#"):
                continue
            if "Session " in line:
                candidates.append(line)
    if not candidates:
        candidates = [ln for ln in lines if len(ln) >= 24 and not ln.startswith("#")]
    if not candidates:
        return ""
    return _compact_ws(candidates[0])[:280]


def _activity_summary_from_quote(quote: str) -> str:
    text = re.sub(r"^[-*]\s*", "", quote)
    text = re.sub(r"^\[[^\]]+\]\s*", "", text)
    text = re.sub(r"^\*\*Session\s+[^*]+\*\*\s*\([^)]*\)\s*", "", text)
    text = _compact_ws(text).strip(" -—")
    if not text:
        return "Session activity captured."
    if len(text) > 140:
        text = text[:137].rstrip() + "..."
    return text[0].upper() + text[1:]


def _compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _load_prompt(spec: ExtractorSpec) -> str:
    try:
        return load_prompt(spec.prompt)
    except FileNotFoundError:
        kinds = ", ".join(spec.kinds)
        return (
            "You are an OpenChronicle extractor. Extract only grounded typed records. "
            f"Allowed kinds: {kinds}. Return strict JSON with a top-level records array. "
            "Each record needs id, kind, status, confidence, summary, payload, source_refs, and links. "
            "Every source_refs item must include a quote or entry_id. Prefer [] over guessing."
        )


def _render_user_message(
    spec: ExtractorSpec,
    *,
    session_id: str,
    event_daily_path: str,
    context: str,
) -> str:
    return (
        f"Extractor id: {spec.id}\n"
        f"Mode: {spec.mode}\n"
        f"Allowed kinds: {', '.join(spec.kinds)}\n"
        f"Max records: {spec.max_records}\n"
        f"Minimum confidence: {spec.min_confidence}\n"
        f"Session: {session_id}\n"
        f"Event daily path: {event_daily_path}\n\n"
        "# Grounded context\n\n"
        f"{context}\n\n"
        "Return JSON only. Do not emit records without source_refs."
    )


def _parse_json_response(text: str) -> dict[str, Any]:
    stripped = (text or "{}").strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        fenced = "\n".join(lines).strip()
        if fenced:
            candidates.append(fenced)
    embedded = _extract_first_json_object(stripped)
    if embedded and embedded not in candidates:
        candidates.append(embedded)

    last_exc: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate or "{}")
        except json.JSONDecodeError as exc:
            last_exc = exc
            continue
        if not isinstance(data, dict):
            raise ValueError("extractor JSON must be an object")
        records = data.get("records", [])
        if not isinstance(records, list):
            raise ValueError("extractor JSON records must be a list")
        return data
    if last_exc is not None:
        raise ValueError(
            f"extractor returned invalid JSON: {last_exc}; shape={_json_text_shape(stripped)}"
        ) from last_exc
    return {"records": []}


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def _json_text_shape(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if stripped.startswith("```"):
        return f"code_fence,len={len(stripped)}"
    if stripped.startswith("{"):
        return f"object_like,len={len(stripped)}"
    if stripped.startswith("["):
        return f"array_like,len={len(stripped)}"
    return f"non_json_prefix,len={len(stripped)},first_char={stripped[0]!r}"


def _record_from_payload(spec: ExtractorSpec, raw: Any, *, now: str) -> ExtractorRecord | None:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind") or "")
    confidence = float(raw.get("confidence") or 0.0)
    if not spec.allows_kind(kind):
        return None
    if confidence < spec.min_confidence:
        return None
    source_refs = raw.get("source_refs") or []
    if not source_refs:
        return None
    try:
        return ExtractorRecord(
            id=str(raw.get("id") or ""),
            extractor_id=spec.id,
            kind=kind,
            status=str(raw.get("status") or "active"),
            confidence=confidence,
            summary=str(raw.get("summary") or ""),
            payload=dict(raw.get("payload") or {}),
            source_refs=list(source_refs),
            links=list(raw.get("links") or []),
            created_at=str(raw.get("created_at") or now),
            updated_at=str(raw.get("updated_at") or now),
            superseded_by=raw.get("superseded_by"),
        )
    except (TypeError, ValueError):
        return None
