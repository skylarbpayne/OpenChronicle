"""Extractor configuration model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ExtractorMode = Literal["single", "variadic"]
RunMode = Literal["classified_window", "session_end", "daily"]

DEFAULT_CORE_KINDS = ["commitment", "person_signal", "decision", "risk", "open_loop"]


@dataclass
class ExtractorSpec:
    """One configured extractor.

    `mode="variadic"` lets a single LLM pass emit multiple record kinds. `mode="single"`
    is still supported for high-precision extractors that deserve their own pass.
    """

    id: str
    enabled: bool = True
    mode: ExtractorMode = "single"
    kind: str = ""
    kinds: list[str] = field(default_factory=list)
    prompt: str = ""
    schema: str = ""
    model_stage: str = "classifier"
    run_on: RunMode = "classified_window"
    max_records: int = 20
    min_confidence: float = 0.5
    materialize_prefix: str = ""
    tags: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"single", "variadic"}:
            raise ValueError(f"invalid extractor mode: {self.mode}")
        if self.run_on not in {"classified_window", "session_end", "daily"}:
            raise ValueError(f"invalid extractor run_on: {self.run_on}")
        if not self.kinds and self.kind:
            self.kinds = [self.kind]
        if self.mode == "single" and not self.kind and len(self.kinds) == 1:
            self.kind = self.kinds[0]
        if not self.kinds:
            raise ValueError(f"extractor {self.id!r} needs kind or kinds")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")

    def allows_kind(self, kind: str) -> bool:
        return kind in self.kinds


@dataclass
class ExtractorConfig:
    enabled: bool = True
    items: list[ExtractorSpec] = field(default_factory=lambda: [default_core_extractor()])

    def enabled_specs(self, *, run_on: RunMode | None = None) -> list[ExtractorSpec]:
        if not self.enabled:
            return []
        specs = [spec for spec in self.items if spec.enabled]
        if run_on is not None:
            specs = [spec for spec in specs if spec.run_on == run_on]
        return specs


def default_core_extractor() -> ExtractorSpec:
    return ExtractorSpec(
        id="core",
        enabled=True,
        mode="variadic",
        kinds=list(DEFAULT_CORE_KINDS),
        prompt="extractors/core.md",
        schema="extractors/core.schema.json",
        model_stage="classifier",
        run_on="classified_window",
        max_records=25,
        min_confidence=0.55,
    )


def build_extractor_config(raw: dict[str, Any]) -> ExtractorConfig:
    """Build extractor config from TOML data, merging item overrides by id.

    The default core variadic extractor is present unless explicitly disabled or
    replaced by an item with `id = "core"`.
    """

    enabled = bool(raw.get("enabled", True))
    specs_by_id: dict[str, ExtractorSpec] = {"core": default_core_extractor()}
    for item in raw.get("items", []) or []:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            continue
        existing = specs_by_id.get(item_id)
        data = existing.__dict__.copy() if existing else {"id": item_id}
        data.update({k: v for k, v in item.items() if k in ExtractorSpec.__dataclass_fields__})
        if "kind" in item and "kinds" not in item:
            data["kinds"] = [item["kind"]] if item["kind"] else []
        specs_by_id[item_id] = ExtractorSpec(**data)
    return ExtractorConfig(enabled=enabled, items=list(specs_by_id.values()))
