from pathlib import Path

from openchronicle import config


def test_default_config_enables_variadic_core_extractor(tmp_path: Path) -> None:
    cfg = config.load(tmp_path / "missing.toml")

    assert cfg.extractors.enabled is True
    assert [spec.id for spec in cfg.extractors.enabled_specs()] == ["core"]
    core = cfg.extractors.enabled_specs()[0]
    assert core.mode == "variadic"
    assert core.kinds == ["commitment", "person_signal", "decision", "risk", "open_loop"]
    assert core.prompt == "extractors/core.md"
    assert core.schema == "extractors/core.schema.json"
    assert core.model_stage == "classifier"
    assert core.run_on == "classified_window"


def test_extractor_config_can_disable_default_and_add_specific_extractor(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[extractors]
enabled = true

[[extractors.items]]
id = "core"
enabled = false

[[extractors.items]]
id = "commitments"
enabled = true
mode = "single"
kind = "commitment"
prompt = "extractors/commitments.md"
schema = "extractors/commitment.schema.json"
model_stage = "classifier"
run_on = "session_end"
max_records = 12
min_confidence = 0.7
"""
    )

    cfg = config.load(path)

    specs = cfg.extractors.enabled_specs()
    assert [spec.id for spec in specs] == ["commitments"]
    assert specs[0].mode == "single"
    assert specs[0].kinds == ["commitment"]
    assert specs[0].run_on == "session_end"
    assert specs[0].max_records == 12
    assert specs[0].min_confidence == 0.7
