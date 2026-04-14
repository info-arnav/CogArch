"""Unit tests for config loader."""

import pytest

from src.config import load_config, load_coordinator_prompt, load_specialist_config


class TestConfigLoader:
    def test_load_default_config(self) -> None:
        cfg = load_config()
        assert "coordinator" in cfg
        assert "specialists" in cfg
        assert cfg["coordinator"]["model_id"] == "gpt-4o-mini"

    def test_load_specialist_config(self) -> None:
        cfg = load_specialist_config("logical")
        assert cfg.name == "logical"
        assert cfg.temperature == 0.4
        assert (
            "analytical" in cfg.system_prompt.lower()
            or "logical" in cfg.system_prompt.lower()
        )

    def test_load_missing_specialist_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_specialist_config("nonexistent")

    def test_load_coordinator_prompt(self) -> None:
        prompt_cfg = load_coordinator_prompt("synthesis")
        assert "prompt" in prompt_cfg
        assert "system_message" in prompt_cfg
        assert "{input}" in prompt_cfg["prompt"]

    def test_load_missing_coordinator_prompt_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_coordinator_prompt("nonexistent")
