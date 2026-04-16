"""Unit tests for config loader."""

import pytest

from src.config import load_config, load_specialist_config


class TestConfigLoader:
    def test_load_default_config(self) -> None:
        cfg = load_config()
        assert "model" in cfg
        assert "specialists" in cfg
        assert "competition" in cfg
        assert cfg["model"]["ollama_id"] == "deepseek-coder:33b"

    def test_load_default_config_specialists(self) -> None:
        cfg = load_config()
        assert "enabled" in cfg["specialists"]
        assert "logical" in cfg["specialists"]["enabled"]

    def test_load_specialist_config(self) -> None:
        cfg = load_specialist_config("logical")
        assert cfg.name == "logical"
        assert (
            "analytical" in cfg.system_prompt.lower()
            or "logical" in cfg.system_prompt.lower()
        )

    def test_load_missing_specialist_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_specialist_config("nonexistent")
