"""Configuration loader — reads YAML configs into typed Python objects."""

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from src.models.specialist import SpecialistConfig


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def load_config(config_path: Path | str = "config/default.yaml") -> dict:
    """Load the main CogArch configuration."""
    return load_yaml(Path(config_path))


def load_specialist_config(
    name: str, prompts_dir: Path | str = "prompts/specialists"
) -> SpecialistConfig:
    """Load a specialist's YAML config into a SpecialistConfig model."""
    path = Path(prompts_dir) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config found for specialist '{name}' at {path}")
    data = load_yaml(path)
    return SpecialistConfig(**data)


def load_all_specialist_configs(
    names: list[str], prompts_dir: Path | str = "prompts/specialists"
) -> dict[str, SpecialistConfig]:
    """Load configs for all enabled specialists."""
    return {name: load_specialist_config(name, prompts_dir) for name in names}


def load_coordinator_prompt(
    name: str, prompts_dir: Path | str = "prompts/coordinator"
) -> dict:
    """Load a coordinator prompt YAML by name."""
    path = Path(prompts_dir) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No prompt found for coordinator '{name}' at {path}")
    return load_yaml(path)
