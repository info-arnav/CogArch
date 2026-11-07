"""Adapter version history per (agent, role), with rollback."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class AdapterVersion(BaseModel):
    cycle: int
    served_name: str
    path: str


class AdapterRegistry:
    """Persists to `<run_dir>/adapters/registry.json`.

    The current version of a role is the last entry; an empty history means
    the role runs on the base model.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._history: dict[str, list[AdapterVersion]] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            self._history = {
                key: [AdapterVersion.model_validate(v) for v in versions]
                for key, versions in raw.items()
            }

    @staticmethod
    def _key(agent: str, role: str) -> str:
        return f"{agent}/{role}"

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(
                {k: [v.model_dump() for v in vs] for k, vs in self._history.items()},
                indent=2,
            )
        )

    def current(self, agent: str, role: str) -> AdapterVersion | None:
        versions = self._history.get(self._key(agent, role), [])
        return versions[-1] if versions else None

    def push(self, agent: str, role: str, version: AdapterVersion) -> None:
        self._history.setdefault(self._key(agent, role), []).append(version)
        self._save()

    def rollback(self, agent: str, role: str, cycle: int) -> AdapterVersion | None:
        """Drop the version trained in `cycle` (if any); return the new current one."""
        versions = self._history.get(self._key(agent, role), [])
        if versions and versions[-1].cycle == cycle:
            versions.pop()
            self._save()
        return versions[-1] if versions else None
