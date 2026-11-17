"""The live dashboard: watch a run's brain think, and inspect its memory.

    cogarch dashboard                     # http://127.0.0.1:8080

Reads only what runs write to disk, so it works for a run in progress and for
any finished run:

  GET /api/runs                              runs with an event log
  GET /api/events?run=R                      the run's events as server-sent
                                             events, then new ones as they come
  GET /api/memory?run=R&agent=A              every stored episode
  GET /api/memory/search?run=R&agent=A&q=..  every episode ranked by similarity
                                             to q, as recall would see it
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from src.config import MemoryConfig
from src.memory.embedder import cosine_similarity

PAGE = Path(__file__).with_name("page.html")
POLL_SECONDS = 0.25
NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _safe(name: str) -> str:
    if not NAME.match(name):
        raise HTTPException(400, "bad name")
    return name


def build_app(runs_dir: Path) -> FastAPI:
    app = FastAPI()
    embedders: dict[str, Any] = {}

    def run_dir(run: str) -> Path:
        path = runs_dir / _safe(run)
        if not (path / "events.jsonl").exists():
            raise HTTPException(404, "no such run")
        return path

    def episodes(run: str, agent: str) -> list[dict[str, Any]]:
        path = run_dir(run) / "memory" / _safe(agent) / "episodic.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def memory_config(run: str) -> MemoryConfig:
        first = (run_dir(run) / "events.jsonl").open().readline()
        start = json.loads(first) if first else {}
        return MemoryConfig.model_validate(start.get("config", {}).get("memory", {}))

    @app.get("/")
    async def page() -> FileResponse:
        return FileResponse(PAGE, headers={"Cache-Control": "no-store"})

    @app.get("/api/runs")
    async def runs() -> list[dict[str, Any]]:
        found = []
        for path in sorted(
            runs_dir.glob("*/events.jsonl"), key=lambda p: -p.stat().st_mtime
        ):
            found.append(
                {
                    "name": path.parent.name,
                    "bytes": path.stat().st_size,
                    "modified": path.stat().st_mtime,
                }
            )
        return found

    @app.get("/api/events")
    async def events(run: str, request: Request) -> StreamingResponse:
        path = run_dir(run) / "events.jsonl"
        offset = int(request.headers.get("last-event-id") or 0)

        async def stream() -> AsyncIterator[str]:
            position = offset
            pending = b""
            yield "retry: 1000\n\n"
            while not await request.is_disconnected():
                with open(path, "rb") as f:
                    f.seek(position)
                    chunk = f.read()
                if chunk:
                    pending += chunk
                    position += len(chunk)
                    *lines, pending = pending.split(b"\n")
                    sent = position - len(pending)
                    body = "".join(f"data: {line.decode()}\n" for line in lines if line)
                    if body:
                        yield f"id: {sent}\n{body}\n"
                else:
                    await asyncio.sleep(POLL_SECONDS)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/memory")
    async def memory(run: str, agent: str) -> dict[str, Any]:
        stored = episodes(run, agent)
        for e in stored:
            e.pop("embedding", None)
        return {
            "episodes": stored,
            "min_similarity": memory_config(run).min_similarity,
        }

    @app.get("/api/memory/search")
    async def search(run: str, agent: str, q: str, domain: str = "") -> dict[str, Any]:
        cfg = memory_config(run)
        if cfg.embedding_model not in embedders:
            from src.memory.embedder import SentenceTransformerEmbedder

            embedders[cfg.embedding_model] = await asyncio.to_thread(
                SentenceTransformerEmbedder, cfg.embedding_model
            )
        query = await asyncio.to_thread(embedders[cfg.embedding_model].embed, q)
        ranked = []
        for e in episodes(run, agent):
            meta = e.get("metadata", {})
            if domain and meta.get("domain") != domain:
                continue
            similarity = cosine_similarity(query, e.pop("embedding", []))
            ranked.append({"similarity": round(similarity, 4), **e})
        ranked.sort(key=lambda r: -r["similarity"])
        return {"results": ranked[:50], "min_similarity": cfg.min_similarity}

    return app


def serve(runs_dir: Path, port: int = 8080) -> None:
    import uvicorn

    uvicorn.run(build_app(runs_dir), host="127.0.0.1", port=port, log_level="warning")
