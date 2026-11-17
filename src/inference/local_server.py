"""A small OpenAI-compatible server for laptops (Apple MPS, CUDA or CPU).

It speaks the subset of vLLM's API the experiment uses, so the same code runs
on a laptop and on the GPU server:

  POST /v1/chat/completions     (stream, continue_final_message supported)
  POST /v1/load_lora_adapter    {lora_name, lora_path}
  POST /v1/unload_lora_adapter  {lora_name}
  GET  /v1/models

Requests that arrive together are decoded together in one batch, each row with
its own LoRA adapter (PEFT mixed-adapter batches) and its own temperature.

    python -m src.inference.local_server --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

BASE = "__base__"
DECODE_EVERY = 4
GATHER_SECONDS = 0.15


@dataclass
class Job:
    prompt_ids: list[int]
    adapter: str
    temperature: float
    max_tokens: int
    seed: int
    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    generated: list[int] = field(default_factory=list)
    sent: str = ""
    done: bool = False

    def push(self, item: Any) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, item)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Engine:
    def __init__(self, model_name: str, max_batch: int) -> None:
        self.model_name = model_name
        self.max_batch = max_batch
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        self.model: Any = (
            AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.adapters: dict[str, str] = {}
        self.stop_ids = {
            i
            for i in [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            ]
            if isinstance(i, int) and i >= 0
        }
        self.pending: list[Job] = []
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.stats = {"batches": 0, "tokens": 0, "busy_seconds": 0.0}
        threading.Thread(target=self._worker, daemon=True).start()

    def prompt_ids(
        self, messages: list[dict], continue_final: bool, add_generation_prompt: bool
    ) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt and not continue_final,
            continue_final_message=continue_final,
        )
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def submit(self, job: Job) -> None:
        if job.adapter != BASE and job.adapter not in self.adapters:
            raise HTTPException(404, f"model {job.adapter} is not loaded")
        with self.lock:
            self.pending.append(job)
        self.wake.set()

    def load_adapter(self, name: str, path: str) -> None:
        from peft import PeftModel

        def load() -> None:
            if name in self.adapters:
                return
            if not hasattr(self.model, "peft_config"):
                self.model = PeftModel.from_pretrained(
                    self.model, path, adapter_name=name
                ).eval()
            else:
                self.model.load_adapter(path, adapter_name=name)
            self.model.to(self.device)
            self.adapters[name] = path

        self.executor.submit(load).result()

    def unload_adapter(self, name: str) -> None:
        def unload() -> None:
            if name not in self.adapters:
                raise HTTPException(404, f"adapter {name} is not loaded")
            self.model.delete_adapter(name)
            del self.adapters[name]

        self.executor.submit(unload).result()

    def _worker(self) -> None:
        while True:
            self.wake.wait()
            time.sleep(GATHER_SECONDS)
            with self.lock:
                batch, self.pending = (
                    self.pending[: self.max_batch],
                    self.pending[self.max_batch :],
                )
                if not self.pending:
                    self.wake.clear()
            if batch:
                start = time.perf_counter()
                try:
                    self.executor.submit(self._decode, batch).result()
                except (
                    Exception
                ) as exc:  # noqa: BLE001 — every waiting request must hear
                    for job in batch:
                        if not job.done:
                            job.push(exc)
                self.stats["batches"] += 1
                self.stats["busy_seconds"] += time.perf_counter() - start

    @torch.inference_mode()
    def _decode(self, batch: list[Job]) -> None:
        n = len(batch)
        width = max(len(j.prompt_ids) for j in batch)
        pad = self.tokenizer.pad_token_id or 0
        ids = torch.full((n, width), pad, dtype=torch.long)
        mask = torch.zeros((n, width), dtype=torch.long)
        for row, job in enumerate(batch):
            ids[row, width - len(job.prompt_ids) :] = torch.tensor(job.prompt_ids)
            mask[row, width - len(job.prompt_ids) :] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        temps = torch.tensor(
            [max(j.temperature, 1e-5) for j in batch], device=self.device
        ).unsqueeze(1)
        greedy = torch.tensor([j.temperature <= 0 for j in batch], device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(
            sum(j.seed for j in batch) % (2**31)
        )
        kwargs: dict[str, Any] = {}
        if hasattr(self.model, "peft_config"):
            kwargs["adapter_names"] = [j.adapter for j in batch]
        cache = DynamicCache()
        live = list(batch)
        inputs = ids
        step = 0
        while live:
            out = self.model(
                input_ids=inputs,
                attention_mask=mask,
                position_ids=positions if step == 0 else positions[:, -1:],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                **kwargs,
            )
            logits = out.logits[:, -1, :].float()
            probs = torch.softmax(logits / temps, dim=-1)
            sampled = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            chosen = torch.where(greedy, logits.argmax(-1), sampled)
            for job, token in zip(live, chosen.tolist()):
                if token in self.stop_ids:
                    job.done = True
                else:
                    job.generated.append(token)
                    job.done = len(job.generated) >= job.max_tokens
                if job.done or step % DECODE_EVERY == 0:
                    self._flush(job)
                if job.done:
                    job.push(None)
            self.stats["tokens"] += len(live)
            keep = [i for i, job in enumerate(live) if not job.done]
            if len(keep) < len(live) and keep:
                index = torch.tensor(keep, device=self.device)
                cache.batch_select_indices(index)
                chosen, mask, positions = chosen[index], mask[index], positions[index]
                temps, greedy = temps[index], greedy[index]
                if "adapter_names" in kwargs:
                    kwargs["adapter_names"] = [kwargs["adapter_names"][i] for i in keep]
            live = [live[i] for i in keep]
            inputs = chosen.unsqueeze(1)
            mask = torch.cat(
                [
                    mask,
                    torch.ones((len(live), 1), dtype=mask.dtype, device=self.device),
                ],
                1,
            )
            positions = torch.cat([positions, positions[:, -1:] + 1], 1)
            step += 1

    def _flush(self, job: Job) -> None:
        text = self.tokenizer.decode(job.generated, skip_special_tokens=True)
        if text.endswith("�") and not job.done:
            return
        if len(text) > len(job.sent) and text.startswith(job.sent):
            job.push(text[len(job.sent) :])
            job.sent = text


def build_app(engine: Engine) -> FastAPI:
    app = FastAPI()

    async def tokens(job: Job) -> AsyncIterator[str]:
        while True:
            item = await job.queue.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    @app.get("/v1/models")
    async def models() -> dict:
        names = [engine.model_name, *engine.adapters]
        return {"object": "list", "data": [{"id": n, "object": "model"} for n in names]}

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True, "device": str(engine.device), **engine.stats}

    @app.post("/v1/load_lora_adapter")
    async def load(body: dict) -> dict:
        await asyncio.to_thread(
            engine.load_adapter, body["lora_name"], body["lora_path"]
        )
        return {"loaded": body["lora_name"]}

    @app.post("/v1/unload_lora_adapter")
    async def unload(body: dict) -> dict:
        await asyncio.to_thread(engine.unload_adapter, body["lora_name"])
        return {"unloaded": body["lora_name"]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Any:
        body = await request.json()
        model = body["model"]
        continue_final = bool(body.get("continue_final_message"))
        job = Job(
            prompt_ids=engine.prompt_ids(
                body["messages"],
                continue_final,
                body.get("add_generation_prompt", True),
            ),
            adapter=BASE if model == engine.model_name else model,
            temperature=float(body.get("temperature", 1.0)),
            max_tokens=int(body.get("max_tokens") or 512),
            seed=int(body.get("seed") or 0),
            queue=asyncio.Queue(),
            loop=asyncio.get_running_loop(),
        )
        engine.submit(job)
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        if body.get("stream"):

            async def events() -> AsyncIterator[str]:
                async for delta in tokens(job):
                    chunk = {
                        "id": rid,
                        "choices": [{"index": 0, "delta": {"content": delta}}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                usage = {
                    "prompt_tokens": len(job.prompt_ids),
                    "completion_tokens": len(job.generated),
                }
                yield f"data: {json.dumps({'id': rid, 'choices': [], 'usage': usage})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        text = "".join([d async for d in tokens(job)])
        return JSONResponse(
            {
                "id": rid,
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(job.prompt_ids),
                    "completion_tokens": len(job.generated),
                },
            }
        )

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-batch", type=int, default=48)
    args = parser.parse_args()
    engine = Engine(args.model, args.max_batch)
    uvicorn.run(
        build_app(engine), host="127.0.0.1", port=args.port, log_level="warning"
    )


if __name__ == "__main__":
    main()
