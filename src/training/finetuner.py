"""Fine-tuner — QLoRA per specialist via unsloth, exports GGUF, registers with Ollama.

Requires GPU and:
    pip install "unsloth[colab-new]" trl transformers datasets

Falls back gracefully (logs warning, returns None) when unsloth is not installed.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from rich.console import Console


class SpecialistFinetuner:
    """QLoRA fine-tunes a specialist on its curated JSONL dataset.

    Flow: load JSONL → format as chat conversations → QLoRA train →
    export Q4_K_M GGUF → ollama create {specialist}-{date} → return model name.
    """

    def __init__(
        self,
        base_model: str = "unsloth/llama-3-8b-Instruct-bnb-4bit",
        lora_rank: int = 16,
        lora_alpha: int = 16,
        epochs: int = 3,
        batch_size: int = 4,
        grad_accumulation: int = 4,
        max_seq_length: int = 2048,
        min_examples: int = 10,
        output_dir: str | Path = "models",
        console: Console | None = None,
    ) -> None:
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_accumulation = grad_accumulation
        self.max_seq_length = max_seq_length
        self.min_examples = min_examples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = console or Console()

    def run(
        self,
        specialist_name: str,
        dataset_path: str | Path,
        system_prompt: str,
    ) -> str | None:
        """Fine-tune one specialist. Returns new Ollama model name, or None on skip/fail."""
        try:
            return self._run_inner(specialist_name, dataset_path, system_prompt)
        except Exception as exc:
            self.console.print(
                f"  [yellow]Fine-tune failed for {specialist_name} "
                f"({type(exc).__name__}: {exc}) — skipping[/yellow]"
            )
            return None

    def _run_inner(
        self,
        specialist_name: str,
        dataset_path: str | Path,
        system_prompt: str,
    ) -> str | None:
        try:
            import torch
            from datasets import Dataset
            from transformers import Trainer, TrainingArguments
            from unsloth import FastLanguageModel
            from unsloth.chat_templates import get_chat_template
        except ImportError:
            self.console.print(
                f"  [yellow]unsloth not installed — skipping fine-tune for "
                f"{specialist_name}[/yellow]\n"
                f'  [dim]pip install "unsloth\\[colab-new]" transformers datasets[/dim]'
            )
            return None

        examples = self._load_examples(Path(dataset_path))
        if len(examples) < self.min_examples:
            self.console.print(
                f"  [dim]{specialist_name}: {len(examples)} examples "
                f"< min {self.min_examples} — skipping[/dim]"
            )
            return None

        self.console.print(
            f"  [cyan]Fine-tuning {specialist_name} "
            f"({len(examples)} examples, {self.epochs} epochs)...[/cyan]"
        )

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=self.lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        dataset = self._build_dataset(
            examples, system_prompt, tokenizer, Dataset
        )  # noqa: N803

        version_tag = date.today().isoformat()
        adapter_dir = self.output_dir / specialist_name / version_tag / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        total_steps = max(1, (len(examples) * self.epochs) // self.batch_size)
        warmup = min(5, total_steps // 4)

        _train_kwargs: dict[str, Any] = dict(
            output_dir=str(adapter_dir),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accumulation,
            warmup_steps=warmup,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=max(1, total_steps // 3),
            save_strategy="no",
            report_to="none",
            optim="adamw_8bit",
        )

        trainer = Trainer(
            model=model,
            args=TrainingArguments(**_train_kwargs),
            train_dataset=dataset,
        )
        trainer.train()
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        self.console.print(f"  [dim]Adapter saved → {adapter_dir}[/dim]")

        gguf_dir = self.output_dir / specialist_name / version_tag / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_gguf(
            str(gguf_dir), tokenizer, quantization_method="q4_k_m"
        )

        # Free VRAM before next specialist
        del model
        torch.cuda.empty_cache()

        gguf_path = self._find_gguf(gguf_dir)
        if gguf_path is None:
            self.console.print(f"  [red]GGUF not found in {gguf_dir} — aborting[/red]")
            return None
        self.console.print(f"  [dim]GGUF exported → {gguf_path.name}[/dim]")

        model_name = self._register_ollama(
            specialist_name, version_tag, gguf_path, system_prompt
        )
        if model_name is None:
            return None
        self.console.print(f"  [green]Ollama model created: {model_name}[/green]")
        return model_name

    def _load_examples(self, path: Path) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        if not path.exists():
            return examples
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return examples

    def _build_dataset(
        self,
        examples: list[dict[str, Any]],
        system_prompt: str,
        tokenizer: Any,
        dataset_cls: Any,
    ) -> Any:
        confidence_map = {
            "win": "0.9",
            "vindicated": "0.85",
            "learn_from_winner": "0.75",
        }
        rows = []
        for ex in examples:
            confidence = confidence_map.get(ex.get("training_signal", "win"), "0.8")
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex["input"]},
                {
                    "role": "assistant",
                    "content": (
                        f"REASONING: Based on careful analysis.\n"
                        f"ANSWER: {ex['target_output']}\n"
                        f"CONFIDENCE: {confidence}"
                    ),
                },
            ]
            text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
            )
            rows.append(
                {
                    "input_ids": tokens["input_ids"],
                    "attention_mask": tokens["attention_mask"],
                    "labels": tokens["input_ids"].copy(),
                }
            )

        # Pad all sequences to the same length so the collator can stack them
        pad_id = tokenizer.pad_token_id or 0
        max_len = max(len(r["input_ids"]) for r in rows)
        for row in rows:
            pad_len = max_len - len(row["input_ids"])
            row["input_ids"] = row["input_ids"] + [pad_id] * pad_len
            row["attention_mask"] = row["attention_mask"] + [0] * pad_len
            row["labels"] = row["labels"] + [-100] * pad_len  # -100 ignored in loss

        return dataset_cls.from_list(rows)

    def _find_gguf(self, gguf_dir: Path) -> Path | None:
        # unsloth appends _gguf to the dir name, so search parent recursively
        candidates = sorted(gguf_dir.parent.rglob("*.gguf"))
        q4 = [c for c in candidates if "Q4_K_M" in c.name]
        return q4[0] if q4 else (candidates[0] if candidates else None)

    def _register_ollama(
        self,
        specialist_name: str,
        version_tag: str,
        gguf_path: Path,
        system_prompt: str,
    ) -> str | None:
        model_name = f"{specialist_name}-{version_tag}"
        modelfile = (
            f"FROM {gguf_path.resolve()}\n"
            f'SYSTEM """{system_prompt}"""\n'
            f"PARAMETER temperature 0.7\n"
            f"PARAMETER num_predict 2048\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".Modelfile", delete=False
        ) as mf:
            mf.write(modelfile)
            mf_path = Path(mf.name)
        try:
            subprocess.run(
                ["ollama", "create", model_name, "-f", str(mf_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            return model_name
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            self.console.print(
                f"  [yellow]ollama register failed for {model_name} "
                f"({type(exc).__name__}: {exc}) — skipping[/yellow]"
            )
            return None
        finally:
            mf_path.unlink(missing_ok=True)
