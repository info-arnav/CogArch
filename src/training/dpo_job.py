"""One DPO fine-tuning job, run as its own process so GPU memory is released.

    python -m src.training.dpo_job <job.json>

The job file describes the base model, the adapter to continue from (or none),
the preference dataset, hyperparameters and the output directory.

Reference policy: when continuing an adapter, TRL keeps a frozen copy of the
starting adapter as the reference, so each cycle's DPO is anchored to the
previous cycle's policy. On the first cycle the reference is the base model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class DPOJob(BaseModel):
    base_model: str
    init_adapter: str | None
    dataset: str
    output_dir: str
    seed: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    beta: float
    epochs: int
    batch_size: int
    grad_accumulation: int
    max_length: int


def run(job: DPOJob) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(job.seed)
    tokenizer = AutoTokenizer.from_pretrained(job.base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model: Any = AutoModelForCausalLM.from_pretrained(
        job.base_model, dtype=torch.bfloat16
    )
    peft_config = None
    if job.init_adapter is not None:
        model = PeftModel.from_pretrained(model, job.init_adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=job.lora_rank,
            lora_alpha=job.lora_alpha,
            lora_dropout=job.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    dataset = load_dataset("json", data_files=job.dataset, split="train")
    args = DPOConfig(
        output_dir=job.output_dir,
        seed=job.seed,
        num_train_epochs=job.epochs,
        per_device_train_batch_size=job.batch_size,
        gradient_accumulation_steps=job.grad_accumulation,
        learning_rate=job.learning_rate,
        beta=job.beta,
        max_length=job.max_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.model.save_pretrained(job.output_dir, selected_adapters=["default"])
    Path(job.output_dir, "train_metrics.json").write_text(
        json.dumps(trainer.state.log_history, indent=2)
    )


if __name__ == "__main__":
    run(DPOJob.model_validate_json(Path(sys.argv[1]).read_text()))
