"""
LoRA Model Lab
----------------
Real LoRA/QLoRA training needs a GPU, a base model checkpoint, and a
labeled dataset — none of which exist in this environment. This module
provides:
  1. A real, correct LoRA config builder (uses HuggingFace `peft` if
     installed) so the exact production code path is present.
  2. A lightweight *simulated* training loop (pure NumPy) that exercises
     the same interface — adapter registration, loss curve, checkpoint
     metadata — so the platform is fully runnable end-to-end without a
     GPU. Swap `simulate_training` for `run_real_training` once you have
     compute + data.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LoRAConfig:
    base_model: str
    r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    task_type: str = "CAUSAL_LM"
    quantize_base: bool = False  # True => QLoRA (4-bit base + LoRA adapters)


@dataclass
class TrainingResult:
    adapter_name: str
    config: LoRAConfig
    epochs: int
    final_loss: float
    loss_curve: List[float]
    trainable_params_pct: float
    duration_sec: float


def build_real_peft_config(cfg: LoRAConfig):
    """Builds a real `peft.LoraConfig`. Requires `pip install peft transformers
    torch bitsandbytes` and a GPU for actual training; only the config
    object is constructed here (safe / dependency-light to call)."""
    try:
        from peft import LoraConfig, TaskType  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "peft is not installed. `pip install peft transformers torch` "
            "to build a real config; simulate_training() works without it."
        ) from exc

    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
    )


def simulate_training(
    adapter_name: str, cfg: LoRAConfig, epochs: int = 3, dataset_size: int = 512
) -> TrainingResult:
    """Deterministic-ish simulated fine-tuning loop: produces a plausible
    decaying loss curve and trainable-parameter percentage based on the
    LoRA rank, without requiring a GPU or real weights."""
    random.seed(hash((adapter_name, cfg.r)) % (2**32))
    start = time.time()

    base_params_m = 7_000  # pretend 7B base model, in millions of params
    trainable_m = cfg.r * len(cfg.target_modules) * 4096 * 2 / 1_000_000
    trainable_pct = round((trainable_m / base_params_m) * 100, 4)

    loss = 2.8 + random.uniform(-0.1, 0.1)
    curve: List[float] = []
    steps_per_epoch = max(1, dataset_size // 32)
    for _epoch in range(epochs):
        for _step in range(steps_per_epoch):
            loss = max(0.15, loss * random.uniform(0.985, 0.997))
            curve.append(round(loss, 4))

    duration = round(time.time() - start + steps_per_epoch * epochs * 0.001, 3)

    return TrainingResult(
        adapter_name=adapter_name,
        config=cfg,
        epochs=epochs,
        final_loss=curve[-1],
        loss_curve=curve[:: max(1, len(curve) // 20)],  # downsample for display
        trainable_params_pct=trainable_pct,
        duration_sec=duration,
    )


class LoRALab:
    def __init__(self) -> None:
        self._registry: Dict[str, TrainingResult] = {}

    def train(self, adapter_name: str, cfg: LoRAConfig, epochs: int = 3) -> TrainingResult:
        result = simulate_training(adapter_name, cfg, epochs=epochs)
        self._registry[adapter_name] = result
        return result

    def list_adapters(self) -> Dict[str, TrainingResult]:
        return dict(self._registry)


lora_lab_singleton = LoRALab()
