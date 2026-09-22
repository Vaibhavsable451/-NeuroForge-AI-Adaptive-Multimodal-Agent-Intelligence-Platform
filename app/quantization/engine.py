"""
Quantization Engine
----------------------
Simulates FP16 -> INT8 -> INT4 post-training quantization: estimates
memory footprint reduction, inference speedup, and accuracy delta using
well-known empirical rules of thumb. Provides the real `bitsandbytes`
config path for when actual GPU quantization is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

Precision = Literal["fp16", "int8", "int4"]

_BYTES_PER_PARAM = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}
# Rough, widely-cited empirical accuracy drop vs fp16 baseline.
_ACCURACY_DROP_PCT = {"fp16": 0.0, "int8": 0.6, "int4": 1.8}
# Rough empirical inference speedup vs fp16 on comparable hardware.
_SPEEDUP_FACTOR = {"fp16": 1.0, "int8": 1.6, "int4": 2.4}


@dataclass
class QuantizationReport:
    model_name: str
    param_count_m: float
    source_precision: Precision
    target_precision: Precision
    source_size_gb: float
    target_size_gb: float
    size_reduction_pct: float
    estimated_speedup_x: float
    estimated_accuracy_drop_pct: float


def quantize(
    model_name: str, param_count_m: float, target_precision: Precision, source_precision: Precision = "fp16"
) -> QuantizationReport:
    src_gb = (param_count_m * 1_000_000 * _BYTES_PER_PARAM[source_precision]) / (1024**3)
    tgt_gb = (param_count_m * 1_000_000 * _BYTES_PER_PARAM[target_precision]) / (1024**3)
    reduction = round((1 - tgt_gb / src_gb) * 100, 2) if src_gb else 0.0

    return QuantizationReport(
        model_name=model_name,
        param_count_m=param_count_m,
        source_precision=source_precision,
        target_precision=target_precision,
        source_size_gb=round(src_gb, 3),
        target_size_gb=round(tgt_gb, 3),
        size_reduction_pct=reduction,
        estimated_speedup_x=_SPEEDUP_FACTOR[target_precision],
        estimated_accuracy_drop_pct=_ACCURACY_DROP_PCT[target_precision],
    )


def build_real_bnb_config(target_precision: Precision):
    """Builds a real `transformers.BitsAndBytesConfig`. Requires
    `pip install bitsandbytes transformers` and a CUDA GPU to actually load
    a quantized model; only the config object is constructed here."""
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "transformers/bitsandbytes not installed. "
            "`pip install transformers bitsandbytes` for real quantized loading."
        ) from exc

    if target_precision == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if target_precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError("build_real_bnb_config only supports int8/int4 targets.")


class QuantizationEngine:
    def __init__(self) -> None:
        self._history: Dict[str, QuantizationReport] = {}

    def run(self, model_name: str, param_count_m: float, target_precision: Precision) -> QuantizationReport:
        report = quantize(model_name, param_count_m, target_precision)
        self._history[f"{model_name}:{target_precision}"] = report
        return report

    def history(self) -> Dict[str, QuantizationReport]:
        return dict(self._history)


quantization_engine_singleton = QuantizationEngine()
