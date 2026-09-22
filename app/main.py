"""
NeuroForge AI — FastAPI Application Entry Point
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.dashboard.metrics import metrics_singleton
from app.governance.firewall import evaluate
from app.llm_client import LLMClient
from app.lora_lab.trainer import LoRAConfig, lora_lab_singleton
from app.mcp_fabric.registry import mcp_singleton
from app.models import (
    ChatRequest,
    ChatResponse,
    Complexity,
    DashboardMetrics,
    Domain,
    MCPToolCallRequest,
    MCPToolCallResult,
)
from app.moe.experts import moe_singleton
from app.quantization.engine import quantization_engine_singleton
from app.rag.pipeline import rag_singleton
from app.router.neural_router import router_singleton
from app.speech.pipeline import (
    decode_base64_audio,
    stt_singleton,
    tts_singleton,
)

app = FastAPI(
    title="NeuroForge AI",
    description="Adaptive Multimodal Agent Intelligence Platform API",
    version="1.0.0",
)


class TranscribeRequest(BaseModel):
    audio_base64: str


@app.get("/health")
def health_check() -> Dict[str, Any]:
    llm = LLMClient()
    return {
        "status": "ok",
        "service": "neuroforge-ai",
        "llm_provider": llm.provider or "offline",
        "llm_model": llm.model,
        "rag_backend": rag_singleton.backend,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    start_time = time.time()

    # 1. Adaptive Neural Router
    routing = router_singleton.route(req.message)

    # 2. Agentic RAG
    rag_result = None
    if "rag_retrieval" in routing.required_tools or routing.complexity == Complexity.HIGH:
        rag_result = rag_singleton.retrieve(req.message)

    # 3. MoE Expert Dispatch
    answer = moe_singleton.dispatch(
        domain=routing.domain,
        message=req.message,
        rag=rag_result,
    )

    # 4. Governance & Firewall evaluation
    latency_ms = (time.time() - start_time) * 1000.0
    gov_report, _ = evaluate(req.message, answer, rag_result, latency_ms)

    # 5. Optional Speech (TTS)
    audio_url: Optional[str] = None
    if req.voice:
        synth = tts_singleton.synthesize(answer)
        audio_url = synth.audio_url

    # 6. Metrics Recording
    metrics_singleton.record_request(gov_report)

    return ChatResponse(
        session_id=req.session_id,
        answer=answer,
        routing=routing,
        rag=rag_result,
        governance=gov_report,
        audio_url=audio_url,
    )


@app.get("/dashboard", response_model=DashboardMetrics)
def get_dashboard() -> DashboardMetrics:
    return metrics_singleton.snapshot()


@app.post("/lora/train")
def train_lora(
    adapter_name: str,
    base_model: str = "llama-3-8b",
    r: int = 8,
    epochs: int = 3,
) -> Dict[str, Any]:
    cfg = LoRAConfig(base_model=base_model, r=r)
    result = lora_lab_singleton.train(adapter_name, cfg, epochs=epochs)
    return {
        "adapter_name": result.adapter_name,
        "config": {
            "base_model": result.config.base_model,
            "r": result.config.r,
        },
        "epochs": result.epochs,
        "final_loss": result.final_loss,
        "loss_curve_sample": result.loss_curve,
        "trainable_params_pct": result.trainable_params_pct,
        "duration_sec": result.duration_sec,
    }


@app.get("/lora/adapters")
def list_lora_adapters() -> Dict[str, Any]:
    adapters = lora_lab_singleton.list_adapters()
    return {
        name: {
            "adapter_name": res.adapter_name,
            "final_loss": res.final_loss,
            "trainable_params_pct": res.trainable_params_pct,
            "epochs": res.epochs,
            "duration_sec": res.duration_sec,
        }
        for name, res in adapters.items()
    }


@app.post("/quantize")
def quantize_model(
    model_name: str,
    param_count_m: float,
    target_precision: str,
) -> Dict[str, Any]:
    if target_precision not in ("fp16", "int8", "int4"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target precision: '{target_precision}'. Must be fp16, int8, or int4.",
        )

    report = quantization_engine_singleton.run(
        model_name=model_name,
        param_count_m=param_count_m,
        target_precision=target_precision,  # type: ignore
    )
    return {
        "model_name": report.model_name,
        "param_count_m": report.param_count_m,
        "source_precision": report.source_precision,
        "target_precision": report.target_precision,
        "source_size_gb": report.source_size_gb,
        "target_size_gb": report.target_size_gb,
        "size_reduction_pct": report.size_reduction_pct,
        "estimated_speedup_x": report.estimated_speedup_x,
        "estimated_accuracy_drop_pct": report.estimated_accuracy_drop_pct,
    }


@app.post("/mcp/call", response_model=MCPToolCallResult)
def call_mcp_tool(req: MCPToolCallRequest) -> MCPToolCallResult:
    res = mcp_singleton.call(
        agent=req.agent,
        tool=req.tool,
        action=req.action,
        payload=req.payload,
    )
    if res.allowed:
        metrics_singleton.record_tool_call()
    return res


@app.get("/mcp/permissions/{agent_name}")
def get_mcp_permissions(agent_name: str) -> Dict[str, Any]:
    try:
        agent_domain = Domain(agent_name.lower())
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown agent domain: '{agent_name}'.",
        )

    perms = mcp_singleton.permissions_for(agent_domain)
    return {
        "agent": agent_domain.value,
        "permissions": perms,
    }


@app.post("/speech/transcribe")
def transcribe_speech(req: TranscribeRequest) -> Dict[str, Any]:
    try:
        audio_bytes = decode_base64_audio(req.audio_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    result = stt_singleton.transcribe(audio_bytes)
    return {
        "text": result.text,
        "confidence": result.confidence,
        "duration_sec": result.duration_sec,
    }