"""
Shared Pydantic schemas for NeuroForge AI.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, Field

# ============================================================
# SHARED RAG SCHEMAS
# ============================================================

from app.rag.schemas import RAGResult


# ============================================================
# DOMAIN
# ============================================================

class Domain(str, Enum):
    CODING = "coding"
    RESEARCH = "research"
    REASONING = "reasoning"
    SECURITY = "security"
    ANALYTICS = "analytics"


# ============================================================
# COMPLEXITY
# ============================================================

class Complexity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ============================================================
# RISK LEVEL
# ============================================================

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    BLOCKED = "BLOCKED"


# ============================================================
# CHAT REQUEST
# ============================================================

class ChatRequest(BaseModel):
    session_id: str = Field(default="default")
    message: str
    voice: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# ROUTING DECISION
# ============================================================

class RoutingDecision(BaseModel):
    domain: Domain
    complexity: Complexity
    required_tools: List[str]
    expert: Domain
    confidence: float
    adapter: str


# ============================================================
# HALLUCINATION REPORT
# ============================================================

class HallucinationReport(BaseModel):
    hallucination_score: float
    grounding_score: float
    citations_verified: bool
    risk_level: RiskLevel
    unsupported_claims: List[str] = Field(default_factory=list)
    regenerated: bool = False


# ============================================================
# GOVERNANCE REPORT
# ============================================================

class GovernanceReport(BaseModel):
    pii_detected: List[str]
    prompt_injection_detected: bool
    hallucination: HallucinationReport
    tokens_used: int
    estimated_cost_usd: float
    latency_ms: float
    blocked: bool
    block_reason: Optional[str] = None


# ============================================================
# CHAT RESPONSE
# ============================================================

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    routing: RoutingDecision

    # IMPORTANT:
    # RAGResult comes from app.rag.schemas.
    # There is NO local RAGResult definition here.
    rag: Optional[RAGResult] = None

    governance: GovernanceReport
    audio_url: Optional[str] = None

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ============================================================
# MCP TOOL REQUEST
# ============================================================

class MCPToolCallRequest(BaseModel):
    agent: Domain
    tool: str
    action: str
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("payload", "input"),
    )


# ============================================================
# MCP TOOL RESULT
# ============================================================

class MCPToolCallResult(BaseModel):
    allowed: bool
    tool: str
    action: str
    result: Optional[Any] = None
    reason: Optional[str] = None


# ============================================================
# DASHBOARD METRICS
# ============================================================

class DashboardMetrics(BaseModel):
    requests_total: int
    avg_latency_ms: float
    tokens_total: int
    estimated_cost_usd: float
    hallucination_rate: float
    blocked_requests: int
    pii_incidents: int
    tool_calls: int