"""
NeuroForge AI — Streamlit Frontend
====================================================================
A single-file Streamlit UI that talks to the FastAPI backend
(app/main.py) over HTTP and exposes every module through a page:

  1. Chat            -> POST /chat            (router + RAG + MoE + governance)
  2. Dashboard        -> GET  /dashboard       (live governance metrics)
  3. LoRA Lab          -> POST /lora/train, GET /lora/adapters
  4. Quantization      -> POST /quantize
  5. MCP Tool Fabric   -> POST /mcp/call, GET /mcp/permissions/{agent}
  6. Speech (STT demo) -> POST /speech/transcribe

Run:
    # In one terminal — start the API
    uvicorn app.main:app --reload --port 8000

    # In another terminal — start the UI
    streamlit run streamlit_app.py

The backend base URL defaults to http://localhost:8000 and can be
overridden from the sidebar or via the NEUROFORGE_API_URL env var.
"""
from __future__ import annotations

import base64
import os
from typing import Any, Dict, Optional

import requests
import streamlit as st

# --------------------------------------------------------------------------
# Page config & session state
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroForge AI",
    page_icon="🧠",
    layout="wide",
)

DEFAULT_API_URL = os.environ.get("NEUROFORGE_API_URL", "http://localhost:8000")

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: role, content, meta
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-session"

DOMAINS = ["coding", "research", "reasoning", "security", "analytics"]
TOOLS_BY_DOMAIN = {
    "research": {"document_store": ["READ"], "search": ["SEARCH"], "database": ["READ"]},
    "coding": {"github": ["READ"], "document_store": ["READ", "WRITE"]},
    "reasoning": {"document_store": ["READ"], "search": ["SEARCH"]},
    "security": {"monitoring": ["READ"], "database": ["READ"]},
    "analytics": {"monitoring": ["READ"], "database": ["READ"]},
}

# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------
def _base_url() -> str:
    return st.session_state.api_url.rstrip("/")


def api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resp = requests.get(f"{_base_url()}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, json_body: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resp = requests.post(f"{_base_url()}{path}", json=json_body, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def safe_call(fn, *args, **kwargs):
    """Run an API call, surface a clean error in the UI on failure."""
    try:
        return fn(*args, **kwargs), None
    except requests.exceptions.ConnectionError:
        return None, f"Couldn't reach the API at {_base_url()}. Is uvicorn running?"
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            detail = exc.response.text[:300]
        return None, f"API error ({exc.response.status_code}): {detail}"
    except requests.exceptions.Timeout:
        return None, "Request timed out."
    except Exception as exc:  # noqa: BLE001
        return None, f"Unexpected error: {exc}"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 NeuroForge AI")
    st.caption("Adaptive Multimodal Agent Intelligence Platform")

    st.session_state.api_url = st.text_input(
        "Backend API URL", value=st.session_state.api_url, help="FastAPI base URL (uvicorn app.main:app)"
    )

    health, err = safe_call(api_get, "/health")
    if err:
        st.error("Backend offline")
        st.caption(err)
    else:
        st.success(f"Backend online — {health.get('service', 'neuroforge-ai')}")
        provider = health.get("llm_provider", "offline")
        model = health.get("llm_model")
        rag_backend = health.get("rag_backend", "tfidf-memory")

        if provider == "offline":
            st.caption("💬 LLM: offline fallback (no API key set)")
        else:
            st.caption(f"💬 LLM: **{provider}** (`{model}`)")

        if rag_backend == "pinecone":
            st.caption("📚 RAG: **Pinecone** (real vector DB)")
        else:
            st.caption("📚 RAG: in-memory TF-IDF (no Pinecone key set)")

    st.divider()
    page = st.radio(
        "Module",
        [
            "💬 Chat",
            "📊 Dashboard",
            "🧬 LoRA Lab",
            "🗜️ Quantization",
            "🛠️ MCP Tool Fabric",
            "🎙️ Speech (STT demo)",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    with st.expander("What's real vs. simulated"):
        st.markdown(
            "- **Router**: real TF-IDF + logistic regression\n"
            "- **RAG**: real TF-IDF vector store + cosine similarity\n"
            "- **Governance**: real PII / prompt-injection / hallucination checks\n"
            "- **MoE experts**: real Anthropic calls if `ANTHROPIC_API_KEY` is set, else deterministic fallback\n"
            "- **LoRA / Quantization**: real config builders, simulated training/inference (no GPU)\n"
            "- **Speech**: mock STT/TTS (no real audio codec)\n"
            "- **MCP tools**: real permission enforcement, mock tool backends"
        )

# --------------------------------------------------------------------------
# 💬 Chat
# --------------------------------------------------------------------------
if page == "💬 Chat":
    st.header("💬 Chat")
    st.caption("Full pipeline: route → RAG (if needed) → expert dispatch → governance → optional TTS")

    col_session, col_voice, col_clear = st.columns([2, 1, 1])
    with col_session:
        st.session_state.session_id = st.text_input("Session ID", value=st.session_state.session_id)
    with col_voice:
        voice = st.toggle("Voice reply (TTS)", value=False)
    with col_clear:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            meta = turn.get("meta")
            if meta:
                with st.expander("Details"):
                    r = meta["routing"]
                    st.markdown(
                        f"**Domain:** `{r['domain']}` &nbsp;|&nbsp; "
                        f"**Expert:** `{r['expert']}` &nbsp;|&nbsp; "
                        f"**Complexity:** `{r['complexity']}` &nbsp;|&nbsp; "
                        f"**Confidence:** {r['confidence']:.2f} &nbsp;|&nbsp; "
                        f"**Adapter:** `{r['adapter']}`"
                    )
                    if r.get("required_tools"):
                        st.markdown(f"**Tools used:** {', '.join(r['required_tools'])}")

                    if meta.get("rag"):
                        rag = meta["rag"]
                        st.markdown("**Retrieved context:**")
                        for chunk in rag["chunks"]:
                            st.markdown(f"- `{chunk['source']}` (score {chunk['score']:.2f}): {chunk['text']}")

                    gov = meta["governance"]
                    hall = gov["hallucination"]
                    g1, g2, g3, g4 = st.columns(4)
                    g1.metric("Risk level", hall["risk_level"])
                    g2.metric("Grounding score", f"{hall['grounding_score']:.2f}")
                    g3.metric("Latency (ms)", f"{gov['latency_ms']:.0f}")
                    g4.metric("Est. cost ($)", f"{gov['estimated_cost_usd']:.4f}")
                    if gov.get("pii_detected"):
                        st.warning(f"PII detected: {', '.join(gov['pii_detected'])}")
                    if gov.get("prompt_injection_detected"):
                        st.warning("Prompt injection heuristics triggered.")
                    if hall.get("unsupported_claims"):
                        st.info("Unsupported claims: " + "; ".join(hall["unsupported_claims"]))
                    if gov.get("blocked"):
                        st.error(f"Blocked — reason: {gov.get('block_reason')}")

                    if meta.get("audio_url"):
                        st.caption(f"TTS pseudo-URL: `{meta['audio_url']}`")

    prompt = st.chat_input("Ask NeuroForge something…")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Routing → retrieving → dispatching → checking…"):
                data, err = safe_call(
                    api_post,
                    "/chat",
                    {
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                        "voice": voice,
                        "metadata": {},
                    },
                )
            if err:
                st.error(err)
                st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ {err}"})
            else:
                st.markdown(data["answer"])
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "meta": {
                            "routing": data["routing"],
                            "rag": data.get("rag"),
                            "governance": data["governance"],
                            "audio_url": data.get("audio_url"),
                        },
                    }
                )
        st.rerun()

# --------------------------------------------------------------------------
# 📊 Dashboard
# --------------------------------------------------------------------------
elif page == "📊 Dashboard":
    st.header("📊 Governance Dashboard")
    st.caption("Live, in-process metrics accumulated from real requests handled by this API instance.")

    if st.button("🔄 Refresh"):
        st.rerun()

    data, err = safe_call(api_get, "/dashboard")
    if err:
        st.error(err)
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total requests", data["requests_total"])
        c2.metric("Avg latency (ms)", data["avg_latency_ms"])
        c3.metric("Tokens used", data["tokens_total"])
        c4.metric("Est. cost ($)", f"{data['estimated_cost_usd']:.4f}")

        c5, c6, c7 = st.columns(3)
        c5.metric("Hallucination rate (%)", data["hallucination_rate"])
        c6.metric("Blocked requests", data["blocked_requests"])
        c7.metric("PII incidents", data["pii_incidents"])

        st.metric("Tool calls made", data["tool_calls"])

        if data["requests_total"] == 0:
            st.info("No traffic yet — send a message in **Chat** to populate these metrics.")

# --------------------------------------------------------------------------
# 🧬 LoRA Lab
# --------------------------------------------------------------------------
elif page == "🧬 LoRA Lab":
    st.header("🧬 LoRA Model Lab")
    st.caption("Real `peft.LoraConfig` builder; training is a NumPy loss-curve simulator (no GPU needed).")

    with st.form("lora_train_form"):
        col1, col2 = st.columns(2)
        with col1:
            adapter_name = st.text_input("Adapter name", value="my-adapter")
            base_model = st.selectbox(
                "Base model", ["llama-3-8b", "llama-3-70b", "mistral-7b", "qwen2-7b"], index=0
            )
        with col2:
            r = st.slider("LoRA rank (r)", min_value=1, max_value=64, value=8)
            epochs = st.slider("Epochs", min_value=1, max_value=20, value=3)
        submitted = st.form_submit_button("🚀 Train adapter", use_container_width=True)

    if submitted:
        if not adapter_name.strip():
            st.error("Adapter name is required.")
        else:
            with st.spinner("Training (simulated)…"):
                data, err = safe_call(
                    api_post,
                    "/lora/train",
                    params={
                        "adapter_name": adapter_name,
                        "base_model": base_model,
                        "r": r,
                        "epochs": epochs,
                    },
                )
            if err:
                st.error(err)
            else:
                st.success(f"Trained adapter `{data['adapter_name']}` in {data['duration_sec']:.2f}s")
                m1, m2 = st.columns(2)
                m1.metric("Final loss", f"{data['final_loss']:.4f}")
                m2.metric("Trainable params (%)", f"{data['trainable_params_pct']:.3f}")
                if data.get("loss_curve_sample"):
                    st.line_chart(data["loss_curve_sample"])

    st.divider()
    st.subheader("Trained adapters")
    if st.button("🔄 Refresh adapter list"):
        st.rerun()
    adapters, err = safe_call(api_get, "/lora/adapters")
    if err:
        st.error(err)
    elif not adapters:
        st.info("No adapters trained yet in this session.")
    else:
        rows = [
            {
                "adapter": name,
                "final_loss": round(v["final_loss"], 4),
                "trainable_params_pct": round(v["trainable_params_pct"], 3),
                "epochs": v["epochs"],
            }
            for name, v in adapters.items()
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# 🗜️ Quantization
# --------------------------------------------------------------------------
elif page == "🗜️ Quantization":
    st.header("🗜️ Quantization Engine")
    st.caption("Real `BitsAndBytesConfig` builder; size/speedup estimates use cited empirical ratios.")

    with st.form("quantize_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            model_name = st.text_input("Model name", value="llama-3-8b")
        with col2:
            param_count_m = st.number_input(
                "Parameter count (millions)", min_value=1.0, value=8000.0, step=100.0
            )
        with col3:
            target_precision = st.selectbox("Target precision", ["fp16", "int8", "int4"], index=1)
        submitted = st.form_submit_button("⚙️ Run analysis", use_container_width=True)

    if submitted:
        data, err = safe_call(
            api_post,
            "/quantize",
            params={
                "model_name": model_name,
                "param_count_m": param_count_m,
                "target_precision": target_precision,
            },
        )
        if err:
            st.error(err)
        else:
            st.success(f"Quantization report for `{model_name}` → `{target_precision}`")
            st.json(data)

# --------------------------------------------------------------------------
# 🛠️ MCP Tool Fabric
# --------------------------------------------------------------------------
elif page == "🛠️ MCP Tool Fabric":
    st.header("🛠️ MCP Tool Fabric")
    st.caption("Permissioned tool registry — deny-by-default outside each agent's declared permissions.")

    agent = st.selectbox("Agent (domain)", DOMAINS)

    perms, err = safe_call(api_get, f"/mcp/permissions/{agent}")
    if err:
        st.error(err)
        perms = {"permissions": {}}
    else:
        st.markdown(f"**Permissions for `{agent}`:**")
        if perms["permissions"]:
            for tool, actions in perms["permissions"].items():
                st.markdown(f"- `{tool}` → {', '.join(actions)}")
        else:
            st.info("No permissions declared for this agent.")

    st.divider()
    st.subheader("Invoke a tool")

    tool_options = list(perms["permissions"].keys()) or ["database", "document_store", "search", "github", "monitoring"]
    col1, col2 = st.columns(2)
    with col1:
        tool = st.selectbox("Tool", tool_options)
    with col2:
        allowed_actions = perms["permissions"].get(tool, ["READ", "WRITE", "SEARCH", "DELETE"])
        action = st.selectbox("Action", allowed_actions)

    st.caption("Payload (JSON) — e.g. `{\"key\": \"user:1\"}` for database READ")
    payload_str = st.text_area("Payload", value='{"key": "user:1"}', height=80)

    if st.button("▶️ Call tool", use_container_width=True):
        import json

        try:
            payload = json.loads(payload_str) if payload_str.strip() else {}
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON payload: {exc}")
            payload = None

        if payload is not None:
            data, err = safe_call(
                api_post,
                "/mcp/call",
                {"agent": agent, "tool": tool, "action": action, "payload": payload},
            )
            if err:
                st.error(err)
            elif data["allowed"]:
                st.success("Call allowed")
                st.json(data["result"])
            else:
                st.error(f"Call denied — {data['reason']}")

# --------------------------------------------------------------------------
# 🎙️ Speech (STT demo)
# --------------------------------------------------------------------------
elif page == "🎙️ Speech (STT demo)":
    st.header("🎙️ Speech Intelligence — STT demo")
    st.caption(
        "This demo backend treats UTF-8 text encoded as bytes as the 'audio' payload — "
        "swap `stt_singleton` for Whisper/etc. for real audio. Type text below to simulate a transcription call."
    )

    text_input = st.text_area("Text to send as pseudo-audio", value="Hello NeuroForge, this is a test.", height=100)

    if st.button("🎤 Transcribe", use_container_width=True):
        audio_b64 = base64.b64encode(text_input.encode("utf-8")).decode("ascii")
        data, err = safe_call(api_post, "/speech/transcribe", {"audio_base64": audio_b64})
        if err:
            st.error(err)
        else:
            st.success("Transcription result")
            c1, c2 = st.columns(2)
            c1.metric("Confidence", f"{data['confidence']:.2f}")
            c2.metric("Duration (sec)", f"{data['duration_sec']:.2f}")
            st.markdown(f"**Text:** {data['text']}")
