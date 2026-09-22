# NeuroForge AI — Adaptive Multimodal Agent Intelligence Platform

A runnable FastAPI implementation of the 8-module architecture: **Adaptive
Neural Router → Mixture-of-Experts → LoRA Model Lab → Quantization Engine →
Hybrid Agentic RAG → Speech Intelligence → AI Governance/Hallucination
Firewall → MCP Tool Fabric**.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for interactive Swagger docs, or run
the test suite:

```bash
pytest tests/ -v
```

18/18 tests pass out of the box, with zero API keys or GPU required.

### Streamlit frontend

A single-file Streamlit UI (`streamlit_app.py`) covers every endpoint —
Chat, Governance Dashboard, LoRA Lab, Quantization, MCP Tool Fabric, and
a Speech/STT demo. With the API already running in one terminal:

```bash
streamlit run streamlit_app.py
```

It defaults to `http://localhost:8000`; change it from the sidebar or
set `NEUROFORGE_API_URL` before launching, e.g.:

```bash
NEUROFORGE_API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

### AWS EC2 Direct Deployment

The GitHub Actions CI/CD pipeline (`.github/workflows/ci.yml`) automatically tests, security scans, and deploys main branch updates directly to AWS EC2 over SSH.

To manage the application manually on AWS EC2:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## What's real vs. simulated (read this first)

This is an honest engineering artifact, not a marketing demo — every
module has **real, working logic**, but a few pieces that need a GPU,
cloud credentials, or real audio hardware are implemented as functional
interfaces with a deterministic offline fallback, clearly marked in code:

| Module | What's real | What's simulated / stubbed |
|---|---|---|
| **Adaptive Neural Router** | Real TF-IDF + logistic-regression classifier, trained live on startup; real complexity heuristics | Bootstrap training set is small (24 examples) — swap in real routing logs for production accuracy |
| **Mixture-of-Experts** | Real dispatch logic, real prompt construction, pluggable per-domain experts | Falls back to deterministic template answers unless `GROQ_API_KEY` or `ANTHROPIC_API_KEY` is set (then it calls the real Groq or Anthropic API) |
| **LoRA Model Lab** | Real `peft.LoraConfig` builder (`build_real_peft_config`) | Actual training (`simulate_training`) is a NumPy-based loss-curve simulator — no GPU/base model in this environment. Swap for a real `Trainer` loop when you have compute |
| **Quantization Engine** | Real `BitsAndBytesConfig` builder (`build_real_bnb_config`); size/speedup math uses cited empirical ratios | No real model is loaded/quantized in-process (needs GPU + real weights) |
| **Hybrid Agentic RAG** | Real Pinecone serverless index + hosted embeddings when `PINECONE_API_KEY` is set (real cosine-similarity retrieval), plus real lexical reranking and metadata filtering on top | Falls back to an in-memory TF-IDF store with a 5-document seed corpus when no Pinecone key is set; same `upsert`/`query` interface either way |
| **Speech Intelligence** | Real STT/TTS interface (`SpeechToText`/`TextToSpeech` ABCs) fully wired into `/chat` | Mock implementations (no real audio codec/model) — text-as-bytes for STT, hash-based pseudo-URL for TTS. Swap `stt_singleton`/`tts_singleton` for Whisper/ElevenLabs etc. |
| **AI Governance & Hallucination Firewall** | Real PII regex detection, real prompt-injection heuristics, real claim-extraction → evidence-matching pipeline that actually blocks ungrounded/high-risk answers | Claim/evidence matching uses lexical overlap, not a trained entailment model — swap `_claim_supported` for an NLI model for production accuracy |
| **MCP Tool Fabric** | Real permission enforcement (agent × tool × action allow-list), real deny-by-default on unlisted actions | Tool backends (`database`, `github`, `monitoring`, etc.) are in-memory mocks — same call interface as a real MCP server |
| **Deployment (AWS EC2 CI/CD)** | Automated GitHub Actions CI/CD pipeline (test → bandit security scan → direct SSH push & deployment to AWS EC2) | Configure `EC2_HOST`, `EC2_USERNAME`, and `EC2_SSH_KEY` as GitHub repository secrets |

**Bottom line:** every request through `/chat` really is routed by a
trained classifier, really is retrieved against a real vector index,
really is scored by a working hallucination firewall that will actually
block a response — you can watch it happen in `/dashboard`. The parts
that would need a GPU cluster, cloud account, or microphone to be "real"
are isolated behind clean interfaces so swapping them in production is a
scoped, single-file change per module (called out above).

## API surface

| Endpoint | Purpose |
|---|---|
| `POST /chat` | Full pipeline: route → RAG (if needed) → expert → governance → (optional) TTS |
| `POST /speech/transcribe` | STT demo endpoint |
| `POST /mcp/call` | Invoke a tool through the permissioned MCP fabric |
| `GET /mcp/permissions/{agent}` | Inspect an agent's tool permissions |
| `POST /lora/train` | Kick off a (simulated) LoRA fine-tune, get a loss curve |
| `GET /lora/adapters` | List trained adapters |
| `POST /quantize` | Run FP16→INT8/INT4 quantization analysis |
| `GET /dashboard` | Live governance metrics (requests, latency, cost, hallucination rate, blocks) |

## Project layout

```
app/
  main.py                    FastAPI app wiring all modules together
  models.py                  Shared Pydantic schemas
  llm_client.py               Optional real-Anthropic-API backend
  router/neural_router.py     1. Adaptive Neural Router
  moe/experts.py               2. Mixture-of-Experts Engine
  lora_lab/trainer.py          3. LoRA Model Lab
  quantization/engine.py       4. Quantization Engine
  rag/pipeline.py              5. Hybrid Agentic RAG
  speech/pipeline.py           6. Speech Intelligence
  governance/firewall.py       7. AI Governance & Hallucination Firewall
  mcp_fabric/registry.py       8. MCP Tool Fabric
  dashboard/metrics.py         Governance dashboard aggregator
tests/test_basic.py           18 tests covering every module + endpoint
.github/workflows/ci.yml      test → security scan → direct AWS EC2 push & deploy
```

## Enabling real LLM completions (Groq)

Copy `.env.example` → `.env`, set `GROQ_API_KEY` (get one free at
https://console.groq.com), and `pip install groq` (already in
`requirements.txt`). Every expert in `app/moe/experts.py` will then call
the real Groq API (`llama-3.3-70b-versatile` by default, override with
`GROQ_MODEL`) instead of its offline fallback — no other code changes
needed. Set `ANTHROPIC_API_KEY` instead (or `LLM_PROVIDER=anthropic`) to
use Claude models.

## Enabling a real vector database (Pinecone)

Copy `.env.example` → `.env`, set `PINECONE_API_KEY` (get one free at
https://app.pinecone.io), and `pip install pinecone` (already in
`requirements.txt`). On startup, `app/rag/pipeline.py` will create a
serverless Pinecone index (`neuroforge-ai` by default — override with
`PINECONE_INDEX_NAME`), embed documents with Pinecone's hosted
`multilingual-e5-large` model (no local embedding model or GPU needed),
and route all retrieval through it instead of the in-memory TF-IDF
store. If Pinecone init fails for any reason (bad key, no network), the
app logs a warning and falls back to TF-IDF automatically rather than
crashing. Check which backend is active via `GET /health` (`rag_backend`
field) or the Streamlit sidebar.

## Known limitations (by design, not oversight)

- The router's classifier is trained on a 24-example bootstrap set; it
  will occasionally misclassify ambiguous short prompts (e.g. "define
  recursion"). Replace with real historical routing logs for production.
- The hallucination firewall only flags claims as unsupported when RAG
  context was actually retrieved — an expert answering from a domain
  that doesn't use RAG (e.g. coding) isn't marked "hallucinating" simply
  for being ungrounded, since there's no evidence set to check against.
- Metrics in `/dashboard` are in-process and reset on restart; wire a
  real store (Postgres/CloudWatch) for persistence across deploys.
