# General Bots Models (Python) - Project Guidelines

**Version:** 1.0.0
**Role:** AI Inference Service for BotServer
**Primary Directive:** Provide access to the latest open-source AI models (Python ecosystem) that are impractical to implement in Rust.

---

## 🐍 PHILOSOPHY & SCOPE

### Why Python?
While `botserver` (Rust) handles the heavy lifting, networking, and business logic, `botmodels` exists solely to leverage the extensive **Python AI/ML ecosystem**.

- **Rust vs. Python Rule**:
  - If logic is deterministic, systems-level, or performance-critical logic: **Do it in Rust (botserver)**.
  - If logic requires cutting-edge ML models, rapid experimentation with HuggingFace, or specific Python-only libraries: **Do it here**.

### Architecture
- **Inference Only**: This service should NOT hold business state. It accepts inputs, runs inference, and returns predictions.
- **Stateless**: Treated as a sidecar to `botserver`.
- **API First**: Exposes strict HTTP/REST endpoints (or gRPC) consumed by `botserver`.

---

## 🛠 TECHNOLOGY STACK

- **Runtime**: Python 3.10+
- **Web Framework**: FastAPI (preferred over Flask for async/performance) or Flask (legacy support).
- **ML Frameworks**: PyTorch, HuggingFace Transformers, raw ONNX (if speed needed).
- **Quality**: `ruff` (linting), `black` (formatting), `mypy` (typing).

---

## ⚡️ IMPERATIVES

### 1. Modern Model Usage
- **Deprecate Legacy**: Move away from outdated libs (e.g., old `allennlp` if superseded) in favor of **HuggingFace Transformers** and **Diffusers**.
- **Quantization**: Always consider quantized models (bitsandbytes, GGUF) to reduce VRAM usage given the "consumer/prosumer" target of General Bots.

### 2. Performance & Loading
- **Lazy Loading**: Do NOT load 10GB models at module import time. Load on startup lifecycle or first request with locking.
- **GPU Handling**: robustly detect CUDA/MPS (Mac) and fallback to CPU gracefully.

### 3. Code Quality
- **Type Hints**: All functions MUST have type hints.
- **Error Handling**: No bare check `except:`. Catch precise exceptions and return structured JSON errors to `botserver`.

---

## 📝 DEVELOPMENT WORKFLOW

1.  **Environment**: Always use a `venv`.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2.  **Running**:
    ```bash
    python app.py
    # OR if migrated to FastAPI
    uvicorn src.main:app --port 8089 --reload
    ```

---

## 🔗 INTEGRATION WITH BOTSERVER

- **Port**: Defaults to `8089` (internal).
- **Security**: Must implement the shared secret handshake (HMAC/API Key) validated against `botserver`.
- **Keep-Alive**: `botserver` manages the lifecycle of this process.

---

## ✅ CONTINUATION PROMPT

When working in `botmodels`:
1.  **Prioritize Ecosystem**: If a new SOTA model drops (e.g., Llama 3, Mistral), enable it here immediately.
2.  **Optimize**: Ensure dependencies are minimized. Don't install `tensorflow` if `torch` suffices.
3.  **Strict Typing**: Ensure all input/outputs match the `botserver` expectations perfectly.
