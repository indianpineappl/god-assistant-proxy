from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx, os, time, json
from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com/chat/completions")
MODEL = os.getenv("MODEL", "deepseek-v4-flash")

app = FastAPI(title="God Assistant Proxy (Dev)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "service": "god-assistant-proxy-dev"}

# Bible API key endpoint for client-side ScriptureAPIBibleClient
API_BIBLE_KEY = os.getenv("API_BIBLE_KEY", "")

@app.get("/api/bible-key")
async def bible_key():
    """Supply the API.Bible key to clients."""
    if not API_BIBLE_KEY:
        raise HTTPException(status_code=500, detail="Server missing API_BIBLE_KEY")
    return {"key": API_BIBLE_KEY}

# Accept both /api/chat (iOS client) and /v1/chat/completions (standard)
@app.post("/api/chat")
@app.post("/v1/chat/completions")
async def chat_completions(payload: dict):
    started = time.monotonic()
    if not DEEPSEEK_KEY:
        raise HTTPException(status_code=500, detail="Server missing DEEPSEEK_API_KEY")

    model = payload.get("model", MODEL)
    temperature = payload.get("temperature", 0.2)
    # Enforce a floor of 8000 tokens — anything lower causes thinking mode to
    # consume the entire budget on reasoning, yielding zero/partial responses.
    raw = payload.get("max_tokens", 32000)
    if raw is not None and raw < 8000:
        print(f"[ASSISTANT_PROXY][WARN] client sent max_tokens={raw} — raising to 8000 to prevent empty responses", flush=True)
        max_tokens = 8000
    else:
        max_tokens = raw
    messages = payload.get("messages", [])
    # Disable thinking to prevent reasoning from consuming token budget
    thinking = payload.get("thinking", {"type": "disabled"})
    reasoning_effort = payload.get("reasoning_effort", None)

    # Client sends the complete system prompt as a system role message.
    # Backend just forwards as-is — no duplicate system prompt appended.
    final_messages = messages
    is_stream = payload.get("stream", False)
    upstream_payload = {
        "model": model,
        "messages": final_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": is_stream,
        "thinking": thinking,
        "reasoning_effort": reasoning_effort,
    }
    request_bytes = len(json.dumps(upstream_payload).encode("utf-8"))
    print(f"[ASSISTANT_PROXY][REQUEST] model={model} max_tokens={max_tokens} messages={len(final_messages)} request_bytes={request_bytes} thinking={thinking} reasoning_effort={reasoning_effort} stream={is_stream}", flush=True)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            if is_stream:
                # Streaming: pipe SSE chunks directly to the client
                async def generate():
                    async with client.stream(
                        "POST",
                        DEEPSEEK_BASE,
                        headers={
                            "Authorization": f"Bearer {DEEPSEEK_KEY}",
                            "Content-Type": "application/json",
                        },
                        json=upstream_payload,
                    ) as resp:
                        elapsed_ms = int((time.monotonic() - started) * 1000)
                        print(f"[ASSISTANT_PROXY][STREAM] status={resp.status_code} elapsed_ms={elapsed_ms}", flush=True)
                        if resp.status_code < 200 or resp.status_code >= 300:
                            body = await resp.aread()
                            print(f"[ASSISTANT_PROXY][ERROR] status={resp.status_code} body={body[:2000]}", flush=True)
                            yield json.dumps({"error": f"Upstream error {resp.status_code}"}).encode()
                            return
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                return StreamingResponse(generate(), media_type="text/event-stream")
            else:
                # Non-streaming: buffer and return JSON
                r = await client.post(
                    DEEPSEEK_BASE,
                    headers={
                        "Authorization": f"Bearer {DEEPSEEK_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=upstream_payload,
                )
                elapsed_ms = int((time.monotonic() - started) * 1000)
                print(f"[ASSISTANT_PROXY][RESPONSE] status={r.status_code} bytes={len(r.content)} elapsed_ms={elapsed_ms}", flush=True)
                if r.status_code < 200 or r.status_code >= 300:
                    print(f"[ASSISTANT_PROXY][ERROR] body={r.text[:2000]}", flush=True)
                    raise HTTPException(status_code=r.status_code, detail=f"DeepSeek error {r.status_code}: {r.text}")
                decoded = r.json()
                choice = (decoded.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                usage = decoded.get("usage") or {}
                content = message.get("content") or ""
                print(f"[ASSISTANT_PROXY][DEEPSEEK] finish_reason={choice.get('finish_reason')} content_chars={len(content)} reasoning_chars={len(message.get('reasoning_content') or '')} usage={usage}", flush=True)
                return decoded
        except httpx.RequestError as e:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            print(f"[ASSISTANT_PROXY][REQUEST_ERROR] elapsed_ms={elapsed_ms} error={e}", flush=True)
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e
