from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx, os, time, json
from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com/chat/completions")
MODEL = os.getenv("MODEL", "deepseek-v4-flash")
API_BIBLE_KEY = os.getenv("API_BIBLE_KEY", "")

app = FastAPI(title="God Assistant Proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "service": "god-assistant-proxy"}

@app.get("/api/bible-key")
async def bible_key():
    """Supply the API.Bible key to iOS clients. No auth required — key is not sensitive enough to gate."""
    if not API_BIBLE_KEY:
        raise HTTPException(status_code=500, detail="Server missing API_BIBLE_KEY")
    return {"key": API_BIBLE_KEY}

@app.post("/api/chat")
@app.post("/v1/chat/completions")
async def chat_completions(payload: dict):
    started = time.monotonic()
    if not DEEPSEEK_KEY:
        raise HTTPException(status_code=500, detail="Server missing DEEPSEEK_API_KEY")

    model = payload.get("model", MODEL)
    temperature = payload.get("temperature", 0.8)
    raw = payload.get("max_tokens", 32000)
    # Enforce a floor of 8000 tokens to prevent empty responses in thinking mode
    max_tokens = max(raw, 8000) if raw is not None else 32000
    messages = payload.get("messages", [])
    thinking = payload.get("thinking", {"type": "disabled"})
    reasoning_effort = payload.get("reasoning_effort", None)
    is_stream = payload.get("stream", False)

    upstream_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": is_stream,
        "thinking": thinking,
        "reasoning_effort": reasoning_effort,
    }
    print(f"[PROXY][REQUEST] model={model} max_tokens={max_tokens} messages={len(messages)} stream={is_stream}", flush=True)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            if is_stream:
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
                        print(f"[PROXY][STREAM] status={resp.status_code} elapsed_ms={elapsed_ms}", flush=True)
                        if resp.status_code < 200 or resp.status_code >= 300:
                            body = await resp.aread()
                            print(f"[PROXY][ERROR] status={resp.status_code} body={body[:500]}", flush=True)
                            yield json.dumps({"error": f"Upstream error {resp.status_code}"}).encode()
                            return
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                return StreamingResponse(generate(), media_type="text/event-stream")
            else:
                r = await client.post(
                    DEEPSEEK_BASE,
                    headers={
                        "Authorization": f"Bearer {DEEPSEEK_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=upstream_payload,
                )
                elapsed_ms = int((time.monotonic() - started) * 1000)
                print(f"[PROXY][RESPONSE] status={r.status_code} bytes={len(r.content)} elapsed_ms={elapsed_ms}", flush=True)
                if r.status_code < 200 or r.status_code >= 300:
                    raise HTTPException(status_code=r.status_code, detail=f"DeepSeek error: {r.text[:500]}")
                return r.json()
        except httpx.RequestError as e:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            print(f"[PROXY][REQUEST_ERROR] elapsed_ms={elapsed_ms} error={e}", flush=True)
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e
