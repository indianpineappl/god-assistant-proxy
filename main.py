from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx, os, time, json
from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com/chat/completions")
MODEL = os.getenv("MODEL", "deepseek-v4-flash")
DEV_API_TOKEN = os.getenv("DEV_API_TOKEN")

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

# Accept both /api/chat (iOS client) and /v1/chat/completions (standard)
@app.post("/api/chat")
@app.post("/v1/chat/completions")
async def chat_completions(payload: dict, authorization: str | None = Header(default=None)):
    started = time.monotonic()
    if not DEEPSEEK_KEY:
        raise HTTPException(status_code=500, detail="Server missing DEEPSEEK_API_KEY")

    # Simple dev auth
    if DEV_API_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Authorization")
        token = authorization.split(" ", 1)[1]
        if token != DEV_API_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid token")

    model = payload.get("model", MODEL)
    temperature = payload.get("temperature", 0.2)
    max_tokens = payload.get("max_tokens", 32000)
    messages = payload.get("messages", [])
    thinking = payload.get("thinking", {"type": "enabled"})
    reasoning_effort = payload.get("reasoning_effort", "low")

    # Client sends the complete system prompt as a system role message.
    # Backend just forwards as-is — no duplicate system prompt appended.
    final_messages = messages
    upstream_payload = {
        "model": model,
        "messages": final_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "reasoning_effort": reasoning_effort,
    }
    request_bytes = len(json.dumps(upstream_payload).encode("utf-8"))
    print(f"[ASSISTANT_PROXY][REQUEST] model={model} max_tokens={max_tokens} messages={len(final_messages)} request_bytes={request_bytes} thinking={thinking} reasoning_effort={reasoning_effort}", flush=True)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
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
