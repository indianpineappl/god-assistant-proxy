from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx, os
from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com/chat/completions")
MODEL = os.getenv("MODEL", "deepseek-chat")
DEV_API_TOKEN = os.getenv("DEV_API_TOKEN")

app = FastAPI(title="God Assistant Proxy (Dev)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "service": "god-assistant-proxy-dev"}

@app.post("/v1/chat/completions")
async def chat_completions(payload: dict, authorization: str | None = Header(default=None)):
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
    max_tokens = payload.get("max_tokens", 800)
    messages = payload.get("messages", [])
    meta = payload.get("meta", {})

    system_prompt = build_system_prompt(meta)
    final_messages = ([{"role": "system", "content": system_prompt}] + messages)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.post(
                DEEPSEEK_BASE,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": final_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            if r.status_code < 200 or r.status_code >= 300:
                raise HTTPException(status_code=r.status_code, detail=f"DeepSeek error {r.status_code}: {r.text}")
            return r.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e

def build_system_prompt(meta: dict) -> str:
    lines = [
        "You are 'God Assistant', a Bible-only guide.",
        "Your role: answer questions and provide guidance strictly grounded in the Holy Bible.",
        "Rules:",
        "1) Stay within Biblical context only; do not discuss unrelated topics.",
        "2) Prefer the user's denomination defaults and selected translation.",
        "3) Cite passages by Book Chapter:Verse alongside explanations.",
        "4) If uncertain or the Bible is silent, say so and suggest relevant passages.",
        "5) Avoid medical, legal, or financial advice beyond Biblical teaching.",
        "6) Be pastoral, concise, and faithful to Scripture; avoid speculation.",
    ]
    denom = meta.get("denomination")
    translation = meta.get("translation")
    if denom:
        lines.append(f"Denomination: {denom}")
    if translation:
        lines.append(f"Primary Translation: {translation}")
    grounding = meta.get("grounding") or []
    if grounding:
        verses_block = "\n".join([f"- {v.get('book')} {v.get('chapter')}:{v.get('verse')} {v.get('text')}" for v in grounding[:10]])
        lines.append("Grounding Passages (context):\n" + verses_block)
    return "\n".join(lines)
