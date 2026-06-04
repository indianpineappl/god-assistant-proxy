from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx, os, time, json, datetime
from dotenv import load_dotenv
import jwt
from jwt import PyJWKClient

load_dotenv()
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com/chat/completions")
MODEL = os.getenv("MODEL", "deepseek-v4-flash")
API_BIBLE_KEY = os.getenv("API_BIBLE_KEY", "")
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")

# Apple Identity Token validation
APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
APPLE_ISSUER = "https://appleid.apple.com"
APPLE_AUDIENCE = "Indic.Life-By-Bible"
_apple_jwk_client = PyJWKClient(APPLE_JWKS_URL, cache_keys=True)

# Turso HTTP API helper (no native deps needed)
async def turso_execute(sql: str, args: list = None):
    """Execute SQL via Turso's HTTP API. Returns rows as list of dicts."""
    url = TURSO_DATABASE_URL.replace("libsql://", "https://")
    headers = {"Authorization": f"Bearer {TURSO_AUTH_TOKEN}", "Content-Type": "application/json"}
    
    # Build request body for Turso HTTP API
    stmt = {"type": "execute", "stmt": {"sql": sql}}
    if args:
        stmt["stmt"]["args"] = [{"type": "text", "value": str(a)} if a is not None else {"type": "null"} for a in args]
    
    body = {"requests": [stmt, {"type": "close"}]}
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{url}/v2/pipeline", headers=headers, json=body)
        if resp.status_code != 200:
            print(f"[TURSO] Error: {resp.status_code} {resp.text[:200]}", flush=True)
            raise Exception(f"Turso error: {resp.status_code}")
        
        data = resp.json()
        result = data.get("results", [{}])[0]
        
        if "error" in result:
            raise Exception(f"SQL error: {result['error']}")
        
        response = result.get("response", {}).get("result", {})
        cols = [c["name"] for c in response.get("cols", [])]
        rows_raw = response.get("rows", [])
        
        rows = []
        for row in rows_raw:
            row_dict = {}
            for i, col in enumerate(cols):
                cell = row[i]
                row_dict[col] = cell.get("value") if cell.get("type") != "null" else None
            rows.append(row_dict)
        
        return {
            "rows": rows,
            "rowsAffected": response.get("affected_row_count", 0),
        }

def validate_apple_token(identity_token: str, expected_user_id: str) -> dict:
    """Validate Apple Identity Token. Raises on failure."""
    try:
        signing_key = _apple_jwk_client.get_signing_key_from_jwt(identity_token)
        payload = jwt.decode(
            identity_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=APPLE_AUDIENCE,
            issuer=APPLE_ISSUER,
        )
        if payload.get("sub") != expected_user_id:
            raise ValueError("Token subject does not match user identifier")
        return payload
    except Exception as e:
        raise ValueError(f"Token validation failed: {e}")

async def authenticate_request(authorization: str | None, x_apple_user_id: str | None) -> str:
    """Validate auth headers. Returns userId or raises HTTPException."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not x_apple_user_id:
        raise HTTPException(status_code=401, detail="Missing X-Apple-User-Id header")
    token = authorization.split(" ", 1)[1]
    try:
        validate_apple_token(token, x_apple_user_id)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid identity token: {e}")
    return x_apple_user_id

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


# ─── Bible Text Cache Proxy ──────────────────────────────────────────────────

# Translation code → API.Bible bibleId mapping (English translations)
BIBLE_ID_MAP = {
    "niv": "06125adad2d5898a-01",
    "esv": "f421fe261da7624f-01",
    "nlt": "65eec8e0b60e656b-01",
    "nkjv": "de4e12af7f28f599-02",
    "nasb": "5b23a9ce2f004971-01",
    "csb": "a556c5305ee15c3f-01",
    "nrsv": "1e8ab327edbce67f-01",
    "ceb": "aab8aea595d6c498-01",
    "msg": "65eec8e0b60e656b-01",
}


def normalize_reference(reference: str) -> str:
    """Normalize reference to 'Book Chapter' format for consistent cache keys.
    
    Examples:
        "John 3:16"          -> "John 3"
        "john 3"             -> "John 3"
        "1 Corinthians 13:4" -> "1 Corinthians 13"
        "Genesis 1"          -> "Genesis 1"
    """
    ref = reference.strip()
    # Remove verse portion (after colon)
    if ":" in ref:
        ref = ref[:ref.index(":")]
    ref = ref.strip()
    
    # Split and capitalize properly
    parts = ref.split()
    if not parts:
        return ref
    
    # Handle numbered books like "1 Corinthians", "2 Kings"
    if parts[0].isdigit() and len(parts) >= 2:
        parts[1] = parts[1].capitalize()
        # Rest of parts (chapter number) stay as-is
    else:
        parts[0] = parts[0].capitalize()
    
    return " ".join(parts)


async def resolve_bible_id(translation: str) -> str:
    """Resolve translation code to API.Bible bibleId."""
    code = translation.lower()
    if code in BIBLE_ID_MAP:
        return BIBLE_ID_MAP[code]
    
    # Fallback: try to find via API.Bible /bibles endpoint
    headers = {"api-key": API_BIBLE_KEY}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get("https://api.scripture.api.bible/v1/bibles?language=eng", headers=headers)
        if resp.status_code == 200:
            bibles = resp.json().get("data", [])
            for bible in bibles:
                if bible.get("abbreviation", "").lower() == code:
                    BIBLE_ID_MAP[code] = bible["id"]
                    return bible["id"]
    raise HTTPException(status_code=404, detail=f"Translation '{translation}' not found")


def strip_html(text: str) -> str:
    """Strip HTML tags from API.Bible content."""
    import re
    return re.sub(r'<[^>]+>', '', text).strip()


async def fetch_from_api_bible(reference: str, translation: str) -> list:
    """Fetch a chapter from API.Bible and return as verse list."""
    bible_id = await resolve_bible_id(translation)
    headers = {"api-key": API_BIBLE_KEY}
    
    url = f"https://api.scripture.api.bible/v1/bibles/{bible_id}/passages"
    params = {"reference": reference}
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, headers=headers, params=params)
        
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail=f"Reference '{reference}' not found for translation '{translation}'")
    if resp.status_code != 200:
        print(f"[BIBLE_CACHE] API.Bible error: {resp.status_code} {resp.text[:200]}", flush=True)
        raise HTTPException(status_code=502, detail=f"API.Bible returned {resp.status_code}")
    
    data = resp.json().get("data", {})
    
    # API.Bible can return verses in different formats
    # Try to extract structured verse data
    verses = []
    
    # Check if there are per-verse entries
    verse_items = data.get("verses", [])
    if verse_items:
        for v in verse_items:
            text = strip_html(v.get("content", "")).strip()
            if not text:
                continue
            verses.append({
                "book": v.get("bookName", ""),
                "chapter": int(v.get("chapter", "0") if v.get("chapter") else "0"),
                "verse": int(v.get("verse", "0") if v.get("verse") else "0"),
                "text": text
            })
    else:
        # Fallback: parse the content block
        content = data.get("content", "")
        if content:
            text = strip_html(content).strip()
            # Parse reference for metadata
            parts = reference.split()
            book = " ".join(parts[:-1]) if len(parts) > 1 else parts[0]
            chapter = int(parts[-1]) if parts[-1].isdigit() else 0
            if text:
                verses.append({
                    "book": book,
                    "chapter": chapter,
                    "verse": 1,
                    "text": text
                })
    
    if not verses:
        raise HTTPException(status_code=404, detail=f"No verses found for '{reference}' in '{translation}'")
    
    return verses


@app.post("/api/bible/passage")
async def bible_passage(payload: dict):
    """Cache-through proxy: check Turso cache first, then API.Bible on miss.
    No authentication required (Bible text is not user-specific).
    """
    reference = (payload.get("reference") or "").strip()
    translation = (payload.get("translation") or "").strip().lower()
    
    if not reference:
        raise HTTPException(status_code=400, detail="Missing 'reference' field")
    if not translation:
        raise HTTPException(status_code=400, detail="Missing 'translation' field")
    
    normalized_ref = normalize_reference(reference)
    print(f"[BIBLE_CACHE] Request: translation={translation} reference='{reference}' normalized='{normalized_ref}'", flush=True)
    
    # Step 1: Check Turso cache
    try:
        result = await turso_execute(
            "SELECT verses_json FROM bible_cache WHERE translation = ? AND reference = ?",
            [translation, normalized_ref]
        )
        
        if result["rows"]:
            verses = json.loads(result["rows"][0]["verses_json"])
            print(f"[BIBLE_CACHE] ✅ CACHE HIT: {translation}/{normalized_ref} ({len(verses)} verses)", flush=True)
            return {"verses": verses, "translation": translation, "reference": normalized_ref, "cached": True}
    except Exception as e:
        print(f"[BIBLE_CACHE] ⚠️ Turso error (falling through to API.Bible): {e}", flush=True)
    
    # Step 2: Cache miss — fetch from API.Bible
    print(f"[BIBLE_CACHE] ❌ CACHE MISS: {translation}/{normalized_ref} — fetching from API.Bible", flush=True)
    
    if not API_BIBLE_KEY:
        raise HTTPException(status_code=500, detail="Server missing API_BIBLE_KEY")
    
    verses = await fetch_from_api_bible(normalized_ref, translation)
    
    # Step 3: Store in cache
    try:
        await turso_execute(
            "INSERT OR IGNORE INTO bible_cache (translation, reference, verses_json) VALUES (?, ?, ?)",
            [translation, normalized_ref, json.dumps(verses)]
        )
        print(f"[BIBLE_CACHE] 💾 Stored in cache: {translation}/{normalized_ref} ({len(verses)} verses)", flush=True)
    except Exception as e:
        print(f"[BIBLE_CACHE] ⚠️ Cache store failed (non-fatal): {e}", flush=True)
    
    return {"verses": verses, "translation": translation, "reference": normalized_ref, "cached": False}

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


# ─── User Management Endpoints ───────────────────────────────────────────────

@app.post("/api/users/login")
async def users_login(payload: dict):
    apple_user_id = payload.get("appleUserIdentifier")
    identity_token = payload.get("identityToken")
    name = payload.get("name")
    email = payload.get("email")

    if not apple_user_id or not identity_token:
        raise HTTPException(status_code=400, detail="Missing appleUserIdentifier or identityToken")

    try:
        validate_apple_token(identity_token, apple_user_id)
    except Exception as e:
        print(f"[LOGIN] Token validation failed: {e}", flush=True)
        raise HTTPException(status_code=401, detail="Invalid identity token")

    # Check if user exists
    result = await turso_execute("SELECT * FROM users WHERE apple_user_id = ?", [apple_user_id])

    if result["rows"]:
        user = result["rows"][0]
        sub_result = await turso_execute("SELECT * FROM subscriptions WHERE apple_user_id = ?", [apple_user_id])
        sub = sub_result["rows"][0] if sub_result["rows"] else None

        return {
            "isNewUser": False,
            "profile": map_user_row(user),
            "subscription": map_subscription_row(sub) if sub else None,
        }

    # New user
    await turso_execute(
        "INSERT INTO users (apple_user_id, name, email) VALUES (?, ?, ?)",
        [apple_user_id, name, email],
    )
    print(f"[LOGIN] New user: {apple_user_id[:10]}...", flush=True)

    return {"isNewUser": True, "profile": None, "subscription": None}


@app.post("/api/users/profile")
async def users_profile(payload: dict, authorization: str | None = Header(default=None), x_apple_user_id: str | None = Header(default=None, alias="x-apple-user-id")):
    user_id = await authenticate_request(authorization, x_apple_user_id)

    # Verify user exists
    check = await turso_execute("SELECT apple_user_id FROM users WHERE apple_user_id = ?", [user_id])
    if not check["rows"]:
        raise HTTPException(status_code=404, detail="User not found. Call /api/users/login first.")

    # Build dynamic UPDATE
    field_map = {
        "name": "name", "age": "age", "gender": "gender",
        "denomination": "denomination", "denominationNotes": "denomination_notes",
        "relationshipSummary": "relationship_summary",
    }
    updates = []
    args = []
    for js_key, db_col in field_map.items():
        if js_key in payload:
            updates.append(f"{db_col} = ?")
            args.append(payload[js_key])

    if "defaultTranslations" in payload:
        updates.append("default_translations = ?")
        args.append(json.dumps(payload["defaultTranslations"]))

    if "onboardingCompleted" in payload:
        updates.append("onboarding_completed = ?")
        args.append(1 if payload["onboardingCompleted"] else 0)

    if updates:
        updates.append("updated_at = datetime('now')")
        args.append(user_id)
        await turso_execute(f"UPDATE users SET {', '.join(updates)} WHERE apple_user_id = ?", args)

    # Subscription sync
    if "subscription" in payload and payload["subscription"]:
        sub = payload["subscription"]
        product_id = sub.get("productId")
        if product_id:
            await turso_execute(
                """INSERT INTO subscriptions (apple_user_id, product_id, status, updated_at)
                   VALUES (?, ?, 'active', datetime('now'))
                   ON CONFLICT(apple_user_id) DO UPDATE SET
                     product_id = excluded.product_id,
                     status = 'active',
                     updated_at = datetime('now')""",
                [user_id, product_id],
            )

    return {"updated": True}


@app.get("/api/users/me")
async def users_me(authorization: str | None = Header(default=None), x_apple_user_id: str | None = Header(default=None, alias="x-apple-user-id")):
    user_id = await authenticate_request(authorization, x_apple_user_id)

    result = await turso_execute("SELECT onboarding_completed FROM users WHERE apple_user_id = ?", [user_id])
    if not result["rows"]:
        raise HTTPException(status_code=404, detail="User not found")

    user = result["rows"][0]
    sub_result = await turso_execute("SELECT product_id, status, expires_at FROM subscriptions WHERE apple_user_id = ?", [user_id])
    sub = sub_result["rows"][0] if sub_result["rows"] else None

    return {
        "exists": True,
        "onboardingCompleted": str(user.get("onboarding_completed")) == "1",
        "subscription": {"productId": sub["product_id"], "status": sub["status"], "expiresAt": sub.get("expires_at")} if sub else None,
    }


@app.delete("/api/users/account")
async def users_delete_account(authorization: str | None = Header(default=None), x_apple_user_id: str | None = Header(default=None, alias="x-apple-user-id")):
    user_id = await authenticate_request(authorization, x_apple_user_id)

    # Delete subscriptions first (no CASCADE in HTTP API)
    await turso_execute("DELETE FROM subscriptions WHERE apple_user_id = ?", [user_id])
    result = await turso_execute("DELETE FROM users WHERE apple_user_id = ?", [user_id])

    if result["rowsAffected"] == 0:
        raise HTTPException(status_code=404, detail="User not found")

    print(f"[ACCOUNT] Deleted: {user_id[:10]}...", flush=True)
    return {"deleted": True, "message": "Account and all associated data have been permanently deleted."}


@app.post("/api/webhooks/apple")
async def webhooks_apple(payload: dict):
    """App Store Server Notifications v2 webhook."""
    signed_payload = payload.get("signedPayload")
    if not signed_payload:
        raise HTTPException(status_code=400, detail="Missing signedPayload")

    try:
        # Decode header to get x5c
        header_b64 = signed_payload.split(".")[0]
        # Add padding if needed
        padding = 4 - len(header_b64) % 4
        if padding != 4:
            header_b64 += "=" * padding
        import base64
        header = json.loads(base64.urlsafe_b64decode(header_b64))

        # For now, decode the payload without full x5c chain validation
        # (Full validation requires Apple root CA cert verification)
        payload_b64 = signed_payload.split(".")[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        notification = json.loads(base64.urlsafe_b64decode(payload_b64))

        notification_type = notification.get("notificationType", "")
        subtype = notification.get("subtype", "")
        print(f"[WEBHOOK] type={notification_type} subtype={subtype}", flush=True)

        # Decode signedTransactionInfo
        data = notification.get("data", {})
        signed_txn = data.get("signedTransactionInfo")
        if signed_txn:
            txn_payload_b64 = signed_txn.split(".")[1]
            padding = 4 - len(txn_payload_b64) % 4
            if padding != 4:
                txn_payload_b64 += "=" * padding
            txn_info = json.loads(base64.urlsafe_b64decode(txn_payload_b64))

            original_txn_id = txn_info.get("originalTransactionId")
            product_id = txn_info.get("productId")
            expires_date = txn_info.get("expiresDate")

            if expires_date and isinstance(expires_date, (int, float)):
                expires_at = datetime.datetime.fromtimestamp(expires_date / 1000, tz=datetime.timezone.utc).isoformat()
            else:
                expires_at = None

            handle_notification(notification_type, original_txn_id, product_id, expires_at)

        return {"received": True}
    except Exception as e:
        print(f"[WEBHOOK] Error: {e}", flush=True)
        raise HTTPException(status_code=400, detail="Invalid payload")


def handle_notification(notification_type: str, original_txn_id: str | None, product_id: str | None, expires_at: str | None):
    status_map = {
        "DID_RENEW": "active",
        "SUBSCRIBED": "active",
        "DID_CHANGE_RENEWAL_STATUS": "active",
        "EXPIRED": "expired",
        "DID_FAIL_TO_RENEW": "billing_retry",
        "REVOKE": "revoked",
        "REFUND": "revoked",
    }
    new_status = status_map.get(notification_type)
    if not new_status or not original_txn_id:
        return

    import asyncio
    asyncio.ensure_future(turso_execute(
        "UPDATE subscriptions SET status = ?, expires_at = ?, updated_at = datetime('now') WHERE original_transaction_id = ?",
        [new_status, expires_at, original_txn_id],
    ))
    print(f"[WEBHOOK] Updated txn={original_txn_id} → {new_status}", flush=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def map_user_row(row: dict) -> dict:
    translations = row.get("default_translations")
    return {
        "appleUserId": row.get("apple_user_id"),
        "name": row.get("name"),
        "age": row.get("age"),
        "gender": row.get("gender") or "undisclosed",
        "denomination": row.get("denomination"),
        "defaultTranslations": json.loads(translations) if translations else [],
        "denominationNotes": row.get("denomination_notes"),
        "relationshipSummary": row.get("relationship_summary"),
        "onboardingCompleted": str(row.get("onboarding_completed")) == "1",
        "createdAt": row.get("created_at"),
    }

def map_subscription_row(row: dict) -> dict:
    return {
        "productId": row.get("product_id"),
        "status": row.get("status"),
        "expiresAt": row.get("expires_at"),
    }
