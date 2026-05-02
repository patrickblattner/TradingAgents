"""Codex OAuth flow + token storage.

PKCE flow against `auth.openai.com`, mimicking the Codex CLI client_id so
the resulting tokens grant access to `chatgpt.com/backend-api/codex`.

Token store layout (compatible with codex CLI's auth.json):
    {
      "OPENAI_API_KEY": null,
      "tokens": {
        "id_token": "<JWT>",
        "access_token": "<JWT>",
        "refresh_token": "<opaque>",
        "account_id": "<extracted from JWT>"
      },
      "last_refresh": "<ISO8601 UTC>"
    }
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import socketserver
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# OAuth client constants — these mirror the official Codex CLI so the resulting
# tokens are accepted by the chatgpt.com Codex backend. Changing them breaks auth.
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"

# Refresh access token when this close to JWT exp (seconds).
REFRESH_SKEW_SECONDS = 5 * 60

# Default token store + optional codex CLI fallback path.
DEFAULT_TOKEN_PATH = Path.home() / ".tradingagents" / "codex_auth.json"
CODEX_CLI_TOKEN_PATH = Path.home() / ".codex" / "auth.json"


class CodexAuthError(RuntimeError):
    """Raised when the Codex OAuth flow or token refresh fails."""


class CodexAuthRequired(CodexAuthError):
    """Raised when no usable Codex tokens are available — user must run login."""


# --- PKCE helpers -----------------------------------------------------------


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _generate_pkce() -> tuple[str, str]:
    """Return (verifier, challenge) for PKCE S256."""
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


# --- JWT decode (no signature verification — server validates) --------------


def _decode_jwt_payload(token: str) -> dict:
    """Decode the payload portion of a JWT. Raises ValueError on malformed input."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("not a JWT")
    payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
    return json.loads(base64.urlsafe_b64decode(payload_b64))


def extract_account_id(access_token: str) -> Optional[str]:
    """Pull `chatgpt_account_id` out of the access token's JWT claims.

    Required for the `ChatGPT-Account-ID` header on Codex backend requests.
    Returns None on any decode failure — never raises, since callers want to
    surface auth errors as 401s, not crashes at client construction.
    """
    if not isinstance(access_token, str) or not access_token.strip():
        return None
    try:
        claims = _decode_jwt_payload(access_token)
        acct = claims.get("https://api.openai.com/auth", {}).get("chatgpt_account_id")
        return acct if isinstance(acct, str) and acct else None
    except Exception:
        return None


def _jwt_exp_or_none(access_token: str) -> Optional[int]:
    try:
        return int(_decode_jwt_payload(access_token).get("exp", 0)) or None
    except Exception:
        return None


# --- Token persistence ------------------------------------------------------


def _token_path() -> Path:
    override = os.environ.get("TRADINGAGENTS_CODEX_AUTH_PATH")
    return Path(override).expanduser() if override else DEFAULT_TOKEN_PATH


def load_tokens(*, allow_codex_cli_fallback: bool = True) -> Optional[dict]:
    """Load tokens from our store, optionally falling back to the codex CLI's store."""
    path = _token_path()
    for candidate in [path] + ([CODEX_CLI_TOKEN_PATH] if allow_codex_cli_fallback else []):
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tokens = data.get("tokens") if isinstance(data, dict) else None
        if isinstance(tokens, dict) and tokens.get("access_token"):
            return data
    return None


def save_tokens(data: dict) -> Path:
    path = _token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_token_record(access: str, refresh: str, id_token: Optional[str] = None) -> dict:
    return {
        "OPENAI_API_KEY": None,
        "tokens": {
            "id_token": id_token,
            "access_token": access,
            "refresh_token": refresh,
            "account_id": extract_account_id(access),
        },
        "last_refresh": _now_iso(),
    }


# --- HTTP token endpoint calls ----------------------------------------------


def _post_form(url: str, fields: dict) -> dict:
    body = urllib.parse.urlencode(fields).encode("ascii")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise CodexAuthError(f"OAuth {url} -> {exc.code}: {detail}") from exc


def _exchange_code(code: str, verifier: str) -> dict:
    resp = _post_form(TOKEN_URL, {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "code_verifier": verifier,
        "redirect_uri": REDIRECT_URI,
    })
    if not (resp.get("access_token") and resp.get("refresh_token")):
        raise CodexAuthError(f"token endpoint missing fields: {resp}")
    return resp


def refresh_tokens(refresh_token: str) -> dict:
    """Exchange a refresh token for a new access+refresh pair."""
    resp = _post_form(TOKEN_URL, {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
    })
    if not (resp.get("access_token") and resp.get("refresh_token")):
        raise CodexAuthError(f"refresh response missing fields: {resp}")
    return resp


# --- Login flow -------------------------------------------------------------


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """One-shot HTTP handler: captures the OAuth code from /auth/callback."""

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_response(404)
            self.end_headers()
            return
        params = urllib.parse.parse_qs(parsed.query)
        self.server.oauth_code = (params.get("code") or [None])[0]  # type: ignore[attr-defined]
        self.server.oauth_state = (params.get("state") or [None])[0]  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<html><body style='font-family:sans-serif'>"
            b"<h2>TradingAgents: Codex OAuth complete</h2>"
            b"<p>You can close this tab and return to the terminal.</p>"
            b"</body></html>"
        )

    def log_message(self, *args, **kwargs):  # silence default access log
        pass


def _build_authorize_url(state: str, challenge: str) -> str:
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "codex_cli_rs",
    }
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode(params)


def login(*, open_browser: bool = True, timeout_seconds: int = 300) -> dict:
    """Run the Codex OAuth flow interactively and persist the resulting tokens.

    Spins up a localhost listener on port 1455 (the Codex CLI's redirect port),
    opens the user's browser to auth.openai.com, and waits for the callback.
    """
    verifier, challenge = _generate_pkce()
    state = secrets.token_hex(16)
    auth_url = _build_authorize_url(state, challenge)

    server = socketserver.TCPServer(("127.0.0.1", 1455), _CallbackHandler)
    server.oauth_code = None  # type: ignore[attr-defined]
    server.oauth_state = None  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print("Opening browser for ChatGPT Codex OAuth login...")
    print(f"  If it doesn't open, visit: {auth_url}\n")
    if open_browser:
        try:
            webbrowser.open(auth_url)
        except Exception:
            pass

    deadline = time.time() + timeout_seconds
    try:
        while time.time() < deadline:
            if server.oauth_code:  # type: ignore[attr-defined]
                break
            time.sleep(0.2)
    finally:
        server.shutdown()
        server.server_close()

    code = server.oauth_code  # type: ignore[attr-defined]
    received_state = server.oauth_state  # type: ignore[attr-defined]
    if not code:
        raise CodexAuthError("OAuth flow timed out before callback was received.")
    if received_state != state:
        raise CodexAuthError("OAuth state mismatch (possible CSRF) — aborting.")

    resp = _exchange_code(code, verifier)
    record = _build_token_record(
        access=resp["access_token"],
        refresh=resp["refresh_token"],
        id_token=resp.get("id_token"),
    )
    saved_to = save_tokens(record)
    acct = record["tokens"]["account_id"] or "<unknown>"
    print(f"Logged in. Account: {acct}")
    print(f"Tokens stored at: {saved_to}")
    return record


# --- Public access-token getter (refresh-on-demand) -------------------------


def get_access_token(*, allow_refresh: bool = True) -> str:
    """Return a non-expired access token, refreshing if needed.

    Raises CodexAuthRequired when no tokens are available so the caller can
    surface a clear "run codex-login" message instead of crashing on a 401.
    """
    record = load_tokens()
    if not record:
        raise CodexAuthRequired(
            "No Codex tokens found. Run `tradingagents codex-login` to authenticate "
            "with your ChatGPT Plus/Pro subscription."
        )
    tokens = record.get("tokens") or {}
    access = tokens.get("access_token")
    refresh = tokens.get("refresh_token")
    if not access or not refresh:
        raise CodexAuthRequired("Codex token store is missing access/refresh fields.")

    exp = _jwt_exp_or_none(access)
    needs_refresh = exp is not None and exp - time.time() < REFRESH_SKEW_SECONDS

    if needs_refresh and allow_refresh:
        resp = refresh_tokens(refresh)
        record = _build_token_record(
            access=resp["access_token"],
            refresh=resp["refresh_token"],
            id_token=resp.get("id_token") or tokens.get("id_token"),
        )
        save_tokens(record)
        access = record["tokens"]["access_token"]

    return access
