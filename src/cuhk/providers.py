"""API credentials, endpoint validation and image attachment encoding."""

import base64
import os
from pathlib import Path

import httpx

KEY_NAMES = ("OPENAI_API_KEY",)
DEFAULT_ENDPOINT = "https://api.openai.com/v1"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def load_keys(path=Path(".env")):
    """Read only provider key assignments; never execute dotenv contents."""
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key in KEY_NAMES:
                os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def data_url(path, mime):
    return "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def api_endpoint(explicit=None):
    """Explicit frozen configuration wins; environment configures new invocations."""
    value = explicit or os.environ.get("OPENAI_BASE_URL") or DEFAULT_ENDPOINT
    parsed = httpx.URL(value)
    if (
        not parsed.host
        or parsed.scheme not in {"http", "https"}
        or (parsed.scheme == "http" and parsed.host not in LOOPBACK)
        or parsed.userinfo
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "API endpoint requires HTTPS (HTTP allowed on loopback), without credentials, query or fragment"
        )
    return str(parsed).rstrip("/")


def api_key(endpoint):
    """Return the API key for the endpoint; fail early when it is absent."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY")
    return key
