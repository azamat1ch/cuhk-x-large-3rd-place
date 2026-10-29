"""Conservative per-model-call retries; the SDK retains the live tool history."""

import random

import httpx
from agents.retry import ModelRetrySettings, RetryDecision
from openai import APIStatusError


def retry_reason(error):
    """Return a safe reason only when the request was not accepted upstream."""
    if isinstance(error, APIStatusError):
        body = error.body if isinstance(error.body, dict) else {}
        detail = body.get("error", body)
        detail = detail if isinstance(detail, dict) else {}
        if error.status_code == 429:
            if detail.get("code") == "rate_limit_exceeded":
                return "rate_limit_rejected"
            return None
        # Some upstream services wrap dial failures in HTTP500. Only this explicit
        # pre-connection wrapper is safe; a generic500/read reset is ambiguous.
        message = detail.get("message", "")
        if error.status_code == 500 and isinstance(message, str):
            if "utls: dial upstream: dial tcp" in message and any(
                marker in message
                for marker in (
                    "lookup ",
                    "connect: cannot assign requested address",
                    "connect: network is unreachable",
                    "connect: connection refused",
                    "connect: no route to host",
                )
            ):
                return "upstream_not_connected"
        return None
    # Local TCP connection establishment failed, before sending an HTTP request.
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
            return "local_connection_not_established"
        error = error.__cause__
    return None


def transport_retry_settings(record_event):
    def policy(context):
        reason = retry_reason(context.error)
        if reason is None:
            return False
        delay = min(32.0, 2.0**context.attempt * random.uniform(0.8, 1.2))
        requested = context.normalized.retry_after
        if requested is not None:
            if requested > 60:
                return False
            delay = max(delay, requested)
        record_event({"reason": reason, "retry": context.attempt, "delay_s": round(delay, 3)})
        return RetryDecision(retry=True, delay=delay, reason=reason)

    return ModelRetrySettings(max_retries=5, policy=policy)
