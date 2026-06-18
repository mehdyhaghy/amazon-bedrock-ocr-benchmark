"""Per-call response logging for troubleshooting OCR engine results.

Writes a JSON file per engine call into logs/responses/ capturing everything
needed to diagnose the cases that surface as 0-token rows in the results table:
truncation (stopReason == max_tokens), strict-parse failures, grammar-compilation
failures, and capability rejections.

The key design point: failures must record the REAL token usage and the raw model
output, instead of the zeroed-out values the engine returns to the UI. That makes
it possible to tell apart "model truncated mid-JSON" from "model returned valid
text that failed to parse" from "API rejected the request before any generation".
"""

import datetime
import json
import os
import re
from typing import Any, Dict, Optional

from shared.config import logger, RESPONSE_LOG_DIR


def _slugify(value: str) -> str:
    """Make a filesystem-safe fragment from an engine/model name."""
    value = str(value or "unknown")
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return value.strip("_") or "unknown"


def _truncate(text: Optional[str], limit: int = 20000) -> Optional[str]:
    """Cap very large raw responses so log files stay manageable."""
    if text is None:
        return None
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def log_engine_response(
    engine_name: str,
    *,
    model_id: str = "",
    status: str = "success",
    options: Optional[Dict[str, Any]] = None,
    token_usage: Optional[Dict[str, Any]] = None,
    stop_reason: Optional[str] = None,
    raw_text: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    structured_output: Optional[bool] = None,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Write a troubleshooting record for one engine call.

    Args:
        engine_name: Display name / variant label (e.g. "Claude Haiku 4.5 (4096)").
        model_id: Bedrock model id used.
        status: "success", "truncated", "parse_error", "grammar_error",
            "output_config_rejected", or "error".
        options: The options dict passed to process_image (effort_level, etc.).
            Non-serializable values are coerced to str.
        token_usage: REAL token usage from the API response, even on failure.
        stop_reason: The API stopReason (e.g. "max_tokens", "end_turn").
        raw_text: The raw extracted text before JSON parsing (the thing that
            failed to parse, when applicable).
        max_output_tokens: The resolved output-token cap for this call.
        structured_output: Whether the constrained-decoding path was used.
        error: Exception/failure message, if any.
        extra: Any additional fields to record.

    Returns:
        The path to the written log file, or None if logging failed.
    """
    try:
        timestamp = datetime.datetime.now()
        ts_file = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{ts_file}_{_slugify(engine_name)}_{_slugify(status)}.json"
        path = os.path.join(RESPONSE_LOG_DIR, filename)

        # Coerce options to something JSON-serializable without losing info.
        safe_options = {}
        for k, v in (options or {}).items():
            try:
                json.dumps(v)
                safe_options[k] = v
            except (TypeError, ValueError):
                safe_options[k] = str(v)

        record = {
            "timestamp": timestamp.isoformat(),
            "engine": engine_name,
            "model_id": model_id,
            "status": status,
            "stop_reason": stop_reason,
            "max_output_tokens": max_output_tokens,
            "structured_output": structured_output,
            "token_usage": token_usage,
            "raw_text_len": len(raw_text) if raw_text is not None else 0,
            "raw_text": _truncate(raw_text),
            "error": error,
            "options": safe_options,
        }
        if extra:
            record["extra"] = extra

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        # Also surface a concise line in app.log so the trace is greppable
        # without opening individual files.
        tu = token_usage or {}
        logger.info(
            "Response log [%s] engine=%s model=%s stop_reason=%s tokens=%s/%s -> %s",
            status,
            engine_name,
            model_id,
            stop_reason,
            tu.get("inputTokens", 0),
            tu.get("outputTokens", 0),
            filename,
        )
        return path
    except Exception as log_err:  # never let logging break a run
        logger.warning(f"Failed to write response log for {engine_name}: {log_err}")
        return None
