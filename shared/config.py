import logging
import os
import gradio as gr

# Directory for troubleshooting logs (app log + per-engine response dumps).
# Lives at the repo root so it's easy to find; gitignored.
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Per-engine raw response dumps go in a subfolder so the top-level logs/ stays
# readable (app.log + one file per failed/successful call when enabled).
RESPONSE_LOG_DIR = os.path.join(LOGS_DIR, "responses")
os.makedirs(RESPONSE_LOG_DIR, exist_ok=True)

# Configure logging: INFO to console (basicConfig) AND to logs/app.log so a full
# trace is captured for troubleshooting the 0-token / truncation / parse cases.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
if not any(
    isinstance(h, logging.FileHandler)
    and getattr(h, "baseFilename", "") == os.path.join(LOGS_DIR, "app.log")
    for h in logger.handlers
):
    _file_handler = logging.FileHandler(os.path.join(LOGS_DIR, "app.log"), encoding="utf-8")
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(_file_handler)
    # Don't double-propagate to the root logger's handlers (basicConfig already
    # prints to console); keep the file handler dedicated to this logger.
    logger.propagate = True

# Image size constants
MAX_IMAGE_SIZE = 5 * 1024 * 1024 - 100000  # 5MB minus buffer for Bedrock

# Default S3 bucket used by Textract (for PDFs) and BDA.
# Resolution order: shared/local_settings.py (gitignored) -> OCR_S3_BUCKET env var -> "".
# Users can also override at runtime via the UI textbox.
try:
    from .local_settings import DEFAULT_S3_BUCKET as _LOCAL_S3_BUCKET
except ImportError:
    _LOCAL_S3_BUCKET = ""
DEFAULT_S3_BUCKET = _LOCAL_S3_BUCKET or os.environ.get("OCR_S3_BUCKET", "")

# Cerebras API key for the external Cerebras Inference provider.
# Resolution order: shared/local_settings.py (gitignored) -> CEREBRAS_API_KEY env var -> "".
try:
    from .local_settings import CEREBRAS_API_KEY as _LOCAL_CEREBRAS_API_KEY
except ImportError:
    _LOCAL_CEREBRAS_API_KEY = ""
CEREBRAS_API_KEY = _LOCAL_CEREBRAS_API_KEY or os.environ.get("CEREBRAS_API_KEY", "")

# Optional override for the Cerebras API base URL (defaults to the SDK's built-in endpoint).
CEREBRAS_BASE_URL = os.environ.get("CEREBRAS_BASE_URL", "")

# Available Cerebras models (external OpenAI-style chat-completions provider).
# Maps a display name -> Cerebras model ID. See the Gemma 4 Multimodal Quick Start Guide.
CEREBRAS_MODELS = {
    "Gemma 4 31B (Cerebras)": "gemma-4-31b-trial",
}

# Available Bedrock models
BEDROCK_MODELS = {
    "Claude Opus 4.8": "us.anthropic.claude-opus-4-8",
    "Claude Sonnet 4.6": "us.anthropic.claude-sonnet-4-6",
    "Claude Haiku 4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "Amazon Nova 2 Lite": "us.amazon.nova-2-lite-v1:0",
    "Pixtral Large": "us.mistral.pixtral-large-2502-v1:0",
    "Mistral Large 3": "mistral.mistral-large-3-675b-instruct",
    "Llama 4 Maverick 17B": "us.meta.llama4-maverick-17b-instruct-v1:0",
    "Llama 4 Scout 17B": "us.meta.llama4-scout-17b-instruct-v1:0",
    # Gemma 3 multimodal models — available ON_DEMAND in this account/region
    # (confirmed via list_foundation_models, TEXT+IMAGE input).
    "Gemma 3 4B": "google.gemma-3-4b-it",
    "Gemma 3 12B": "google.gemma-3-12b-it",
    "Gemma 3 27B": "google.gemma-3-27b-it",
    # Gemma 4 on Bedrock — announced (https://aws.amazon.com/blogs/machine-learning/introducing-gemma-4-models-on-amazon-bedrock/)
    # but NOT yet available in this account/region (list_foundation_models returns only
    # google.gemma-3-*). Commented out until access is granted so benchmark runs don't fail.
    # Model IDs below are unverified — confirm against list_foundation_models before enabling.
    # "Gemma 4 31B": "google.gemma-4-31b-it",
    # "Gemma 4 26B-A4B": "google.gemma-4-26b-a4b-it",
    # "Gemma 4 E2B": "google.gemma-4-e2b-it"
}

# Default model for post-processing
POSTPROCESSING_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Effort levels per model: maps model_id to (thinking_type, list_of_levels)
# - "adaptive": Claude Sonnet 4.6 / Opus 4.8 — uses thinking.type="adaptive" + effort param via invoke_model
# - "budget": Claude Haiku 4.5 — uses thinking.type="enabled" + budget_tokens via invoke_model
# - "nova": Nova 2 Lite — uses reasoningConfig via converse additionalModelRequestFields
EFFORT_LEVELS = {
    "us.anthropic.claude-opus-4-8": ("adaptive", ["low", "medium"]),
    "us.anthropic.claude-sonnet-4-6": ("adaptive", ["low", "medium"]),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": ("budget", [1024, 4096, 16384]),
    "us.amazon.nova-2-lite-v1:0": ("nova", ["low"]),
}

# Bedrock models that do NOT support the Converse `outputConfig.textFormat`
# (json_schema) structured-output field. Verified via live converse() probe
# against this account/region: these return either
#   ValidationException: This model doesn't support the outputConfig field.
# or (Opus 4.8) output_config.format: Extra inputs are not permitted.
#
# Default path for every model is Converse + outputConfig (strict schema).
# For the models listed here we instead make a plain Converse call (no
# outputConfig) using the SAME prompt + structured-extraction instructions,
# then run an output parsing/formatting step to recover schema-shaped JSON.
# This is a capability gate (the model rejects the API field), not a
# bad-JSON fallback for models that do support structured output.
#
# Coverage notes (this account/region, verified live):
#   - Pixtral Large and Gemma 3 4B reject Converse outputConfig but DO support
#     the invoke_model `response_format` path. We still route them through the
#     parsing step here to keep a single Converse code path.
#   - Claude Opus 4.8 rejects structured output on ALL paths (Converse,
#     InvokeModel Anthropic output_config.format, open-weight response_format).
#     AWS docs list structured-output support only through Opus 4.6.
#   - Claude Sonnet 4.6 and Haiku 4.5 ACCEPT the outputConfig field but their
#     constrained-decoding grammar compiler cannot handle realistic OCR schemas
#     in this region: Sonnet fails fast with "Schema is too complex" (~5s) and
#     Haiku hangs ~180s then fails with "Grammar compilation timed out" — for
#     both the small insurance_card schema AND the driver_license schema.
#     Routing them to the prompt-based path avoids the grammar step entirely
#     (and the wasted 180s timeout per call). The bedrock_engine retry covers
#     any other model that hits these errors unexpectedly.
MODELS_WITHOUT_STRUCTURED_OUTPUT = {
    "us.anthropic.claude-opus-4-8",
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.amazon.nova-2-lite-v1:0",
    "us.mistral.pixtral-large-2502-v1:0",
    # Mistral Large 3 ACCEPTS the Converse outputConfig field but constrained
    # decoding returns an EMPTY object ("{}", 2 output tokens) instead of
    # populated data — verified live against insurance_card (0% accuracy).
    # The prompt-based path extracts fields correctly, so route it there.
    "mistral.mistral-large-3-675b-instruct",
    "us.meta.llama4-maverick-17b-instruct-v1:0",
    "us.meta.llama4-scout-17b-instruct-v1:0",
    "google.gemma-3-4b-it",
    # Gemma 3 12B/27B ACCEPT the Converse outputConfig field but constrained
    # decoding is unreliable for them on this account/region: it either runs
    # away to the 8000-token output cap without closing the JSON (stop_reason
    # max_tokens -> 0/0 rows) or terminates early with a near-empty object
    # (~3/18 fields, ~16% accuracy). Verified live against the insurance_card
    # sample: constrained = 2/3 runs truncated + 16.7% on the survivor, while
    # the prompt-based path scored 94-100% across 3 runs with no truncation.
    # Route them to the prompt-based path like Gemma 3 4B.
    "google.gemma-3-12b-it",
    "google.gemma-3-27b-it",
}

# Number of fixed rows reserved in the results Dataframe. Gradio 6.x only
# renders streaming Dataframe updates correctly when the row count is constant
# across generator yields, so we fix the grid size and pad incremental results
# to this length. Must be >= the max number of engine variants produced by a
# run (8 Bedrock models with effort expansions + Cerebras + Textract + BDA).
RESULTS_TABLE_ROWS = 30

# Max output tokens for standard (non-reasoning) generation. OCR of dense
# documents (full forms, large tables) can produce sizeable JSON, and the
# prompt-based fallback path is more verbose than constrained decoding — so we
# give generous headroom to avoid mid-JSON truncation. You only pay for tokens
# actually generated. Reasoning/effort calls use a higher cap (see engines).
#
# NOTE: this is the DESIRED cap. Bedrock enforces a per-model output ceiling and
# rejects requests that exceed it (ValidationException), so the engine clamps
# this to each model's documented max via resolve_max_output_tokens(). 12K gives
# comfortable headroom over the old borderline 8K for every model that allows it
# (Pixtral 16K, Mistral/Nova/Claude 32K+); Llama 4 and Gemma 3 are hard-capped at
# 8K by Bedrock and stay clamped there (12K would be rejected for them).
MAX_OUTPUT_TOKENS = 12000

# Higher cap used when reasoning/effort is enabled (thinking tokens count toward
# the budget, so the limit must accommodate both reasoning and the final JSON).
# Also clamped per-model by resolve_max_output_tokens().
MAX_OUTPUT_TOKENS_REASONING = 32000

# Per-model maximum output tokens, from the Amazon Bedrock model cards
# (verified live, this account/region). Requests exceeding a model's ceiling are
# rejected with a ValidationException, so the desired cap above is clamped to
# these values per model. Keys are matched as substrings against the model_id,
# so cross-region inference profiles (e.g. "us.anthropic.claude-sonnet-4-6")
# resolve to the same ceiling as the base model id. Models not listed fall back
# to MODEL_MAX_OUTPUT_TOKENS_DEFAULT.
#   Claude 4.x / Nova 2 Lite : 64K     Mistral Large 3 : 32K
#   Pixtral Large            : 16K     Llama 4 (both)  : 8K
#   Gemma 3 (all sizes)      : 8K
MODEL_MAX_OUTPUT_TOKENS = {
    "claude-opus-4": 64000,
    "claude-sonnet-4": 64000,
    "claude-haiku-4": 64000,
    "nova-2-lite": 64000,
    "mistral-large-3": 32000,
    "pixtral-large": 16000,
    "llama4-maverick": 8000,
    "llama4-scout": 8000,
    "gemma-3": 8000,
    # Cerebras-hosted Gemma 4 (external provider, not Bedrock). Supports a large
    # output window; cap generously since OCR JSON stays well under this.
    "gemma-4-31b-trial": 32000,
}
# Conservative default for any model not explicitly listed above. 8K is the
# lowest ceiling among current models, so it is always a valid request.
MODEL_MAX_OUTPUT_TOKENS_DEFAULT = 8000


def resolve_max_output_tokens(model_id, desired):
    """Clamp a desired max-output-tokens value to the given model's documented
    ceiling so Bedrock never rejects the request for exceeding the model limit.

    Matches MODEL_MAX_OUTPUT_TOKENS keys as substrings of model_id (handles
    cross-region inference profile prefixes like "us.", "eu."). Returns
    min(desired, model_ceiling); unknown models use MODEL_MAX_OUTPUT_TOKENS_DEFAULT.
    """
    ceiling = MODEL_MAX_OUTPUT_TOKENS_DEFAULT
    if model_id:
        for key, limit in MODEL_MAX_OUTPUT_TOKENS.items():
            if key in model_id:
                ceiling = limit
                break
    return min(desired, ceiling)

# API cost information - Only for APIs currently in use
API_COSTS = {
    # Currently used Textract APIs
    'textract_detect': 1.50 / 1000,  # DetectDocumentText API: $1.50 per 1,000 pages
    'textract_async': 1.50 / 1000,   # StartDocumentTextDetection API: $1.50 per 1,000 pages
    
    'bedrock': {
        # Claude models
        'us.anthropic.claude-opus-4-8': {
            'input': 0.015 / 1000,   # $15.00 per 1M input tokens
            'output': 0.075 / 1000   # $75.00 per 1M output tokens
        },
        'us.anthropic.claude-sonnet-4-6': {
            'input': 0.003 / 1000,   # $3.00 per 1M input tokens
            'output': 0.015 / 1000   # $15.00 per 1M output tokens
        },
        'us.anthropic.claude-haiku-4-5-20251001-v1:0': {
            'input': 0.001 / 1000,   # $1.00 per 1M input tokens
            'output': 0.005 / 1000   # $5.00 per 1M output tokens
        },
        # Nova models
        'us.amazon.nova-2-lite-v1:0': {
            'input': 0.00008 / 1000,  # $0.08 per 1M input tokens
            'output': 0.00032 / 1000  # $0.32 per 1M output tokens
        },
        # Mistral models
        'us.mistral.pixtral-large-2502-v1:0': {
            'input': 0.002 / 1000,    # $2.00 per 1M input tokens
            'output': 0.006 / 1000    # $6.00 per 1M output tokens
        },
        'mistral.mistral-large-3-675b-instruct': {
            'input': 0.002 / 1000,    # $2.00 per 1M input tokens
            'output': 0.006 / 1000    # $6.00 per 1M output tokens
        },
        # Meta Llama 4 models
        'us.meta.llama4-maverick-17b-instruct-v1:0': {
            'input': 0.00020 / 1000,  # $0.20 per 1M input tokens
            'output': 0.00060 / 1000  # $0.60 per 1M output tokens
        },
        'us.meta.llama4-scout-17b-instruct-v1:0': {
            'input': 0.00015 / 1000,  # $0.15 per 1M input tokens
            'output': 0.00045 / 1000  # $0.45 per 1M output tokens
        },
        # Google Gemma 3 multimodal models (Bedrock on-demand pricing).
        'google.gemma-3-4b-it': {
            'input': 0.00004 / 1000,  # $0.04 per 1M input tokens
            'output': 0.00008 / 1000  # $0.08 per 1M output tokens
        },
        'google.gemma-3-12b-it': {
            'input': 0.00009 / 1000,  # $0.09 per 1M input tokens
            'output': 0.00029 / 1000  # $0.29 per 1M output tokens
        },
        'google.gemma-3-27b-it': {
            'input': 0.00023 / 1000,  # $0.23 per 1M input tokens
            'output': 0.00038 / 1000  # $0.38 per 1M output tokens
        },
        # Cerebras-hosted models (external provider). Keyed here so the shared
        # variant cost path (calculate_bedrock_cost) works unchanged.
        # NOTE: Gemma 4 31B is a Cerebras Private Preview "trial" with no
        # published per-token price yet — placeholder 0.0 until pricing is known.
        'gemma-4-31b-trial': {
            'input': 0.0,   # TODO: update when Cerebras publishes pricing
            'output': 0.0   # TODO: update when Cerebras publishes pricing
        }
    },
    'bda': {
        'standard': {
            'document': 0.010,  # $0.010 per page
            'image': 0.003      # $0.003 per image
        },
        'custom': {
            'document': 0.040,  # $0.040 per page
            'image': 0.005,     # $0.005 per image
            'extra_field': 0.0005  # $0.0005 per additional field (beyond 30)
        }
    }
}



# Status HTML templates
STATUS_HTML = {
    "processing": lambda engine: f"""<div style='padding: 10px; background-color: #3b5998; color: white; 
                                     border-radius: 5px; font-weight: bold;'>Processing with {engine}...</div>""",
    "completed": lambda engine, time, cost, token_info="": f"""<div style='padding: 10px; background-color: #2e7d32; color: white; 
                                         border-radius: 5px; font-weight: bold;'> {engine} completed in {time:.3f} seconds (Est. cost: ${cost:.6f}){token_info}</div>""",
    "error": lambda engine, time, error: f"""<div style='padding: 10px; background-color: #c62828; color: white; 
                                           border-radius: 5px; font-weight: bold;'> {engine} error ({time:.3f}s): {error}</div>""",
    "global_processing": lambda: """<div style='padding: 10px; background-color: #3b5998; color: white; 
                                    border-radius: 5px; font-weight: bold;'>Processing with selected engines...</div>""",
    "global_completed": lambda time, cost: f"""<div style='padding: 10px; background-color: #2e7d32; color: white; 
                                        border-radius: 5px; font-weight: bold; position: relative;'>All processing completed in {time:.3f} seconds (Total est. cost: ${cost:.6f})<span onclick="this.parentElement.style.display='none'" style="position:absolute; top:6px; right:10px; cursor:pointer; font-size:18px; line-height:1;">×</span></div>""",
    "global_partial": lambda success, total, time, cost: f"""<div style='padding: 10px; background-color: #ed6c02; color: white; 
                                                     border-radius: 5px; font-weight: bold;'>{success}/{total} engines completed in {time:.3f} seconds (Total est. cost: ${cost:.6f})</div>"""
}

# Use a simpler theme approach that works across Gradio versions
CUSTOM_THEME = gr.themes.Default()