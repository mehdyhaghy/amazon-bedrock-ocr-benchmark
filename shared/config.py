import logging
import os
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    "us.amazon.nova-2-lite-v1:0": ("nova", ["low", "medium"]),
}

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