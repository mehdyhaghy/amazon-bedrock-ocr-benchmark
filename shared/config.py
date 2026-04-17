import logging
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Image size constants
MAX_IMAGE_SIZE = 5 * 1024 * 1024 - 100000  # 5MB minus buffer for Bedrock

# Available Bedrock models
BEDROCK_MODELS = {
    "Claude Opus 4.7": "us.anthropic.claude-opus-4-7",
    "Claude Sonnet 4.6": "us.anthropic.claude-sonnet-4-6",
    "Claude Haiku 4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "Amazon Nova 2 Lite": "us.amazon.nova-2-lite-v1:0",
    "Pixtral Large": "us.mistral.pixtral-large-2502-v1:0",
    "Mistral Large 3": "mistral.mistral-large-3-675b-instruct",
    "Llama 4 Maverick 17B": "us.meta.llama4-maverick-17b-instruct-v1:0",
    "Llama 4 Scout 17B": "us.meta.llama4-scout-17b-instruct-v1:0"
}

# Default model for post-processing
POSTPROCESSING_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Effort levels per model: maps model_id to (thinking_type, list_of_levels)
# - "adaptive": Claude Opus 4.7 / Sonnet 4.6 — uses thinking.type="adaptive" + effort param via invoke_model
# - "budget": Claude Sonnet 4 / Haiku 4.5 — uses thinking.type="enabled" + budget_tokens via invoke_model
# - "nova": Nova 2 Lite — uses reasoningConfig via converse additionalModelRequestFields
EFFORT_LEVELS = {
    "us.anthropic.claude-opus-4-7": ("adaptive", ["low", "medium", "high", "max"]),
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
        'us.anthropic.claude-opus-4-7': {
            'input': 0.005 / 1000,   # $5.00 per 1M input tokens
            'output': 0.025 / 1000   # $25.00 per 1M output tokens
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