"""
Example local settings.

Copy this file to ``shared/local_settings.py`` and fill in your own
values. ``local_settings.py`` is gitignored so your real bucket and
account ID never get committed.

Alternatively, set the ``OCR_S3_BUCKET`` environment variable.
"""

# S3 bucket used by Textract (for PDFs) and BDA. Leave as "" to require
# the user to type one in the UI at runtime.
DEFAULT_S3_BUCKET = ""

# API key for the external Cerebras Inference provider (Gemma 4, etc.).
# Leave as "" to fall back to the CEREBRAS_API_KEY environment variable.
CEREBRAS_API_KEY = ""
