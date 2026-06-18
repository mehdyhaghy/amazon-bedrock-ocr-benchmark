import time
import json
import base64
import numpy as np
from PIL import ImageDraw
from typing import Dict, Any, Tuple, Optional

from engines.base import OCREngine
from shared.image_utils import convert_to_bytes
from shared.config import (
    logger,
    API_COSTS,
    MAX_IMAGE_SIZE,
    CEREBRAS_API_KEY,
    CEREBRAS_BASE_URL,
)
from shared.prompt_manager import (
    get_prompt_for_document_type,
    get_structured_extraction_instructions,
    sanitize_schema_for_structured_output,
    OCR_SYSTEM_PROMPT,
)


class CerebrasEngine(OCREngine):
    """
    OCR engine backed by the external Cerebras Inference API (native SDK).

    Uses the Cerebras Chat Completions API with base64 image inputs, mirroring
    the return contract of BedrockEngine so the shared processor/cost/results
    machinery works unchanged. See the Gemma 4 Multimodal Quick Start Guide.

    Notes from the Cerebras docs:
    - Images must be base64-encoded data URIs in image_url.url (PNG or JPEG).
    - HTTPS image URLs and PDF/video inputs are not supported.
    - Reasoning is off by default; enable via reasoning_effort
      ("none" | "low" | "medium" | "high"). On Gemma 4 today low/medium/high
      are equivalent.
    """

    def __init__(self):
        super().__init__("Cerebras")

    def _get_client(self):
        """Lazily construct the Cerebras SDK client.

        Imported lazily so the rest of the app still loads if the optional
        cerebras-cloud-sdk dependency is not installed.
        """
        try:
            from cerebras.cloud.sdk import Cerebras
        except ImportError as e:
            raise ImportError(
                "cerebras-cloud-sdk is not installed. Install it with "
                "`uv pip install cerebras-cloud-sdk` (or add it to requirements.txt)."
            ) from e

        if not CEREBRAS_API_KEY:
            raise ValueError(
                "CEREBRAS_API_KEY is not set. Set the CEREBRAS_API_KEY environment "
                "variable or add it to shared/local_settings.py."
            )

        kwargs = {"api_key": CEREBRAS_API_KEY}
        if CEREBRAS_BASE_URL:
            kwargs["base_url"] = CEREBRAS_BASE_URL
        return Cerebras(**kwargs)

    def process_image(self, image, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an image with the Cerebras Chat Completions API.

        Args:
            image: PIL Image, numpy array, or path to an image file.
                   PDFs are not supported by Cerebras.
            options: Dictionary of options including:
                - model_id: Cerebras model ID (e.g. "gemma-4-31b-trial")
                - document_type: Document type for prompt selection
                - output_schema: Optional JSON schema for structured output
                - effort_level: reasoning_effort value (None/"none"/"low"/"medium"/"high")

        Returns:
            Dictionary matching the BedrockEngine contract: text, json, image,
            process_time, token_usage, model_id, operation_type, file_type.
        """
        options = options or {}
        model_id = options.get('model_id', 'gemma-4-31b-trial')
        document_type = options.get('document_type', 'generic')
        output_schema = options.get('output_schema')
        effort_level = options.get('effort_level')

        # Cerebras does not accept PDF input.
        if self._is_pdf_input(image):
            return {
                "text": "Cerebras Error: PDF input is not supported (image-only provider).",
                "json": None,
                "image": None,
                "process_time": 0,
                "token_usage": {'inputTokens': 0, 'outputTokens': 0, 'totalTokens': 0},
                "model_id": model_id,
                "operation_type": "error",
                "pages": 0,
            }

        overall_start_time = time.time()
        timing_ctx = self.get_timing_wrapper()

        # Convert image to JPEG bytes OUTSIDE the timing context (matches BedrockEngine).
        image_bytes, img_pil = convert_to_bytes(image, MAX_IMAGE_SIZE)
        logger.info(f"Image bytes size before Cerebras call: {len(image_bytes) / 1024:.2f}KB")
        data_uri = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")

        with timing_ctx:
            try:
                client = self._get_client()

                # Build prompt (document-type guidance + structured-extraction
                # instruction; schema conformance is enforced via response_format).
                prompt = get_prompt_for_document_type(document_type)
                prompt += get_structured_extraction_instructions()

                # Structured output: enforce the schema via Cerebras
                # response_format json_schema with strict constrained decoding.
                # A bad/missing schema raises here — no prompt-based fallback.
                sanitized_schema = sanitize_schema_for_structured_output(output_schema)
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ocr_extraction",
                        "strict": True,
                        "schema": sanitized_schema,
                    },
                }

                # Per Cerebras best practices, place image content before the text.
                messages = [
                    {"role": "system", "content": OCR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]

                create_args = {
                    "model": model_id,
                    "messages": messages,
                    "temperature": 0,
                    "max_completion_tokens": 32000 if effort_level else 4000,
                    "response_format": response_format,
                }
                # reasoning_effort: None -> "none" (off). Otherwise pass through.
                create_args["reasoning_effort"] = effort_level if effort_level else "none"

                response = client.chat.completions.create(**create_args)

                # Extract text from the chat-completions response.
                extracted_text = ""
                try:
                    extracted_text = (response.choices[0].message.content or "").strip()
                except (AttributeError, IndexError) as parse_err:
                    logger.warning(f"Unexpected Cerebras response shape: {parse_err}")
                    extracted_text = ""

                # Strict structured output returns raw JSON with no markdown
                # fences. Keep only a defensive whitespace trim.
                extracted_text = extracted_text.strip()

                # Token usage — map Cerebras/OpenAI field names to the shared contract.
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
                total_tokens = getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens)
                token_usage = {
                    'inputTokens': input_tokens,
                    'outputTokens': output_tokens,
                    'totalTokens': total_tokens,
                }
                logger.info(
                    f"Cerebras token usage - Input: {input_tokens}, "
                    f"Output: {output_tokens}, Total: {total_tokens}"
                )

                # Annotated image (same visual treatment as BedrockEngine).
                annotated_img_copy = img_pil.copy()
                draw = ImageDraw.Draw(annotated_img_copy)
                width, height = annotated_img_copy.size
                draw.rectangle([(0, 0), (width, height)], outline='#FF6B00', width=10)
                draw.text((20, 20), f"Processed with Cerebras {model_id} ({width}x{height})", fill='#FF6B00')
                annotated_image = np.array(annotated_img_copy)

                # Parse the JSON. Structured output (strict) guarantees
                # schema-valid JSON, so parse strictly — any failure propagates
                # to the outer except and surfaces as an error result (no lenient
                # repair / no fallback).
                structured_json = json.loads(extracted_text)

                overall_process_time = time.time() - overall_start_time
                logger.info(f"Cerebras total processing time: {overall_process_time:.2f} seconds")

                return {
                    "text": extracted_text,
                    "json": structured_json,
                    "image": annotated_image,
                    "process_time": overall_process_time,
                    "token_usage": token_usage,
                    "model_id": model_id,
                    "pages": 1,
                    "operation_type": "cerebras",
                    "file_type": "image",
                }

            except Exception as e:
                logger.error(f"Error in Cerebras processing: {str(e)}")
                overall_process_time = time.time() - overall_start_time
                return {
                    "text": f"Cerebras Error: {str(e)}",
                    "json": None,
                    "image": None,
                    "process_time": overall_process_time,
                    "token_usage": {'inputTokens': 0, 'outputTokens': 0, 'totalTokens': 0},
                    "model_id": model_id,
                    "operation_type": "error",
                    "pages": 0,
                }

    def get_cost(self, result: Dict[str, Any]) -> Tuple[str, float]:
        """
        Calculate cost for Cerebras processing using the shared API_COSTS table.
        Mirrors BedrockEngine.get_cost so the contract is identical.
        """
        token_usage = result.get('token_usage')
        model_id = result.get('model_id', '')

        if not token_usage or model_id not in API_COSTS.get('bedrock', {}):
            return '<div class="cost-none">No cost data available</div>', 0.0

        model_costs = API_COSTS['bedrock'][model_id]
        cost_per_1k_input = model_costs['input']
        cost_per_1k_output = model_costs['output']

        input_tokens = token_usage.get('inputTokens', 0)
        output_tokens = token_usage.get('outputTokens', 0)

        input_cost = (input_tokens / 1000) * cost_per_1k_input
        output_cost = (output_tokens / 1000) * cost_per_1k_output
        total_cost = input_cost + output_cost

        html = f'''
        <div class="cost-container">
            <div class="cost-total">${total_cost:.6f} total</div>
            <div class="cost-breakdown">
                <span>${input_cost:.6f} for {input_tokens} input tokens (${cost_per_1k_input:.6f}/1K tokens)</span><br>
                <span>${output_cost:.6f} for {output_tokens} output tokens (${cost_per_1k_output:.6f}/1K tokens)</span>
            </div>
        </div>
        '''
        return html, total_cost

    def _is_pdf_input(self, image) -> bool:
        """Check if input is a PDF file (unsupported by Cerebras)."""
        if hasattr(image, 'name') and image.name and image.name.lower().endswith('.pdf'):
            return True
        if isinstance(image, str) and image.lower().endswith('.pdf'):
            return True
        return False
