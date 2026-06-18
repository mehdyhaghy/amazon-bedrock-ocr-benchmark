
# prompt_manager.py
import json
import time
import logging

from .aws_client import get_aws_client
from .config import POSTPROCESSING_MODEL, logger

OCR_SYSTEM_PROMPT = """
You are an expert OCR and document analysis system. Extract all text and information from the image 
with high accuracy. Organize the results clearly and preserve the structure of the document. 
Pay special attention to tables, forms, and formatted data. Use the provided schema to structure 
your response if one is specified.
"""

DOCUMENT_TYPE_PROMPT_MAP = {
    "generic": """
Extract all the text from the provided image. Preserve the original structure and formatting as much as possible.
For tables, maintain the rows and columns relationships. For forms, associate labels with their corresponding values.
For multi-column layouts, process each column in order from left to right.
""",
    "form": """
This image contains a form or receipt. Please extract all fields and their values, preserving the 
relationships between labels and data. Format the output as key-value pairs.
""",
    "receipt": """
This image contains a form or receipt. Please extract all fields and their values, preserving the 
relationships between labels and data. Format the output as key-value pairs.
""",
    "table": """
This image contains tabular data. Extract the complete table including headers and all cell values.
Preserve the row and column structure precisely.
""",
    "handwritten": """
This image contains handwritten text. Try to extract all the handwritten content with high accuracy.
If parts are illegible, indicate that with [illegible].
"""
}

JSON_SYSTEM_PROMPT = """
You are an AI assistant specialized in structuring extracted document text into JSON format.
Your task is to analyze the document content and create the most appropriate JSON structure.
Focus on capturing all relevant information in a logical hierarchy.
Ensure the JSON is valid and represents the information accurately.
IMPORTANT: When returning JSON, do not use code blocks, backticks or markdown formatting.
"""


def get_structured_extraction_instructions():
    """Prompt guidance used WITH schema-enforced structured output.

    Schema enforcement guarantees the response shape (valid JSON, allowed keys,
    types) but does NOT make the model populate fields — unrequired fields may
    be left empty, so the model can dump everything into one permissive string
    field. This instruction tells the model to actually extract each field from
    the document into its matching schema key. It deliberately omits the old
    "return only valid JSON / escape quotes / no markdown" language, which is
    now handled by constrained decoding.
    """
    return (
        "\n\nExtract the document's information into every applicable field of the "
        "provided output structure. Map each piece of text to its most specific "
        "matching field rather than placing everything into a single field. Leave a "
        "field empty only when the document genuinely contains no value for it. "
        "Preserve values exactly as they appear in the document."
    )


# Keywords unsupported by the strict structured-output paths of the providers
# we target. Cerebras strict mode rejects pattern/format/min*/max*/array bounds;
# Bedrock rejects numeric and length constraints. We strip the union of both so
# a single sanitized schema is accepted by either provider.
_UNSUPPORTED_SCHEMA_KEYWORDS = {
    "pattern", "format",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength",
    "minItems", "maxItems", "uniqueItems",
    "minProperties", "maxProperties",
    "default", "examples", "$comment", "readOnly", "writeOnly",
}


def sanitize_schema_for_structured_output(output_schema):
    """Normalize a user-provided JSON schema into the common strict subset
    accepted by both Bedrock Converse (outputConfig.textFormat) and Cerebras
    (response_format json_schema strict).

    Transformations:
      - Parse a string schema into a dict (raises on invalid JSON).
      - Require the root to be an object schema.
      - Recursively set ``additionalProperties: false`` on every object.
      - Strip keywords unsupported by the strict paths.

    Raises ValueError if the schema is missing, unparseable, or not an object
    schema. There is intentionally NO fallback: an unusable schema must fail
    loudly rather than silently degrade to prompt-based extraction.

    Returns the sanitized schema as a dict.
    """
    if output_schema is None:
        raise ValueError("Structured output requires a JSON schema, but none was provided.")

    if isinstance(output_schema, str):
        text = output_schema.strip()
        if not text:
            raise ValueError("Structured output requires a non-empty JSON schema.")
        try:
            schema = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Output schema is not valid JSON: {e}") from e
    elif isinstance(output_schema, dict):
        schema = output_schema
    else:
        raise ValueError(f"Unsupported schema type: {type(output_schema).__name__}")

    if not isinstance(schema, dict):
        raise ValueError("Output schema must be a JSON object.")

    # A bare/generic placeholder like {"type": "object"} with no properties is
    # not useful for enforcement — treat as unusable.
    if schema.get("type") != "object":
        raise ValueError('Structured output schema root must be {"type": "object"}.')
    if not isinstance(schema.get("properties"), dict) or not schema["properties"]:
        raise ValueError("Structured output schema must define at least one property.")

    def _clean(node):
        if isinstance(node, dict):
            cleaned = {}
            for k, v in node.items():
                if k in _UNSUPPORTED_SCHEMA_KEYWORDS:
                    continue
                cleaned[k] = _clean(v)
            # Enforce additionalProperties:false on every object node.
            if cleaned.get("type") == "object" or "properties" in cleaned:
                cleaned["additionalProperties"] = False
            return cleaned
        if isinstance(node, list):
            return [_clean(v) for v in node]
        return node

    return _clean(schema)


def get_prompt_for_document_type(document_type="generic"):
    """
    Get appropriate user prompt based on document type
    
    Args:
        document_type: Type of document (generic, form, receipt, table, handwritten)
        
    Returns:
        str: The appropriate user prompt
    """
    return DOCUMENT_TYPE_PROMPT_MAP.get(document_type.lower(), DOCUMENT_TYPE_PROMPT_MAP["generic"])

def process_text_with_llm(text, output_schema=None):
    """
    Process extracted text with LLM to structure it as JSON
    
    Args:
        text: Extracted text to process
        output_schema: Optional JSON schema to conform to
        
    Returns:
        Tuple of (structured JSON, token usage)
    """
    # Start timing
    start_time = time.time()
    
    # Create Bedrock Runtime client
    bedrock_runtime = get_aws_client('bedrock-runtime')
    
    try:
        # Prepare the prompt for JSON conversion
        prompt = f"""Please convert the following extracted text into structured JSON format:

{text}
"""
        if output_schema:
            prompt += f"\n\nPlease format the output according to this JSON schema: {output_schema}"
            prompt += "\nIMPORTANT: Return ONLY the JSON data without any markdown code blocks, backticks or formatting. Ensure all quotes and special characters are properly escaped."
        else:
            prompt += "\n\n" + JSON_TEMPLATE_NO_SCHEMA
        
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        system_messages = [{"text": JSON_SYSTEM_PROMPT}]
        
        logger.info(f"Calling Bedrock with model: {POSTPROCESSING_MODEL}")
        
        response = bedrock_runtime.converse(
            modelId=POSTPROCESSING_MODEL,
            messages=messages,
            system=system_messages
        )
        
        # Extract text and token usage
        structured_text = ""
        token_usage = {
            'inputTokens': response.get('usage', {}).get('inputTokens', 0),
            'outputTokens': response.get('usage', {}).get('outputTokens', 0),
            'totalTokens': response.get('usage', {}).get('totalTokens', 0)
        }
        
        logger.info(f"Token usage - Input: {token_usage['inputTokens']}, Output: {token_usage['outputTokens']}")
        
        if 'output' in response and 'message' in response['output']:
            message = response['output']['message']
            if 'content' in message:
                for content_item in message['content']:
                    if 'text' in content_item:
                        text = content_item['text'].strip()
                        if text.startswith("```json"):
                            text = text[7:]
                        elif text.startswith("```"):
                            text = text[3:]
                        if text.endswith("```"):
                            text = text[:-3]
                        structured_text += text.strip()
        
        try:
            structured_json = json.loads(structured_text)
            json_process_time = time.time() - start_time
            logger.info(f"JSON conversion completed in {json_process_time:.2f} seconds")
            return structured_json, token_usage
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            
            error_pos = e.pos
            context_range = 20
            start_pos = max(0, error_pos - context_range)
            end_pos = min(len(structured_text), error_pos + context_range)
            error_context = structured_text[start_pos:end_pos]
            logger.error(f"Context around error: '...{error_context}...'")
            
            json_process_time = time.time() - start_time
            logger.info(f"JSON conversion failed in {json_process_time:.2f} seconds")
            return {"error": "Failed to parse JSON", "raw_text": structured_text}, token_usage
            
    except Exception as e:
        logger.error(f"Error in LLM processing: {str(e)}")
        json_process_time = time.time() - start_time
        logger.info(f"JSON conversion failed in {json_process_time:.2f} seconds")
        raise
