
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


def get_json_only_instructions(output_schema=None):
    """Prompt guidance for the capability-gated path (models that reject the
    Converse `outputConfig` field).

    Without constrained decoding these models tend to return markdown prose
    instead of JSON, so — unlike get_structured_extraction_instructions(), which
    assumes the schema is enforced by the API — this explicitly instructs the
    model to emit a single raw JSON object conforming to the schema. The base
    document-type prompt and OCR system prompt are unchanged, so the two paths
    stay as similar as possible; only this trailing instruction differs.
    """
    instr = (
        "\n\nExtract the document's information into every applicable field of the "
        "provided JSON schema. Map each piece of text to its most specific matching "
        "field rather than placing everything into one field. Leave a field empty "
        "only when the document genuinely contains no value for it. Preserve values "
        "exactly as they appear.\n\n"
        "Return ONLY a single raw JSON object that conforms to the schema. Do not "
        "include any explanation, prose, headings, or markdown code fences — output "
        "must start with '{' and end with '}'."
    )
    if output_schema:
        schema_text = output_schema if isinstance(output_schema, str) else json.dumps(output_schema)
        instr += f"\n\nJSON schema:\n{schema_text}"
    return instr


def _repair_unclosed_json(text):
    """Append the closing tokens needed to balance a structurally-truncated
    JSON object/array.

    Walks `text` (which must start at the opening '{' or '[') tracking a stack
    of open '{'/'[' while respecting string literals and escapes. If the text
    ends with unclosed openers, returns `text` plus the matching closers in the
    correct (reverse) order. Returns None if the structure is already balanced
    or is otherwise unrepairable by simple closing (e.g. ends inside an open
    string literal).

    This is intentionally conservative: it ONLY appends '}'/']' and never edits
    existing characters, so it cannot invent or change field values — it can
    only close an object the model failed to terminate.
    """
    stack = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()

    # If we ended inside an open string, we can't safely repair (would need to
    # guess where the value ends). Bail out.
    if in_string or not stack:
        return None

    closers = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    return text + closers


def parse_structured_output_fallback(raw_text):
    """Tolerant JSON extraction for models that do NOT support Converse
    structured output (outputConfig).

    The default path uses constrained decoding and a strict json.loads. This
    helper is ONLY used on the capability-gated path where the model produced
    free-form text guided by the prompt. It recovers a JSON object from common
    wrappers the model may add:
      - markdown code fences (```json ... ``` or ``` ... ```)
      - leading/trailing prose around the JSON object
      - a trailing comma before a closing brace/bracket

    It does NOT attempt aggressive repair. If a valid JSON object cannot be
    recovered it raises ValueError so the caller surfaces an error result
    (no silent degradation).

    Returns the parsed dict.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response; no JSON to parse.")

    text = raw_text.strip()

    # Strip a single leading/trailing markdown code fence if present.
    if text.startswith("```"):
        # remove opening fence line (``` or ```json)
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    # First, try a direct parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to extracting the outermost {...} object via brace matching.
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")

    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        # The model under-generated and left the root object unclosed (observed
        # with Llama 4 Maverick on the large insurance_claim schema: it emitted
        # stop_reason=end_turn but was missing trailing closing brace(s)). Try a
        # bounded repair: append the closers needed to balance unclosed
        # {/[ openers, in the correct order. This ONLY appends } / ] — it never
        # alters or removes existing content — so it can't fabricate field
        # values, just close a structurally-truncated object.
        repaired = _repair_unclosed_json(text[start:])
        if repaired is not None:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                # Also strip trailing commas before the appended closers.
                import re as _re
                repaired2 = _re.sub(r",(\s*[}\]])", r"\1", repaired)
                try:
                    return json.loads(repaired2)
                except json.JSONDecodeError:
                    pass
        raise ValueError("Unbalanced JSON object in model response.")

    candidate = text[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Last permitted normalization: remove trailing commas before } or ].
        import re as _re
        candidate2 = _re.sub(r",(\s*[}\]])", r"\1", candidate)
        try:
            return json.loads(candidate2)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON from model response: {e}") from e


def get_prompt_for_document_type(document_type="generic"):
    """
    Get appropriate user prompt based on document type
    
    Args:
        document_type: Type of document (generic, form, receipt, table, handwritten)
        
    Returns:
        str: The appropriate user prompt
    """
    return DOCUMENT_TYPE_PROMPT_MAP.get(document_type.lower(), DOCUMENT_TYPE_PROMPT_MAP["generic"])

def _unwrap_schema_envelope(data):
    """Unwrap a JSON Schema envelope that some models emit instead of plain data.

    When asked to "format output according to this JSON schema", some models
    echo the SCHEMA shape instead of a flat populated object. Two variants are
    handled, recursively and in combination:

      1. Object envelope:
           {"type": "object", "properties": {<fields>}}   ->  {<fields>}
      2. Per-field value wrapper:
           {"type": "string", "value": "JOHN DOE"}        ->  "JOHN DOE"
           {"type": "array",  "items": [...]}             ->  [...]

    Example (Llama 4 Maverick on insurance_claim):
        {"type":"object","properties":{
            "form_title":{"type":"string","value":"HEALTH INSURANCE CLAIM FORM"},
            "5_patient_address":{"type":"object","properties":{
                "city":{"type":"string","value":"Any City"}}}}}
    unwraps to:
        {"form_title":"HEALTH INSURANCE CLAIM FORM",
         "5_patient_address":{"city":"Any City"}}

    A genuine JSON *schema* (fields shaped like {"type":"string"} with NO
    "value"/"items"/"properties" payload) is left unchanged, so we never
    destroy a real schema definition or invent values.
    """
    _SCHEMA_STRUCTURAL_KEYS = {
        "type", "properties", "items", "value", "$schema", "required",
        "title", "description", "inferenceType", "instruction", "enum", "format",
        "additionalProperties",
    }

    def _is_value_wrapper(d):
        # {"type": <str>, "value": X}  (X may be None) — a wrapped scalar value.
        return (
            isinstance(d, dict)
            and "value" in d
            and "type" in d
            and set(d.keys()) <= _SCHEMA_STRUCTURAL_KEYS
        )

    def _is_bare_schema_fragment(v):
        # A genuine schema field: {"type": "string"} with NO data payload
        # (no value/items, and properties only if it's a nested schema).
        return (
            isinstance(v, dict)
            and "type" in v
            and "value" not in v
            and "items" not in v
            and "properties" not in v
            and set(v.keys()) <= _SCHEMA_STRUCTURAL_KEYS
        )

    def _is_object_envelope(d):
        # {"type":"object"?, "properties": {...}} with only structural keys.
        # Accept when it holds DATA (BDA: bare resolved values; or value/items/
        # properties wrappers). Reject a genuine schema where EVERY property is a
        # bare {"type": ...} fragment (no values), so real schemas pass through.
        if not (
            isinstance(d, dict)
            and isinstance(d.get("properties"), dict)
            and d["properties"]
            and set(d.keys()) <= _SCHEMA_STRUCTURAL_KEYS
        ):
            return False
        return not all(_is_bare_schema_fragment(v) for v in d["properties"].values())

    def _is_array_wrapper(d):
        # {"type":"array", "items":[...]} with only structural keys.
        return (
            isinstance(d, dict)
            and isinstance(d.get("items"), list)
            and set(d.keys()) <= _SCHEMA_STRUCTURAL_KEYS
        )

    def _unwrap(node, changed):
        if isinstance(node, dict):
            if _is_value_wrapper(node):
                changed[0] = True
                return _unwrap(node["value"], changed)
            if _is_object_envelope(node):
                changed[0] = True
                return {k: _unwrap(v, changed) for k, v in node["properties"].items()}
            if _is_array_wrapper(node):
                changed[0] = True
                return [_unwrap(v, changed) for v in node["items"]]
            # Plain object: recurse into each value.
            return {k: _unwrap(v, changed) for k, v in node.items()}
        if isinstance(node, list):
            return [_unwrap(v, changed) for v in node]
        return node

    if not isinstance(data, (dict, list)):
        return data
    changed = [False]
    result = _unwrap(data, changed)
    if changed[0]:
        logger.info("Unwrapped JSON Schema envelope from model response")
    return result


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
            structured_json = _unwrap_schema_envelope(structured_json)
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
