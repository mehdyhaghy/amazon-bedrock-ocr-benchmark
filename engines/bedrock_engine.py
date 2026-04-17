import time
import json
import base64
import os
import tempfile
import shutil
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, Any, Tuple, Optional

from engines.base import OCREngine
from shared.aws_client import get_aws_client
from shared.image_utils import convert_to_bytes
from shared.config import logger, API_COSTS, MAX_IMAGE_SIZE
from shared.prompt_manager import get_prompt_for_document_type, get_json_formatting_instructions, OCR_SYSTEM_PROMPT

def _build_thinking_params(model_id, effort_level):
    """Build reasoning params for the Converse API additionalModelRequestFields.
    effort_level=None means thinking is disabled (standard call).
    Field name is provider-specific: Claude uses 'thinking', Nova uses 'reasoningConfig'."""
    from shared.config import EFFORT_LEVELS
    if effort_level is None or model_id not in EFFORT_LEVELS:
        return {}
    thinking_type, _ = EFFORT_LEVELS[model_id]
    if thinking_type == "adaptive":
        # Claude Opus 4.7 / Sonnet 4.6 — adaptive thinking with effort in output_config
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort_level}}
    elif thinking_type == "budget":
        # Claude Sonnet 4 / Haiku 4.5 — manual budget_tokens
        budget = effort_level if isinstance(effort_level, int) else 4096
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    elif thinking_type == "nova":
        # Nova 2 Lite — reasoningConfig nested inside inferenceConfig
        return {"inferenceConfig": {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": effort_level}}}
    return {}

class BedrockEngine(OCREngine):
    """
    Implementation of OCR engine using Amazon Bedrock
    """
    
    def __init__(self):
        super().__init__("Bedrock")
    
    def process_image(self, image, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an image or PDF with Amazon Bedrock using converse API
        
        Args:
            image: PIL Image, numpy array, path to image, or PDF file
            options: Dictionary of options including:
                - model_id: Bedrock model ID
                - document_type: Type of document (generic, form, receipt, table, handwritten)
                - output_schema: JSON schema for structuring the output
                
        Returns:
            Dictionary containing results including:
            - text: Extracted text
            - image: Annotated image
            - process_time: Processing time
            - token_usage: Token usage information
            - model_id: Model ID used
        """

        options = options or {}
        model_id = options.get('model_id', '')
        document_type = options.get('document_type', 'generic')
        output_schema = options.get('output_schema')
        effort_level = options.get('effort_level')
        
        reasoning_config = _build_thinking_params(model_id, effort_level)
        
        overall_start_time = time.time()
        # Set up timing context manager
        timing_ctx = self.get_timing_wrapper()
        
        # Check if input is a PDF file
        is_pdf = self._is_pdf_input(image)
        
        if is_pdf:
            # Handle PDF files - copy to temp with clean name
            temp_pdf_path = None
            try:
                # Get original file content
                if hasattr(image, 'name') and image.name:
                    with open(image.name, 'rb') as f:
                        file_bytes = f.read()
                elif isinstance(image, str):
                    with open(image, 'rb') as f:
                        file_bytes = f.read()
                else:
                    raise ValueError("PDF input must be a file path or file object")
                
                logger.info(f"PDF file size before API call: {len(file_bytes) / 1024:.2f}KB")
                
                # Create temporary PDF with clean name
                temp_pdf_path = self._create_temp_pdf(file_bytes)
                logger.info(f"Created temporary PDF: {temp_pdf_path}")
                
                img_pil = None  # No PIL image for PDF
                
            except Exception as e:
                logger.error(f"Error handling PDF file: {str(e)}")
                if temp_pdf_path and os.path.exists(temp_pdf_path):
                    os.unlink(temp_pdf_path)
                raise
        else:
            # Convert image to bytes OUTSIDE the timing context
            image_bytes, img_pil = convert_to_bytes(image, MAX_IMAGE_SIZE)
            logger.info(f"Image bytes size before API call: {len(image_bytes) / 1024:.2f}KB")
        
        # Start timing for the actual processing
        with timing_ctx:
            
            try:
                # Create Bedrock Runtime client
                bedrock_runtime = get_aws_client('bedrock-runtime')
                
                # Get appropriate prompt based on document type
                prompt = get_prompt_for_document_type(document_type)
                prompt += get_json_formatting_instructions(output_schema)
                system_prompt = OCR_SYSTEM_PROMPT
                    
                # Create request payload based on file type
                if is_pdf:
                    # For PDF files, use converse API with document content block
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "document": {
                                        "format": "pdf",
                                        "name": "document",
                                        "source": {
                                            "bytes": file_bytes
                                        }
                                    }
                                },
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                    
                    converse_args = {
                        "modelId": model_id,
                        "messages": messages,
                        "system": [{"text": system_prompt}],
                        "inferenceConfig": {"maxTokens": 32000 if reasoning_config else 4000}
                    }
                    if reasoning_config:
                        converse_args["additionalModelRequestFields"] = reasoning_config
                    
                    response = bedrock_runtime.converse(**converse_args)
                    
                    # Extract text from converse response
                    extracted_text = ""
                    if 'output' in response and 'message' in response['output']:
                        for content_item in response['output']['message'].get('content', []):
                            if content_item.get('type') == 'thinking' or 'reasoningContent' in content_item:
                                continue
                            if 'text' in content_item:
                                text = content_item['text'].strip()
                                if text.startswith("```json"):
                                    text = text[7:]
                                if text.startswith("```"):
                                    text = text[3:]
                                if text.endswith("```"):
                                    text = text[:-3]
                                extracted_text += text.strip()
                    
                    # Extract token usage
                    token_usage = {
                        'inputTokens': response.get('usage', {}).get('inputTokens', 0),
                        'outputTokens': response.get('usage', {}).get('outputTokens', 0),
                        'totalTokens': response.get('usage', {}).get('totalTokens', 0)
                    }
                    
                    logger.info(f"PDF processed with converse - Input: {token_usage['inputTokens']}, Output: {token_usage['outputTokens']}")
                    
                else:
                    # For images, use converse API
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": prompt
                                },
                                {
                                    "image": {
                                        "format": "jpeg",
                                        "source": {
                                            "bytes": image_bytes
                                        }
                                    }
                                }
                            ]
                        }
                    ]
                    
                    converse_args = {
                        "modelId": model_id,
                        "messages": messages,
                        "system": [{"text": system_prompt}],
                        "inferenceConfig": {"maxTokens": 32000 if reasoning_config else 4000}
                    }
                    if reasoning_config:
                        converse_args["additionalModelRequestFields"] = reasoning_config
                    
                    response = bedrock_runtime.converse(**converse_args)
                    
                    # Extract text from converse response
                    extracted_text = ""
                    
                    # Extract token usage information
                    token_usage = {
                        'inputTokens': response.get('usage', {}).get('inputTokens', 0),
                        'outputTokens': response.get('usage', {}).get('outputTokens', 0),
                        'totalTokens': response.get('usage', {}).get('totalTokens', 0)
                    }
                    
                    logger.info(f"Token usage - Input: {token_usage['inputTokens']}, Output: {token_usage['outputTokens']}, Total: {token_usage['totalTokens']}")
                    
                    # Process response according to the provided format
                    if 'output' in response and 'message' in response['output']:
                        message = response['output']['message']
                        if 'content' in message:
                            for content_item in message['content']:
                                if content_item.get('type') == 'thinking' or 'reasoningContent' in content_item:
                                    continue
                                if 'text' in content_item:
                                    text = content_item['text']
                                    # Remove any markdown code block wrapping
                                    text = text.strip()
                                    if text.startswith("```json"):
                                        text = text[7:]
                                    if text.startswith("```"):
                                        text = text[3:]
                                    if text.endswith("```"):
                                        text = text[:-3]
                                    extracted_text += text.strip()
                
                # Create visual annotation based on file type
                if is_pdf:
                    # For PDF files, create a simple placeholder image
                    annotated_image = np.zeros((400, 600, 3), dtype=np.uint8)
                    from PIL import Image as PILImage, ImageDraw as PILImageDraw
                    pil_img = PILImage.fromarray(annotated_image)
                    draw = PILImageDraw.Draw(pil_img)
                    model_name = model_id.split(':')[0].split('.')[-1].upper()
                    draw.text((20, 20), f"PDF Processed with {model_name}", fill=(0, 204, 255))
                    draw.text((20, 50), f"Document Type: {document_type}", fill=(0, 204, 255))
                    annotated_image = np.array(pil_img)
                else:
                    # Create a visual indicator on the image
                    annotated_img_copy = img_pil.copy()
                    draw = ImageDraw.Draw(annotated_img_copy)
                    width, height = annotated_img_copy.size
                    
                    # Draw border
                    border_width = 10
                    draw.rectangle(
                        [(0, 0), (width, height)],
                        outline='#00CCFF',
                        width=border_width
                    )
                    
                    # Add model info text
                    model_name = model_id.split(':')[0].split('.')[-1].upper()
                    draw.text(
                        (20, 20),
                        f"Processed with {model_name} ({width}x{height})",
                        fill='#00CCFF'
                    )
                    
                    # Convert to numpy array
                    annotated_image = np.array(annotated_img_copy)
                
                # Try to parse the JSON
                structured_json = None
                try:
                    structured_json = json.loads(extracted_text)
                except json.JSONDecodeError:
                    # Try to extract JSON from mixed content (find first { ... last })
                    try:
                        start = extracted_text.find('{')
                        end = extracted_text.rfind('}')
                        if start != -1 and end > start:
                            candidate = extracted_text[start:end + 1]
                            # Fix common model-introduced JSON issues:
                            # Chinese colons, BOMs, smart quotes, unescaped newlines in strings
                            candidate = (candidate
                                         .replace('\ufeff', '')
                                         .replace('：', ':')
                                         .replace('\u201c', '"').replace('\u201d', '"')
                                         .replace('\u2018', "'").replace('\u2019', "'"))
                            # Remove trailing commas before } or ] (invalid in JSON but some
                            # models produce them)
                            import re
                            candidate = re.sub(r',(\s*[}\]])', r'\1', candidate)
                            try:
                                structured_json = json.loads(candidate, strict=False)
                            except json.JSONDecodeError:
                                # Last resort: replace raw control chars in string regions
                                # by escaping newlines/tabs that appear inside quoted strings
                                def _escape_ctrl_in_strings(s):
                                    # Walk through and escape newlines/tabs inside "..." regions
                                    out = []
                                    in_str = False
                                    i = 0
                                    while i < len(s):
                                        c = s[i]
                                        if c == '"' and (i == 0 or s[i - 1] != '\\'):
                                            in_str = not in_str
                                            out.append(c)
                                        elif in_str and c == '\n':
                                            out.append('\\n')
                                        elif in_str and c == '\t':
                                            out.append('\\t')
                                        elif in_str and c == '\r':
                                            out.append('\\r')
                                        else:
                                            out.append(c)
                                        i += 1
                                    return ''.join(out)
                                try:
                                    structured_json = json.loads(_escape_ctrl_in_strings(candidate), strict=False)
                                except json.JSONDecodeError:
                                    structured_json = {"text": extracted_text}
                        else:
                            structured_json = {"text": extracted_text}
                    except json.JSONDecodeError:
                        structured_json = {"text": extracted_text}
                
                # Unwrap schema-style responses: when the model echoes the schema structure
                # and nests the actual extracted values inside "properties"
                if (isinstance(structured_json, dict)
                        and set(structured_json.keys()) <= {"type", "properties", "required", "items", "$schema"}
                        and isinstance(structured_json.get("properties"), dict)):
                    structured_json = structured_json["properties"]
                
                # Unwrap per-field {"type":"string","value":...} wrappers that some
                # models (e.g. Llama 4) produce when given a JSON schema prompt.
                def _unwrap_field_values(obj):
                    if isinstance(obj, dict):
                        # Schema-field wrapper with a value: return the value
                        if "value" in obj and set(obj.keys()) <= {"type", "value", "description", "format", "enum"}:
                            return _unwrap_field_values(obj["value"])
                        # Schema-field wrapper without value: treat as empty/None
                        if "type" in obj and set(obj.keys()) <= {"type", "description", "format", "enum"}:
                            return None
                        return {k: _unwrap_field_values(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_unwrap_field_values(v) for v in obj]
                    return obj
                structured_json = _unwrap_field_values(structured_json)
                
                logger.info(f"Bedrock processing completed in {timing_ctx.process_time:.2f} seconds")
                overall_process_time = time.time() - overall_start_time
                logger.info(f"Bedrock total processing time: {overall_process_time:.2f} seconds")

                # Clean up temporary PDF file if created
                if is_pdf and temp_pdf_path and os.path.exists(temp_pdf_path):
                    try:
                        os.unlink(temp_pdf_path)
                        logger.info(f"Cleaned up temporary PDF: {temp_pdf_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temporary PDF: {cleanup_error}")

                # Return dictionary with all necessary information
                return {
                    "text": extracted_text,
                    "json": structured_json,
                    "image": annotated_image,
                    "process_time": overall_process_time,
                    "token_usage": token_usage,
                    "model_id": model_id,
                    "pages": 1,
                    "operation_type": "bedrock",
                    "file_type": "pdf" if is_pdf else "image"
                }
                
            except Exception as e:
                logger.error(f"Error in Bedrock processing: {str(e)}")
                overall_process_time = time.time() - overall_start_time
                logger.info(f"Bedrock error processing time: {overall_process_time:.2f} seconds")
                
                # Clean up temporary PDF file if created
                if is_pdf and temp_pdf_path and os.path.exists(temp_pdf_path):
                    try:
                        os.unlink(temp_pdf_path)
                        logger.info(f"Cleaned up temporary PDF after error: {temp_pdf_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temporary PDF after error: {cleanup_error}")
                
                return {
                    "text": f"Amazon Bedrock Error: {str(e)}",
                    "json": None,
                    "image": None,
                    "process_time": overall_process_time,
                    "token_usage": {'inputTokens': 0, 'outputTokens': 0, 'totalTokens': 0},
                    "model_id": model_id,
                    "operation_type": "error",
                    "pages": 0
                }
    
    def get_cost(self, result: Dict[str, Any]) -> Tuple[str, float]:
        """
        Calculate the cost for Bedrock processing
        
        Args:
            result: Result dictionary from process_image
            
        Returns:
            Tuple of (HTML representation of cost, actual cost value)
        """
        token_usage = result.get('token_usage')
        model_id = result.get('model_id', '')
        
        if not token_usage or model_id not in API_COSTS.get('bedrock', {}):
            return '<div class="cost-none">No cost data available</div>', 0.0
            
        # Get cost per token for the model from the correct structure
        model_costs = API_COSTS['bedrock'][model_id]
        cost_per_1k_input = model_costs['input']
        cost_per_1k_output = model_costs['output']
        
        # Calculate cost
        input_tokens = token_usage.get('inputTokens', 0)
        output_tokens = token_usage.get('outputTokens', 0)
        
        input_cost = (input_tokens / 1000) * cost_per_1k_input
        output_cost = (output_tokens / 1000) * cost_per_1k_output
        total_cost = input_cost + output_cost
        
        # Format HTML output
        html = f'''
        <div class="cost-container">
            <div class="cost-total">${total_cost:.6f} total</div>
            <div class="cost-breakdown">
                <span>${input_cost:.6f} for {input_tokens} input tokens (${cost_per_1k_input:.6f}/1K tokens)</span><br>
                <span>${output_cost:.6f} for {output_tokens} output tokens (${cost_per_1k_output:.6f}/1K tokens)</span>
            </div>
        </div>
        '''
        
        # Return both the HTML and the actual cost value
        return html, total_cost
    
    def _is_pdf_input(self, image):
        """Check if input is a PDF file"""
        if hasattr(image, 'name') and image.name and image.name.lower().endswith('.pdf'):
            return True
        elif isinstance(image, str) and image.lower().endswith('.pdf'):
            return True
        return False
    
    def _create_temp_pdf(self, file_bytes):
        """
        Create a temporary PDF file with clean name
        
        Args:
            file_bytes: PDF file content as bytes
            
        Returns:
            str: Path to temporary PDF file
        """
        # Create temporary file with clean name
        temp_dir = tempfile.gettempdir()
        temp_filename = f"bedrock_temp_{int(time.time())}_{os.getpid()}.pdf"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        try:
            with open(temp_path, 'wb') as temp_file:
                temp_file.write(file_bytes)
            logger.info(f"Created temporary PDF file: {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Failed to create temporary PDF: {str(e)}")
            raise Exception(f"Failed to create temporary PDF: {str(e)}")
    
    def _sanitize_document_name(self, image):
        """
        Sanitize document name to meet Bedrock requirements:
        - Only alphanumeric characters, whitespace, hyphens, parentheses, and square brackets
        - No more than one consecutive whitespace character
        """
        import re
        import os
        
        # Get original filename
        original_name = None
        if hasattr(image, 'name') and image.name:
            original_name = os.path.basename(image.name)
            logger.info(f"Original filename from image.name: {original_name}")
        elif isinstance(image, str) and image:
            original_name = os.path.basename(image)
            logger.info(f"Original filename from string: {original_name}")
        
        # If no valid filename found, use default
        if not original_name or not original_name.strip():
            logger.info("No valid filename found, using default")
            return "document.pdf"
        
        # Remove file extension for processing
        name_without_ext = os.path.splitext(original_name)[0]
        logger.info(f"Name without extension: '{name_without_ext}'")
        
        # If name without extension is empty, use default
        if not name_without_ext or not name_without_ext.strip():
            logger.info("Name without extension is empty, using default")
            return "document.pdf"
        
        # Replace invalid characters with spaces or hyphens
        # Keep only alphanumeric, whitespace, hyphens, parentheses, and square brackets
        # Convert underscores and dots to hyphens, other invalid chars to spaces
        sanitized = re.sub(r'[_\.]', '-', name_without_ext)  # Convert _ and . to -
        sanitized = re.sub(r'[^a-zA-Z0-9\s\-\(\)\[\]]', ' ', sanitized)  # Convert other invalid chars to space
        
        # Replace multiple consecutive whitespace with single space
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Replace multiple consecutive hyphens with single hyphen
        sanitized = re.sub(r'-+', '-', sanitized)
        
        # Trim whitespace and hyphens from start and end
        sanitized = sanitized.strip(' -')
        
        # If name is empty after sanitization, use default
        if not sanitized:
            logger.info("Name is empty after sanitization, using default")
            sanitized = "document"
        
        # Add .pdf extension back
        final_name = f"{sanitized}.pdf"
        logger.info(f"Final sanitized document name: '{final_name}'")
        return final_name