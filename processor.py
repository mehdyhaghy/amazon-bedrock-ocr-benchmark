import concurrent.futures
import time
import pandas as pd
import os

# Import from reorganized modules
from shared.config import logger, BEDROCK_MODELS, CEREBRAS_MODELS, STATUS_HTML, POSTPROCESSING_MODEL, EFFORT_LEVELS, DEFAULT_S3_BUCKET
from engines.textract_engine import TextractEngine
from engines.bedrock_engine import BedrockEngine
from engines.bda_engine import BDAEngine
from engines.cerebras_engine import CerebrasEngine
from shared.cost_calculator import calculate_bedrock_cost, calculate_bda_cost, calculate_full_textract_cost
from shared.evaluator import load_truth_data, calculate_accuracy, get_detailed_accuracy
from shared.comparison_utils import create_diff_view

def initialize_processing(image, image_name=None):
    """Initialize data for image processing"""
    # Determine image name - use provided name or extract from image if available
    if image_name is None:
        if hasattr(image, 'name'):
            image_name = image.name
        else:
            logger.warning("No image name provided and image has no name attribute")
            image_name = "unknown_image"

    logger.info(f"Processing image: {image_name}")
    
    # Load truth data using the determined image name
    truth_data, truth_exists = load_truth_data(image_name)
    
    if truth_exists:
        logger.debug(f"Truth data loaded successfully. Keys: {list(truth_data.keys())}")
        logger.debug(f"Truth data first level values: {str(truth_data)[:200]}...")
        logger.info(f"Loaded truth data for {image_name}")
        truth_status_html = f"""<div style='padding: 10px; background-color: #2e7d32; color: white; 
                                border-radius: 5px; font-weight: bold;'>Ground truth data available for {image_name}</div>"""
    else:
        logger.info(f"No truth data found for {image_name}")
        truth_status_html = f"""<div style='padding: 10px; background-color: #ed6c02; color: white; 
                                border-radius: 5px; font-weight: bold;'>No ground truth data available for {image_name}</div>"""
    
    return image_name, truth_data, truth_exists, truth_status_html

def process_engine_result(engine_name, result, truth_data, truth_exists):
    """Process result from an OCR engine"""
    # Default values
    text = ""
    json_data = None
    image_data = None
    process_time = 0
    status_html = ""
    accuracy = 0.0
    token_usage = None
    cost = 0.0
    cost_html = "<div></div>"
    
    if not isinstance(result, dict):
        return {
            "text": str(result), 
            "json": None, 
            "image": None, 
            "time": 0, 
            "min_time": 0,
            "max_time": 0,
            "runs": 0,
            "status_html": STATUS_HTML["error"](engine_name, 0, "Invalid result format"),
            "accuracy": 0.0,
            "token_usage": None,
            "cost": 0.0,
            "cost_html": "<div></div>"
        }
    
    # Extract common fields from result
    text = result.get('text', '')
    json_data = result.get('json', {})
    image_data = result.get('image')
    # Aggregate timing across repeated calls when available.
    time_samples = result.get('time_samples')
    if time_samples:
        avg_time = sum(time_samples) / len(time_samples)
        min_time = min(time_samples)
        max_time = max(time_samples)
    else:
        avg_time = result.get('process_time', 0)
        min_time = avg_time
        max_time = avg_time
    process_time = avg_time  # status display uses the average
    call_count = len(time_samples) if time_samples else 1
    token_usage = result.get('token_usage')
    
    # Process engine-specific fields
    if engine_name == "Textract":
        pages = result.get('pages', 1)
        textract_base_cost = result.get('textract_cost', 1.50/1000 * pages)
        
        if token_usage:
            haiku_cost_html, haiku_cost = calculate_bedrock_cost(POSTPROCESSING_MODEL, token_usage)
            cost = textract_base_cost + haiku_cost
            cost_detail = f" (Textract: ${textract_base_cost:.6f}, JSON: ${haiku_cost:.6f})"
            status_html = STATUS_HTML["completed"]("Textract", process_time, cost, cost_detail)
        else:
            cost = textract_base_cost
            status_html = STATUS_HTML["completed"]("Textract", process_time, cost)
    
    elif engine_name == "BDA":
        field_count = result.get('field_count', 0)
        use_blueprint = result.get('use_blueprint', False)
        
        # Calculate base BDA cost
        cost_html, bda_base_cost = calculate_bda_cost(use_blueprint, 'document', page_count=1, field_count=field_count)
        cost = bda_base_cost
        
        # Add LLM processing cost if applicable
        if token_usage and not use_blueprint:
            llm_cost_html, llm_cost = calculate_bedrock_cost(POSTPROCESSING_MODEL, token_usage)
            cost += llm_cost
            cost_detail = f" (BDA: ${bda_base_cost:.6f}, JSON: ${llm_cost:.6f})"
            status_html = STATUS_HTML["completed"]("BDA", process_time, cost, cost_detail)
        else:
            status_html = STATUS_HTML["completed"]("BDA", process_time, cost)
    
    else:
        # Any other engine name is treated as a Bedrock model variant
        model_id = result.get('model_id', '')
        if token_usage and model_id:
            cost_html, cost = calculate_bedrock_cost(model_id, token_usage)
        status_html = STATUS_HTML["completed"](engine_name, process_time, cost)
    
    # Calculate accuracy if truth data is available
    if truth_exists and json_data:
        logger.debug(f"Calculating accuracy for {engine_name}")
        accuracy_result = calculate_accuracy(json_data, truth_data)
        accuracy = accuracy_result["total_accuracy"] if isinstance(accuracy_result, dict) else accuracy_result
        logger.info(f"{engine_name} accuracy: {accuracy}%")
        
        # Add accuracy to status
        status_html = status_html.replace("</div>", f" | Accuracy: {accuracy}%</div>")
    
    return {
        "text": text,
        "json": json_data,
        "image": image_data,
        "time": process_time,
        "min_time": min_time,
        "max_time": max_time,
        "runs": call_count,
        "status_html": status_html,
        "accuracy": accuracy,
        "token_usage": token_usage,
        "cost": cost,
        "cost_html": cost_html
    }

def create_comparison_view_for_engines(truth_data, truth_exists, engine_results):
    """Create comparison view based on available engine results"""
    if not truth_exists:
        return "<div>No ground truth data available for comparison</div>"
    
    # Check for any available processed results
    available_engines = []
    for engine_name in list(engine_results.keys()):
        if engine_results[engine_name].get("json"):
            available_engines.append(engine_name)
    
    if not available_engines:
        return "<div>No engine results available for comparison</div>"
    
    # Default to first Bedrock variant if available
    preferred_engine = next((e for e in available_engines if e not in ("Textract", "BDA")), available_engines[0])
    engine_json = engine_results[preferred_engine]["json"]
    
    detailed_results = get_detailed_accuracy(engine_json, truth_data)
    return create_diff_view(truth_data, engine_json)

def create_results_dataframe(engine_results):
    """Create results DataFrame from engine results"""
    final_results = []
    
    for engine_name, data in engine_results.items():
        token_usage = data.get("token_usage") or {}
        in_tok = token_usage.get("inputTokens", 0)
        out_tok = token_usage.get("outputTokens", 0)
        avg_t = data.get("time", 0)
        min_t = data.get("min_time", avg_t)
        max_t = data.get("max_time", avg_t)
        result_row = {
            "Engine": engine_name,
            "Tokens (in/out)": f"{in_tok}/{out_tok}",
            "Avg. Time (s)": f"{avg_t:.3f}",
            "Min Time (s)": f"{min_t:.3f}",
            "Max Time (s)": f"{max_t:.3f}",
            "Avg. Cost ($)": f"${data['cost']:.8f}",
            "Total Cost ($)": f"${data['cost']:.8f}",
            "Accuracy (%)": data["accuracy"]
        }
        final_results.append(result_row)
    
    return pd.DataFrame(final_results)


def build_incremental_results(engine_results, futures, total_start):
    """Build the results DataFrame and row-click json_map from engines that have
    completed so far, for incremental display during streaming.

    Only engines that (a) were actually submitted (present in futures) and
    (b) have produced output (text or json) are included, so partially-finished
    runs show rows as each engine completes — including Cerebras variants.

    Returns (results_df, json_map) where json_map is keyed in the same row order
    as the DataFrame so row-click index lookups resolve correctly.
    """
    completed = {
        name: data for name, data in engine_results.items()
        if name in futures and (data.get("text") or data.get("json"))
    }

    results_df = create_results_dataframe(completed)

    # Sort by processing time ascending so the fastest engines surface first,
    # matching the final table ordering.
    if not results_df.empty and "Avg. Time (s)" in results_df.columns:
        results_df = results_df.sort_values(
            by="Avg. Time (s)",
            key=lambda c: c.astype(float),
            ascending=True
        ).reset_index(drop=True)

    ordered_names = results_df["Engine"].tolist() if not results_df.empty else []
    json_map = {
        name: {
            "json": completed[name].get("json"),
            "text": completed[name].get("text", ""),
            "image": completed[name].get("image"),
            "cost_html": completed[name].get("cost_html", "<div></div>"),
        }
        for name in ordered_names if name in completed
    }
    return results_df, json_map


def run_with_repeats(engine, image, options, call_count):
    """Call engine.process_image up to `call_count` times and aggregate timing.

    Runs the calls sequentially (the outer ThreadPoolExecutor already
    parallelizes across engines/models, so repeats stay in-order per engine).
    Returns the LAST successful result dict augmented with a `time_samples`
    list of per-call process_time values. If a call errors, its result is
    still returned and timing collection stops there.
    """
    try:
        n = int(call_count)
    except (TypeError, ValueError):
        n = 1
    n = max(1, n)

    last_result = None
    samples = []
    for _ in range(n):
        result = engine.process_image(image, options)
        last_result = result
        if isinstance(result, dict):
            samples.append(result.get("process_time", 0) or 0)
            # Stop repeating on error results — no point timing failures.
            if result.get("operation_type") == "error":
                break
        else:
            break

    if isinstance(last_result, dict):
        last_result = {**last_result, "time_samples": samples}
    return last_result


def process_image_with_engines(image, use_textract, use_bedrock, use_bda,
                             bedrock_model_name, bda_s3_bucket="", s3_bucket=DEFAULT_S3_BUCKET,
                             document_type="generic", enable_structured_output=True, output_schema="",
                             use_bda_blueprint=False, image_name=None, use_cerebras=False, call_count=5):
    """Process image with selected OCR engines in parallel"""
    total_start = time.time()
    default_result = {"text": "", "json": None, "image": None, "time": 0, "accuracy": 0, "cost": 0}
    default_bedrock_result = {**default_result, "token_usage": None, "cost_html": "<div></div>"}

    engine_results = {
        "Textract": default_result.copy() if use_textract else default_result.copy(),
        "Bedrock": default_bedrock_result.copy() if use_bedrock else default_bedrock_result.copy(),
        "BDA": default_result.copy() if use_bda else default_result.copy()
    }    

    # Helper to get first Bedrock result for UI tab display
    _non_bedrock = {"Textract", "BDA"}

    def _bedrock_result():
        for k, v in engine_results.items():
            if k not in _non_bedrock:
                if v.get("text") or v.get("json"):
                    return v
        return engine_results.get("Bedrock", default_bedrock_result)

    def _bedrock_status():
        # Show the most recent completed/error Bedrock variant status
        last = "<div></div>"
        for k, v in engine_status.items():
            if k not in _non_bedrock and v != "<div></div>":
                last = v
        return last

    # Empty results for error cases
    empty_df = pd.DataFrame({
        "Engine": [],
        "Tokens (in/out)": [],
        "Avg. Time (s)": [],
        "Min Time (s)": [],
        "Max Time (s)": [],
        "Avg. Cost ($)": [],
        "Total Cost ($)": [],
        "Accuracy (%)" : []
    })
    
    # Check for image and selected engines
    if image is None:
        return [
            STATUS_HTML["error"]("Upload", 0, "No image uploaded"),
            "<div></div>", "", None, None,
            "<div></div>", "", None, None,
            "<div></div>", None,
            "<div></div>", "", None, None,
            "<div></div>", None,
            "<div>No engine selected</div>", "<div>No comparison available</div>",
            empty_df,
            {}
        ]
    
    if not any([use_textract, use_bedrock, use_bda, use_cerebras]):
        return [
            STATUS_HTML["error"]("Selection", 0, "Please select at least one OCR engine"),
            "<div></div>", "", None, None,
            "<div></div>", "", None, None,
            "<div></div>", None,
            "<div></div>", "", None, None,
            "<div></div>", None,
            "<div>No engine selected</div>", "<div>No comparison available</div>",
            empty_df,
            {}
        ]
    
    # Initialize processing data
    image_name, truth_data, truth_exists, truth_status_html = initialize_processing(image, image_name)
    
    # Get bedrock model ID if needed
    model_id = BEDROCK_MODELS.get(bedrock_model_name, "") if use_bedrock else ""
    
    # Initialize engine status and results
    global_status_html = STATUS_HTML["global_processing"]()
    engine_status = {
        "Textract": STATUS_HTML["processing"]("Textract") if use_textract else "<div></div>",
        "Bedrock": STATUS_HTML["processing"]("Bedrock") if use_bedrock else "<div></div>",
        "BDA": STATUS_HTML["processing"]("BDA") if use_bda else "<div></div>"
    }

    # Initial UI update
    yield [
        global_status_html,
        engine_status.get("Textract", "<div></div>"), 
        engine_results.get("Textract", {}).get("text", ""), 
        engine_results.get("Textract", {}).get("json"), 
        engine_results.get("Textract", {}).get("image"),
        
        _bedrock_status(), 
        _bedrock_result().get("text", ""), 
        _bedrock_result().get("json"), 
        _bedrock_result().get("image"),
        
        "<div></div>" if not (use_bedrock or use_cerebras) else _bedrock_result().get("cost_html", "<div></div>"), 
        None if not (use_bedrock or use_cerebras) else _bedrock_result().get("token_usage"),
        
        engine_status.get("BDA", "<div></div>"), 
        engine_results.get("BDA", {}).get("text", ""), 
        engine_results.get("BDA", {}).get("json"), 
        engine_results.get("BDA", {}).get("image"),
        
        truth_status_html, 
        truth_data,
        
        "<div>Processing results... comparison will be available when completed</div>",
        
        empty_df,
        {}
    ]
    
    # Create engine instances
    textract_engine = TextractEngine()
    bedrock_engine = BedrockEngine()
    bda_engine = BDAEngine()
    cerebras_engine = CerebrasEngine()

    # Process with selected engines in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        
        if use_textract:
            futures['Textract'] = executor.submit(
                textract_engine.process_image, 
                image, 
                {
                    'output_schema': output_schema if enable_structured_output else None,
                    's3_bucket': s3_bucket
                }
            )
        
        if use_bedrock:
            # Benchmark mode: run ALL Bedrock models in parallel.
            # Thinking-capable models expand into multiple variants (off + each effort level).
            for display_name, m_id in BEDROCK_MODELS.items():
                effort_config = EFFORT_LEVELS.get(m_id)
                if effort_config:
                    _, levels = effort_config
                    variants = [("off", None)] + [(str(l), l) for l in levels]
                    for label_suffix, level in variants:
                        label = f"{display_name} ({label_suffix})"
                        futures[label] = executor.submit(
                            run_with_repeats,
                            bedrock_engine,
                            image,
                            {
                                'model_id': m_id,
                                'document_type': document_type,
                                'output_schema': output_schema if output_schema else None,
                                'effort_level': level
                            },
                            call_count
                        )
                else:
                    futures[display_name] = executor.submit(
                        run_with_repeats,
                        bedrock_engine,
                        image,
                        {
                            'model_id': m_id,
                            'document_type': document_type,
                            'output_schema': output_schema if output_schema else None
                        },
                        call_count
                    )
        
        if use_cerebras:
            # External Cerebras provider. Gemma 4's reasoning levels are
            # equivalent today, so each model runs as a single variant (reasoning off).
            for display_name, m_id in CEREBRAS_MODELS.items():
                futures[display_name] = executor.submit(
                    run_with_repeats,
                    cerebras_engine,
                    image,
                    {
                        'model_id': m_id,
                        'document_type': document_type,
                        'output_schema': output_schema if output_schema else None
                    },
                    call_count
                )

        if use_bda:
            futures['BDA'] = executor.submit(
                bda_engine.process_image,
                image, 
                {
                    's3_bucket': bda_s3_bucket,
                    'document_type': document_type,
                    'output_schema': output_schema if enable_structured_output and output_schema else None,
                    'use_blueprint': use_bda_blueprint
                }
            )
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures.values()):
            # Find which engine this future belongs to
            engine_name = None
            for name, engine_future in futures.items():
                if future == engine_future:
                    engine_name = name
                    break
                    
            try:
                # Process result for this engine
                result = future.result()
                processed_result = process_engine_result(engine_name, result, truth_data, truth_exists)
                
                # Update engine results
                engine_results[engine_name] = processed_result
                engine_status[engine_name] = processed_result["status_html"]
                
                # Create comparison view with available results
                comparison_html = create_comparison_view_for_engines(truth_data, truth_exists, engine_results)
                
                # Populate the grid incrementally so each engine (including Cerebras)
                # appears as soon as it completes.
                results_df, json_map = build_incremental_results(engine_results, futures, total_start)
                
                # Calculate global status — show partial progress while runs are in flight
                total_time = time.time() - total_start
                total_cost = sum(data["cost"] for name, data in engine_results.items() if name in futures.keys())
                completed_count = sum(1 for f in futures.values() if f.done())
                total_count = len(futures)
                if completed_count < total_count:
                    global_status_html = STATUS_HTML["global_partial"](completed_count, total_count, total_time, total_cost)
                else:
                    global_status_html = STATUS_HTML["global_completed"](total_time, total_cost)
                
                # Update UI
                yield [
                    global_status_html,
                    engine_status.get("Textract", "<div></div>"), 
                    engine_results.get("Textract", {}).get("text", ""), 
                    engine_results.get("Textract", {}).get("json"), 
                    engine_results.get("Textract", {}).get("image"),
                    
                    _bedrock_status(), 
                    _bedrock_result().get("text", ""), 
                    _bedrock_result().get("json"), 
                    _bedrock_result().get("image"),
                    
                    "<div></div>" if not (use_bedrock or use_cerebras) else _bedrock_result().get("cost_html", "<div></div>"), 
                    None if not (use_bedrock or use_cerebras) else _bedrock_result().get("token_usage"),
                    
                    engine_status.get("BDA", "<div></div>"), 
                    engine_results.get("BDA", {}).get("text", ""), 
                    engine_results.get("BDA", {}).get("json"), 
                    engine_results.get("BDA", {}).get("image"),
                    
                    truth_status_html, 
                    truth_data,
                    
                    comparison_html,
                    
                    results_df,
                    json_map
                ]
                
            except Exception as e:
                logger.error(f"Error in {engine_name} processing: {str(e)}")
                
                # Update error status for this engine
                engine_status[engine_name] = STATUS_HTML["error"](engine_name, 0, str(e))
                
                # Create comparison with available results
                comparison_html = create_comparison_view_for_engines(truth_data, truth_exists, engine_results)
                
                # Populate the grid incrementally with whatever has completed so far.
                results_df, json_map = build_incremental_results(engine_results, futures, total_start)
                
                # Calculate global status
                total_time = time.time() - total_start
                total_cost = sum(data.get("cost", 0) for name, data in engine_results.items())
                completed_count = sum(1 for f in futures.values() if f.done())
                total_count = len(futures)
                if completed_count < total_count:
                    global_status_html = STATUS_HTML["global_partial"](completed_count, total_count, total_time, total_cost)
                else:
                    global_status_html = STATUS_HTML["global_completed"](total_time, total_cost)
                
                # Update UI
                yield [
                    global_status_html,
                    engine_status.get("Textract", "<div></div>"), 
                    engine_results.get("Textract", {}).get("text", ""), 
                    engine_results.get("Textract", {}).get("json"), 
                    engine_results.get("Textract", {}).get("image"),
                    
                    _bedrock_status(), 
                    _bedrock_result().get("text", ""), 
                    _bedrock_result().get("json"), 
                    _bedrock_result().get("image"),
                    
                    "<div></div>" if not (use_bedrock or use_cerebras) else _bedrock_result().get("cost_html", "<div></div>"), 
                    None if not (use_bedrock or use_cerebras) else _bedrock_result().get("token_usage"),
                    
                    engine_status.get("BDA", "<div></div>"), 
                    engine_results.get("BDA", {}).get("text", ""), 
                    engine_results.get("BDA", {}).get("json"), 
                    engine_results.get("BDA", {}).get("image"),
                    
                    truth_status_html, 
                    truth_data,
                    
                    comparison_html,
                    
                    results_df,
                    json_map
                ]
    
    # Final comparison view
    comparison_html = create_comparison_view_for_engines(truth_data, truth_exists, engine_results)
    
    # Final results
    results_df = create_results_dataframe({
        name: data for name, data in engine_results.items() 
        if name in futures.keys()  # Only include selected engines
    })
    
    # Sort by processing time ascending (final results only)
    if not results_df.empty and "Avg. Time (s)" in results_df.columns:
        results_df = results_df.sort_values(
            by="Avg. Time (s)",
            key=lambda c: c.astype(float),
            ascending=True
        ).reset_index(drop=True)
    
    # Build final JSON map in the same order as the sorted DataFrame so that
    # row-click index lookups resolve to the correct engine variant.
    ordered_names = results_df["Engine"].tolist() if not results_df.empty else []
    final_json_map = {
        name: {
            "json": engine_results[name].get("json"),
            "text": engine_results[name].get("text", ""),
            "image": engine_results[name].get("image"),
            "cost_html": engine_results[name].get("cost_html", "<div></div>"),
        }
        for name in ordered_names if name in engine_results
    }
    
    # Final status
    total_time = time.time() - total_start
    total_cost = sum(data["cost"] for name, data in engine_results.items() if name in futures.keys())
    global_status_html = STATUS_HTML["global_completed"](total_time, total_cost)
    
    # Return final UI update
    final_output = [
        global_status_html,
        engine_status.get("Textract", "<div></div>"), 
        engine_results.get("Textract", {}).get("text", ""), 
        engine_results.get("Textract", {}).get("json"), 
        engine_results.get("Textract", {}).get("image"),
        
        _bedrock_status(), 
        _bedrock_result().get("text", ""), 
        _bedrock_result().get("json"), 
        _bedrock_result().get("image"),
        
        "<div></div>" if not (use_bedrock or use_cerebras) else _bedrock_result().get("cost_html", "<div></div>"), 
        None if not (use_bedrock or use_cerebras) else _bedrock_result().get("token_usage"),
        
        engine_status.get("BDA", "<div></div>"), 
        engine_results.get("BDA", {}).get("text", ""), 
        engine_results.get("BDA", {}).get("json"), 
        engine_results.get("BDA", {}).get("image"),
        
        truth_status_html, 
        truth_data,
        
        comparison_html,
        
        results_df,
        final_json_map
    ]
    yield final_output
