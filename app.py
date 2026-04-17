import os
import gradio as gr
from shared.config import CUSTOM_THEME, logger
from ui import create_input_panel, create_common_options_panel, create_results_panel, create_results_table
from event_handler import setup_event_handlers

def create_ocr_app():
    """Create the OCR application with all components"""
    with gr.Blocks(title="Amazon Bedrock OCR Benchmark") as app:
        gr.Markdown("# 📝 Amazon Bedrock OCR Benchmark")
        
        # Current timestamp display
        with gr.Row():
            # Left column for inputs
            with gr.Column(scale=1):
                # Create input panel
                (input_panel, sample_dropdown, input_image, refresh_samples, image_preview, pdf_preview,
                 pdf_controls, prev_page_btn, page_info, next_page_btn, current_page, total_pages, current_pdf_path,
                 process_file_button, process_all_samples_button) = create_input_panel()
                
                # Engine selection
                with gr.Row():
                    use_textract = gr.Checkbox(value=True, label="Use Textract")
                    use_bedrock = gr.Checkbox(value=True, label="Use Bedrock")
                    use_bda = gr.Checkbox(value=True, label="Use BDA")
                
                # Create common options panel
                common_options, s3_bucket, document_type, enable_structured_output, output_schema, bedrock_model, bda_s3_bucket, use_bda_blueprint = create_common_options_panel()
                
                # These buttons are now in the input panel under preview
            
            # Right column for results
            with gr.Column(scale=2):
                # Global status for all processing
                global_status = gr.HTML("<div class='status-ready'>Ready for processing</div>", label="Status")
                results_table, results_json_state = create_results_table()
                
                # Results panel with tabs for each engine
                results_panel, input_components, output_components = create_results_panel()
                
                # Wire row-click to show the selected engine's response in the Response tab
                def _on_row_select(results_map, truth, evt: gr.SelectData):
                    empty = (None, gr.update(visible=False, value=""),
                             gr.update(visible=False, value=None), "<div></div>",
                             "<div>Click a row in the Comparison Results table to see its diff against ground truth</div>")
                    print(f"[row_select] results_map keys={list(results_map.keys()) if results_map else None}, truth type={type(truth).__name__}, truth keys={list(truth.keys())[:3] if isinstance(truth, dict) else None}, evt.index={evt.index if evt else None}")
                    if not results_map or evt is None or evt.index is None:
                        return empty
                    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                    keys = list(results_map.keys())
                    if not (0 <= row_idx < len(keys)):
                        return empty
                    entry = results_map[keys[row_idx]] or {}
                    if isinstance(entry, dict) and "json" in entry:
                        j = entry.get("json")
                        text = entry.get("text") or ""
                        img = entry.get("image")
                        cost_html = entry.get("cost_html") or "<div></div>"
                    else:
                        j = entry
                        text = ""
                        img = None
                        cost_html = "<div></div>"
                    # Build diff view if truth is available
                    from shared.comparison_utils import create_diff_view
                    if truth and j:
                        diff_html = create_diff_view(truth, j, engine_name=keys[row_idx])
                    else:
                        diff_html = "<div>No ground truth available for comparison</div>"
                    return (
                        j,
                        gr.update(value=text, visible=bool(text)),
                        gr.update(value=img, visible=img is not None),
                        cost_html,
                        diff_html,
                    )
                
                results_table.select(
                    fn=_on_row_select,
                    inputs=[results_json_state, input_components["truth_json"]],
                    outputs=[
                        input_components["response_json"],
                        input_components["response_text"],
                        input_components["response_image"],
                        input_components["response_cost"],
                        input_components["comparison_view"],
                    ]
                )
        
        # Insert global status at the beginning of output components
        output_components.insert(0, global_status)
        input_components["global_status"] = global_status
        
        # Setup event handlers
        setup_event_handlers(
            use_textract, use_bedrock, use_bda,
            sample_dropdown, input_image, s3_bucket, enable_structured_output, output_schema,
            refresh_samples, process_file_button, process_all_samples_button,
            bedrock_model, document_type, bda_s3_bucket,
            input_components, output_components, use_bda_blueprint,
            results_table, image_preview, pdf_preview, pdf_controls,
            prev_page_btn, page_info, next_page_btn, current_page, total_pages, current_pdf_path,
            results_json_state=results_json_state
        )
    
    return app


if __name__ == "__main__":
    demo = create_ocr_app()
    demo.launch(share=False, theme=CUSTOM_THEME)
