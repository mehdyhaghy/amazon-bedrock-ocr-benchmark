import os
import gradio as gr
import datetime
from shared.config import CUSTOM_THEME, logger
from ui import create_input_panel, create_common_options_panel, create_results_panel, create_results_table
from event_handler import setup_event_handlers

def create_ocr_app():
    """Create the OCR application with all components"""
    with gr.Blocks() as app:
        gr.Markdown("# 📝 Multi-Engine OCR Application\n\nUpload an image containing text and select your preferred processing engines.")
        
        # Current timestamp display
        timestamp_html = gr.HTML(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label="Current Time",
            every=1
        )
        
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
                    use_bda = gr.Checkbox(value=False, label="Use BDA")
                
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
                
                # Wire row-click to show the selected engine's raw JSON in the Bedrock tab
                # and hide the extracted text box (JSON has all the data)
                def _on_row_select(results_map, evt: gr.SelectData):
                    if not results_map or evt is None or evt.index is None:
                        return None, gr.update()
                    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                    keys = list(results_map.keys())
                    if 0 <= row_idx < len(keys):
                        return results_map[keys[row_idx]], gr.update(visible=False)
                    return None, gr.update()
                
                results_table.select(
                    fn=_on_row_select,
                    inputs=[results_json_state],
                    outputs=[input_components["bedrock_json"], input_components["bedrock_text"]]
                )
        
        # Insert global status at the beginning of output components
        output_components.insert(0, global_status)
        
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
