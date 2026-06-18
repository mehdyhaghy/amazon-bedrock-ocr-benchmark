import gradio as gr
import pandas as pd
from shared.config import CUSTOM_THEME, BEDROCK_MODELS, STATUS_HTML, DEFAULT_S3_BUCKET
from sample_handler import list_sample_images

def create_input_panel():
    """Create the input panel with sample selection and image upload"""
    with gr.Column() as panel:
        with gr.Row():
            sample_dropdown = gr.Dropdown(
                choices=list_sample_images(),
                label="Sample Images",
                info="Select a sample image or upload your own",
                scale=4
            )
            
        with gr.Row():
            # Left column for file upload
            with gr.Column(scale=1):
                input_image = gr.File(
                    file_types=["image", ".pdf"], 
                    label="Input Image or PDF"
                )
            
            # Right column for preview
            with gr.Column(scale=1):
                gr.Markdown("### 👁️ Preview")
                
                # PDF page navigation controls  
                with gr.Column(visible=False) as pdf_controls:
                    page_info = gr.HTML(
                        "<div style='text-align: center; padding: 4px 0; font-weight: 500; color: #666; font-size: 14px;'>Page 1 of 1</div>"
                    )
                    with gr.Row():
                        prev_page_btn = gr.Button("◀ Previous", variant="secondary", size="sm", scale=1)
                        next_page_btn = gr.Button("Next ▶", variant="secondary", size="sm", scale=1)
                
                image_preview = gr.Image(
                    label="Image Preview",
                    show_label=False,
                    height=400,
                    visible=True
                )
                pdf_preview = gr.HTML(
                    label="PDF Preview",
                    value="<div style='text-align: center; padding: 50px; color: #666;'>Upload a PDF to see preview</div>",
                    visible=False
                )
                
                # Process buttons moved here, right under preview
                gr.Markdown("---")
                process_file_button = gr.Button("🚀 Process File", variant="primary")
                process_all_button = gr.Button("", visible=False)  # Removed from UI
                
                # Hidden state for PDF navigation
                current_page = gr.State(0)
                total_pages = gr.State(1)
                current_pdf_path = gr.State(None)
    
    return (panel, sample_dropdown, input_image, None, image_preview, pdf_preview, 
            pdf_controls, prev_page_btn, page_info, next_page_btn, current_page, total_pages, current_pdf_path,
            process_file_button, process_all_button)

def create_results_table():
    """Create a table to display comparative performance metrics with dark mode support"""
    # Pre-fill with 5 blank rows so the grid reserves space visually on load
    results_df = pd.DataFrame({
        "Engine": [""] * 5,
        "Tokens (in/out)": [""] * 5,
        "Avg. Time (s)": [""] * 5,
        "Min Time (s)": [""] * 5,
        "Max Time (s)": [""] * 5,
        "Avg. Cost ($)": [""] * 5,
        "Total Cost ($)": [""] * 5,
        "Accuracy (%)": [""] * 5,
    })
    
    results_table = gr.Dataframe(
        value=results_df,
        label="Comparison Results",
        interactive=False,
        wrap=True,
        row_count=(20, "fixed"),
        max_height=500,
        column_widths=["200px", "120px", "100px", "100px", "100px", "120px", "120px", "100px"],
        elem_id="results-dataframe",
        elem_classes="results-dataframe"
    )
    # State to store per-engine JSON outputs for row-click lookup
    results_json_state = gr.State({})
    
    return results_table, results_json_state

def create_common_options_panel():
    """Create common options panel for all engines"""
    with gr.Column() as panel:
        # Basic Configuration Section
        gr.Markdown("### 🔧 Basic Configuration")
        
        with gr.Row():
            document_type = gr.Dropdown(
                choices=["generic", "form", "receipt", "table", "handwritten"],
                value="generic",
                label="Document Type",
                info="Select the type of document to optimize prompt selection",
                scale=1
            )
            
            enable_structured_output = gr.Checkbox(
                label="Enable Structured Output",
                value=True,
                info="Enable structured JSON output processing (uses additional Bedrock API calls)",
                scale=1
            )

            call_count = gr.Number(
                label="Calls per model",
                value=5,
                minimum=1,
                maximum=20,
                step=1,
                precision=0,
                info="How many times to call each model; processing time is reported as avg/min/max",
                scale=1
            )
        
        # S3 Configuration Section
        gr.Markdown("### 🪣 S3 Configuration")
        
        with gr.Row():
            s3_bucket = gr.Textbox(
                label="S3 Bucket for Processing",
                value=DEFAULT_S3_BUCKET,
                placeholder="Enter your S3 bucket name",
                info="S3 bucket for uploading files for processing (required for all engines, including BDA)",
                scale=2
            )

        bda_s3_bucket = s3_bucket  # same bucket used for BDA
        
        # Bedrock Configuration Section
        gr.Markdown("### 🤖 Bedrock Configuration")
        
        with gr.Row():
            use_bda_blueprint = gr.Checkbox(
                label="Use Custom Blueprint (BDA)",
                value=False,
                info="When enabled, creates a custom blueprint based on the output schema. When disabled, uses default extraction with Claude Haiku post-processing.",
                scale=1
            )
        bedrock_model = gr.State("")  # Unused — benchmark mode runs all models
        
        # Output Schema Section
        gr.Markdown("### 📋 Output Schema Configuration")
        gr.Markdown("*Define the JSON schema for structured output*")
        output_schema = gr.Code(
            language="json",
            label="Output Schema",
            value="{\n  \"type\": \"object\"\n}"
        )
    
    return panel, s3_bucket, document_type, enable_structured_output, output_schema, bedrock_model, bda_s3_bucket, use_bda_blueprint, call_count


def create_results_panel():
    """Create the results panel with tabs for each engine"""
    with gr.Column() as panel:
        with gr.Tabs() as tabs:
            with gr.TabItem("Response"):
                gr.Markdown("Click a row in the **Comparison Results** table above to view its response.")
                response_json = gr.JSON(label="Raw JSON Output")
                response_text = gr.Textbox(visible=False)
                response_image = gr.Image(visible=False)
                response_cost = gr.HTML("<div></div>")
            
            with gr.TabItem("Truth"):
                truth_status = gr.HTML("<div></div>", label="Status", visible=False)
                truth_json = gr.JSON(label="Ground Truth Data")
            
            with gr.TabItem("Compare"):
                comparison_view = gr.HTML("<div>Click a row in the Comparison Results table to see its diff against ground truth</div>")
    
    # Hidden placeholders for backward-compatible output_components ordering.
    # Per-engine streaming writes go here; the visible Response tab is populated
    # separately from the row-click handler in app.py.
    textract_status = gr.HTML(visible=False)
    textract_extracted_text = gr.Textbox(visible=False)
    textract_json = gr.JSON(visible=False)
    textract_image = gr.Image(visible=False)
    bedrock_status = gr.HTML(visible=False)
    bedrock_extracted_text = gr.Textbox(visible=False)
    bedrock_json = gr.JSON(visible=False)
    bedrock_image = gr.Image(visible=False)
    bedrock_cost = gr.HTML(visible=False)
    bedrock_token_usage = gr.JSON(visible=False)
    bda_status = gr.HTML(visible=False)
    bda_extracted_text = gr.Textbox(visible=False)
    bda_json = gr.JSON(visible=False)
    bda_image = gr.Image(visible=False)
    
    # Organize components for easier access
    input_components = {
        "textract_status": textract_status,
        "textract_text": textract_extracted_text,
        "textract_json": textract_json,
        "textract_image": textract_image,
        "bedrock_status": bedrock_status,
        "bedrock_text": bedrock_extracted_text,
        "bedrock_json": bedrock_json,
        "bedrock_image": bedrock_image,
        "bedrock_cost": bedrock_cost,
        "bedrock_token_usage": bedrock_token_usage,
        "bda_status": bda_status,
        "bda_text": bda_extracted_text,
        "bda_json": bda_json,
        "bda_image": bda_image,
        "truth_status": truth_status,
        "truth_json": truth_json,
        "comparison_view": comparison_view,
        "response_json": response_json,
        "response_text": response_text,
        "response_image": response_image,
        "response_cost": response_cost
    }
    
    output_components = [
        textract_status, textract_extracted_text, textract_json, textract_image,
        bedrock_status, bedrock_extracted_text, bedrock_json, bedrock_image,
        bedrock_cost, bedrock_token_usage,
        bda_status, bda_extracted_text, bda_json, bda_image,
        truth_status, truth_json,
        comparison_view
    ]
    
    return panel, input_components, output_components
