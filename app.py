import os
import gradio as gr
from shared.config import CUSTOM_THEME, logger
from ui import create_input_panel, create_common_options_panel, create_results_panel, create_results_table
from event_handler import setup_event_handlers

def create_ocr_app():
    """Create the OCR application with all components"""
    _RESIZE_CSS = """
    /* Two-pane layout with a draggable splitter between left (inputs) and
       right (results). The left pane uses an explicit flex-basis that the JS
       splitter updates on drag; the right pane flexes to fill the rest. */
    #main-split { display: flex; flex-wrap: nowrap; align-items: stretch; gap: 0; }
    #left-pane {
        flex-grow: 0 !important;
        flex-shrink: 0 !important;
        flex-basis: var(--left-pane-width, 420px) !important;
        width: var(--left-pane-width, 420px) !important;
        min-width: 260px !important;
        max-width: 80% !important;
        overflow: auto;
    }
    /* Gradio wraps the HTML component in this block — make IT the narrow,
       non-growing flex child and the actual drag target. */
    #split-handle-wrap {
        flex: 0 0 12px !important;
        min-width: 12px !important;
        padding: 0 !important;
        margin: 0 6px !important;
        cursor: col-resize;
        align-self: stretch;
        display: flex;
    }
    #split-handle {
        flex: 1 1 auto;
        width: 100%;
        cursor: col-resize;
        background: var(--border-color-primary, #888);
        opacity: 0.4;
        border-radius: 4px;
        transition: opacity 0.15s ease;
    }
    #split-handle-wrap:hover #split-handle,
    #split-handle.dragging { opacity: 0.85; }
    #right-pane {
        flex: 1 1 0% !important;
        min-width: 0 !important;        /* allow the pane (and table) to shrink properly */
        overflow: hidden;
    }
    /* Let the results table fill its pane and only scroll when truly needed */
    #results-dataframe { width: 100%; }
    /* Gradio 6.x renders the Dataframe with a virtual scroller that divides the
       viewport height by the measured row height to decide how many rows to
       mount. Empty/blank rows can measure as 0px tall, which collapses the
       calculation so only ONE row mounts ("shows 1 row while N processed").
       Force a minimum row/cell height so the virtualizer measures real heights
       and mounts every visible row. */
    #results-dataframe .virtual-body .tr,
    #results-dataframe table tr { min-height: 28px !important; }
    #results-dataframe table td,
    #results-dataframe table th { min-height: 28px !important; }
    /* Allow the Tokens header ("Tokens\n(in/out)") to render on two lines and
       keep all headers compact/centered so the slim time columns aren't clipped. */
    #results-dataframe table th {
        white-space: pre-line !important;
        line-height: 1.15 !important;
        vertical-align: middle;
        padding-top: 4px !important;
        padding-bottom: 4px !important;
    }

    /* Use the full browser width. Gradio's default container (and the
       .fillable svelte wrapper) caps content at ~1280px; override so the
       two-pane layout spans the whole page. */
    .gradio-container { max-width: 100% !important; width: 100% !important; }
    .gradio-container .fillable,
    .gradio-container .main,
    .gradio-container > .wrap { max-width: 100% !important; }
    """

    _SPLITTER_JS = """
    () => {
        const root = document.getElementById('main-split');
        const left = document.getElementById('left-pane');
        // The Gradio wrapper is the actual flex child / drag target.
        const handle = document.getElementById('split-handle-wrap');
        if (!root || !left || !handle || handle.dataset.wired) return;
        handle.dataset.wired = '1';
        const inner = document.getElementById('split-handle');
        let dragging = false;
        const setWidth = (w) => {
            left.style.setProperty('flex-basis', w + 'px', 'important');
            left.style.setProperty('width', w + 'px', 'important');
        };
        const onMove = (e) => {
            if (!dragging) return;
            const rect = root.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            let w = clientX - rect.left;
            const min = 260, max = rect.width * 0.8;
            w = Math.max(min, Math.min(max, w));
            setWidth(w);
            e.preventDefault();
        };
        const stop = () => {
            dragging = false;
            if (inner) inner.classList.remove('dragging');
            document.body.style.userSelect = '';
        };
        const start = (e) => {
            dragging = true;
            if (inner) inner.classList.add('dragging');
            document.body.style.userSelect = 'none';
            e.preventDefault();
        };
        handle.addEventListener('mousedown', start);
        handle.addEventListener('touchstart', start, { passive: false });
        window.addEventListener('mousemove', onMove);
        window.addEventListener('touchmove', onMove, { passive: false });
        window.addEventListener('mouseup', stop);
        window.addEventListener('touchend', stop);
    }
    """
    with gr.Blocks(title="Amazon Bedrock OCR Benchmark", fill_width=True,
                   theme=CUSTOM_THEME, css=_RESIZE_CSS) as app:
        gr.Markdown("# 📝 Amazon Bedrock OCR Benchmark")
        
        # Two-pane layout with a draggable splitter (#split-handle) between
        # the left inputs pane and the right results pane.
        with gr.Row(elem_id="main-split"):
            # Left column for inputs (width controlled by the JS splitter)
            with gr.Column(scale=1, elem_id="left-pane"):
                # Create input panel
                (input_panel, sample_dropdown, input_image, refresh_samples, image_preview, pdf_preview,
                 pdf_controls, prev_page_btn, page_info, next_page_btn, current_page, total_pages, current_pdf_path,
                 process_file_button, process_all_samples_button) = create_input_panel()
                
                # Engine selection
                with gr.Row():
                    use_textract = gr.Checkbox(value=True, label="Use Textract")
                    use_bedrock = gr.Checkbox(value=True, label="Use Bedrock")
                    use_bda = gr.Checkbox(value=True, label="Use BDA")
                    use_cerebras = gr.Checkbox(value=True, label="Use Cerebras")
                
                # Create common options panel
                common_options, s3_bucket, document_type, enable_structured_output, output_schema, bedrock_model, bda_s3_bucket, use_bda_blueprint, call_count = create_common_options_panel()
                
                # These buttons are now in the input panel under preview

            # Draggable splitter handle (wired up by _SPLITTER_JS on load)
            gr.HTML("<div id='split-handle' title='Drag to resize'></div>", elem_id="split-handle-wrap")

            # Right column for results
            with gr.Column(scale=2, elem_id="right-pane"):
                # Global status for all processing
                global_status = gr.HTML("<div class='status-ready'>Ready for processing</div>", label="Status")
                results_table, results_json_state = create_results_table()
                
                # Results panel with tabs for each engine
                results_panel, input_components, output_components, results_tabs = create_results_panel()
                
                # Wire row-click to show the selected engine's response and jump
                # to the Compare tab so the diff against ground truth is visible.
                def _on_row_select(results_map, truth, evt: gr.SelectData):
                    empty = (None, gr.update(visible=False, value=""),
                             gr.update(visible=False, value=None), "<div></div>",
                             "<div>Click a row in the Comparison Results table to see its diff against ground truth</div>",
                             gr.update())
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
                        gr.Tabs(selected="compare"),
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
                        results_tabs,
                    ]
                )
        
        # Insert global status at the beginning of output components
        output_components.insert(0, global_status)
        input_components["global_status"] = global_status
        
        # Setup event handlers
        current_sample_name = setup_event_handlers(
            use_textract, use_bedrock, use_bda,
            sample_dropdown, input_image, s3_bucket, enable_structured_output, output_schema,
            refresh_samples, process_file_button, process_all_samples_button,
            bedrock_model, document_type, bda_s3_bucket,
            input_components, output_components, use_bda_blueprint,
            results_table, image_preview, pdf_preview, pdf_controls,
            prev_page_btn, page_info, next_page_btn, current_page, total_pages, current_pdf_path,
            results_json_state=results_json_state,
            use_cerebras=use_cerebras,
            call_count=call_count
        )

        # On initial page load, populate the preview/image/schema/truth for the
        # default-selected sample so it renders without a manual re-selection.
        def _load_default_sample(sample):
            from sample_handler import on_sample_selected, list_sample_images
            from preview_handler import handle_sample_preview
            if not sample:
                samples = list_sample_images()
                sample = samples[0] if samples else None
            if not sample:
                return ("", None, None, None, None, None,
                        "<div style='text-align: center; padding: 50px; color: #666;'>No sample selected</div>")
            image_path, schema, truth_data, truth_status = on_sample_selected(sample)
            preview_result = handle_sample_preview(image_path)
            preview_image, preview_pdf = preview_result[0], preview_result[1]
            return (sample, image_path, schema, truth_data, truth_status,
                    preview_image, preview_pdf)

        app.load(
            fn=_load_default_sample,
            inputs=sample_dropdown,
            outputs=[current_sample_name, input_image, output_schema,
                     input_components["truth_json"], input_components["truth_status"],
                     image_preview, pdf_preview]
        )

        # Wire up the draggable splitter between the two panes on page load.
        app.load(fn=None, js=_SPLITTER_JS)

    # CSS and theme are now set on the Blocks() constructor above so the app
    # renders correctly under Gradio's hot-reload watch mode (`gradio app.py`),
    # which auto-launches the module-level Blocks without calling .launch().
    # Kept here too for any caller that still reads it.
    app._ocr_css = _RESIZE_CSS
    return app


# Module-level Blocks instance. Required for Gradio hot reload: running
# `gradio app.py` watches the file and re-launches this top-level `demo` object
# on every save. `python app.py` (no reload) still works via the __main__ block.
demo = create_ocr_app()


if __name__ == "__main__":
    # Direct run (no hot reload). For hot reload use:  gradio app.py
    demo.launch(share=False)
