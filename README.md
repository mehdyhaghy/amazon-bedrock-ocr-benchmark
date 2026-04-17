# Amazon Bedrock OCR Benchmark

A side-by-side benchmarking tool that runs the same document through **Amazon Textract**, **Amazon Bedrock Data Automation (BDA)**, and **8 Amazon Bedrock foundation models** (with every available reasoning-effort variant) — all in parallel — and reports latency, cost, and accuracy vs ground truth.

<img src="asset/sample-ui.png" width="900" alt="UI">

## How it works

Enabling **Use Bedrock** automatically runs **all 8 configured models**. Models that support reasoning get expanded into one standalone call per effort level, each appearing as its own row in the results table.

| Model | Provider | Reasoning mode | Variants |
|---|---|---|---|
| Claude Opus 4.7 | Anthropic | adaptive thinking | off, low, medium, high |
| Claude Sonnet 4.6 | Anthropic | adaptive thinking | off, low, medium |
| Claude Haiku 4.5 | Anthropic | budget_tokens | off, 1024, 4096, 16384 |
| Amazon Nova 2 Lite | Amazon | reasoningConfig | off, low, medium |
| Pixtral Large | Mistral | — | 1 |
| Mistral Large 3 | Mistral | — | 1 |
| Llama 4 Maverick 17B | Meta | — | 1 |
| Llama 4 Scout 17B | Meta | — | 1 |

**Total Bedrock variants per image: 18** (11 thinking configurations + 4 non-thinking models + 3 baselines).

All Bedrock calls go through the **Converse API** uniformly for both images and PDFs. Reasoning parameters are passed via `additionalModelRequestFields`:

- **Claude adaptive** → `{"thinking": {"type": "adaptive"}, "output_config": {"effort": "..."}}`
- **Claude budget** → `{"thinking": {"type": "enabled", "budget_tokens": N}}`
- **Nova** → `{"inferenceConfig": {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "..."}}}`

## UI Layout

- **Comparison Results** grid — one row per engine/variant. Populated in one shot when all variants finish, then sorted by processing time ascending. Click any row to:
  - Load its raw JSON in the **Response** tab
  - Load its field-by-field diff against ground truth in the **Compare** tab (heading shows the selected engine name)
- **Truth** tab — shows the loaded ground-truth JSON
- **Response** tab — raw JSON output + API cost (updates from row clicks)
- **Compare** tab — field-by-field diff (updates from row clicks, no dropdown)

The global status bar shows live progress: `N/M engines completed in X seconds (est. cost: $Y)`, switching to a dismissable green "All processing completed" banner when all variants finish. Columns: **Engine**, **Tokens (in/out)**, **Avg. Processing Time (s)**, **Avg. Cost ($)**, **Total Cost ($)**, **Accuracy (%)**.

## Robust JSON parsing

Models sometimes emit slightly-invalid JSON. The parser tries the following in order:
1. Direct `json.loads`
2. Slice from first `{` to last `}` with smart-char fixes (BOMs, Chinese colons `：`, smart quotes)
3. Remove trailing commas (`,}` → `}`)
4. Escape stray control chars (literal `\n`, `\t`, `\r` inside string values)

If the model returns a JSON schema shape (`{"type":"object","properties":{...}}`) instead of filled values, the engine unwraps `properties` automatically. If fields come wrapped as `{"type":"string","value":"..."}` (common with Llama 4), those are unwrapped too. Thinking / `reasoningContent` blocks in the response are stripped before JSON parsing.

## Sample data

12 document samples ship in `sample/`:

| Sample | Schema | Truth |
|---|---|---|
| cvs_reciept.jpg | ❌ | ❌ |
| driver_license.png | ✅ | ✅ |
| graphic.jpg | ✅ | ✅ |
| handwriting.jpg | ✅ | ✅ |
| handwriting2.jpg | ✅ | ✅ |
| insurance_card.png | ✅ | ✅ |
| insurance_claim.png | ✅ | ✅ |
| nutrition.jpg | ✅ | ✅ |
| provider_notes.png | ✅ | ✅ |
| reverse_sheet.png | ✅ | ✅ |
| schedule_table.png | ✅ | ✅ |
| sheet.jpg | ✅ | ✅ |

If an image has no matching schema file, a generic `{"type": "object"}` template is used instead. Accuracy scoring uses a recursive field-level comparison against the truth JSON.

### Add your own samples

1. Drop images in `sample/images/` — the dropdown auto-refreshes on focus
2. (Optional) Add `sample/schema/<basename>.json` for stronger schema guidance
3. (Optional) Add `sample/truth/<basename>.json` for accuracy scoring

## Requirements

- Python 3.10+
- AWS credentials configured locally
- Bedrock model access granted (via the Bedrock console) for all 8 models above in your region
- S3 bucket (optional — only required for BDA processing, and for Textract on PDFs)

## Installation

```bash
git clone https://github.com/mehdyhaghy/amazon-bedrock-ocr-benchmark.git
cd amazon-bedrock-ocr-benchmark

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv
uv pip install -r requirements.txt
```

### Pinned dependencies (April 2026)

- `gradio==6.12.0`
- `boto3==1.42.90`
- `Pillow==12.2.0`
- `numpy==2.4.4`
- `pandas==3.0.2`
- `pymupdf==1.27.2.2`

## Configuration

Update the default S3 bucket name in `ui.py` (or type your own in the UI at runtime):
- **S3 Bucket for Processing** — default `ocr-demo-403202188152` (used by Textract on PDFs and by BDA)

The `bedrock-runtime` client is configured with `read_timeout=120` (2 minutes) and `max_attempts=0` (no retries) so slow/stuck models fail fast and are surfaced as errors in the results grid instead of blocking the whole run.

## Usage

```bash
python app.py
```

Open http://localhost:7860. By default **Use Bedrock** and **Use BDA** are checked. Select a sample or upload your own image/PDF and click **🚀 Process File**.

## Project structure

```
.
├── app.py                      # Gradio entry point + row-click wiring
├── ui.py                       # UI panels (single-tab Response, Truth, Compare)
├── event_handler.py            # Gradio event wiring
├── processor.py                # Parallel orchestration — expands all Bedrock variants
├── sample_handler.py           # Sample loading + generic-schema fallback
├── preview_handler.py          # Image/PDF preview
├── engines/
│   ├── textract_engine.py
│   ├── bedrock_engine.py       # Converse API for all models + JSON repair pipeline
│   └── bda_engine.py
├── shared/
│   ├── config.py               # BEDROCK_MODELS, EFFORT_LEVELS, API_COSTS
│   ├── aws_client.py           # bedrock-runtime client (2-min read_timeout, no retries)
│   ├── prompt_manager.py       # Strict JSON-only prompt instructions
│   ├── evaluator.py            # Recursive field-level accuracy
│   └── comparison_utils.py     # Diff-view HTML renderer
└── sample/
    ├── images/
    ├── schema/
    └── truth/
```

## Credits

This project is based on the [aws-samples/ocr-with-aws-ai-services](https://github.com/aws-samples/ocr-with-aws-ai-services) repository by AWS. This fork adds benchmark mode across 8 Bedrock foundation models with reasoning effort variants, unified Converse API for images and PDFs, an interactive Gradio 6 UI with row-click drill-down, and a robust multi-stage JSON repair pipeline for model outputs.

## License

MIT — see [LICENSE](LICENSE).
