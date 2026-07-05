# Mini-Conductor Colab Run Guide

Generated artifacts:

- `Mini_Conductor_OpenRouter_Unsloth.ipynb` — Colab-ready notebook file.
- `generate_colab_notebook.py` — generator script for the notebook.
- `DISCOVERY.md` — research/discovery notes.
- `PLAN.md` — implementation plan and cost-aware OpenRouter routing plan.

## How to use in Google Colab Pro+

1. Open Google Colab.
2. Upload `Mini_Conductor_OpenRouter_Unsloth.ipynb`.
3. Runtime → Change runtime type → GPU.
4. Prefer A100/H100/RTX 6000 Ada for the 8B run.
5. Run cells top-to-bottom.

The notebook defaults are conservative:

- `RUN_MODEL_LOAD = True`
- `RUN_SFT = True`
- `RUN_GRPO = False`
- `RUN_OPENROUTER_SMOKE = False`
- `RUN_OPENROUTER_DEMO = False`

So the default run loads the coordinator, builds SFT data, runs a short SFT smoke run, validates local JSON planning, and does not spend OpenRouter credits.

## When to enable OpenRouter

Only after SFT output validates should you enable API calls:

```python
RUN_OPENROUTER_SMOKE = True
```

Then later:

```python
RUN_OPENROUTER_DEMO = True
```

Only after Cell 8 reports `3/3` valid local plans should you enable GRPO:

```python
RUN_GRPO = True
OPENROUTER_BUDGET_USD = 5.00
REQUIRE_GRPO_READY = True
```

If you intentionally want GRPO to bootstrap from invalid plans, set `REQUIRE_GRPO_READY = False`; invalid plans receive negative format/correctness rewards and do not trigger worker execution until they parse successfully.

## Producing Harness Artifacts

The notebook saves LoRA adapters to a run-family-specific Google Drive folder, and each run gets its own timestamped subfolder so a rerun never overwrites a previous run's data:

```python
PROJECT_SLUG = "mini-conductor-qwen3-router-sft"   # stable run family
RUN_TAG = os.environ.get("RUN_TAG") or datetime.now().strftime("run-%Y%m%d-%H%M%S")
ROOT = Path('/content/drive/MyDrive') / PROJECT_SLUG / RUN_TAG
```

That resolves to (with an example tag):

- SFT adapter: `/content/drive/MyDrive/mini-conductor-qwen3-router-sft/run-20260629-1200/checkpoints/sft_lora`
- GRPO adapter: `/content/drive/MyDrive/mini-conductor-qwen3-router-sft/run-20260629-1200/checkpoints/grpo_lora`

Cell 10 exports harness-facing metadata to:

```text
/content/drive/MyDrive/mini-conductor-qwen3-router-sft/run-20260629-1200/harness_artifacts/
```

Because `RUN_TAG` defaults to a timestamp, every fresh run lands in a new folder automatically. To **resume or overwrite** one specific run, pin the tag to a fixed value (or set the `RUN_TAG` environment variable), e.g. `RUN_TAG = "sft-v1"`.

That directory contains:

- `manifest.json` — adapter path, base model, worker list, prompts, and schema metadata.
- `latest_adapter.txt` — path to the preferred adapter, choosing GRPO if present, otherwise SFT.
- `conductor_harness_loader.py` — minimal Colab-side loader helper.

Cell 10 can optionally upload the adapter and harness metadata to Hugging Face:

```python
PUSH_TO_HF = True
PROJECT_SLUG = "mini-conductor-qwen3-router-sft"
HF_REPO_ID = f"bebrws/{PROJECT_SLUG}"
HF_PRIVATE_REPO = True
```

Uploads are namespaced per run under `runs/<RUN_TAG>/` inside the repo (`runs/<RUN_TAG>/adapter`, `runs/<RUN_TAG>/harness_artifacts`, and `runs/<RUN_TAG>/merged_16bit` when exported), so reruns do not overwrite earlier uploads in the same repo. `HF_TOKEN` must be present in the Colab environment. The default is `PUSH_TO_HF = False` so export does not fail in runtimes without a token.

## Repo Harness

The local harness CLI is:

```bash
python3 -m mini_conductor.harness \
  --harness-dir /path/to/harness_artifacts \
  --question "Write a Python function to reverse a string." \
  --conductor-mode local
```

Add `--execute` to call OpenRouter workers. Without `--execute`, it dry-runs the workflow. `OPENROUTER_API_KEY` must be set when using `--execute`.

## Cursor / OpenAI-compatible gateway

For Cursor, use the OpenAI-compatible gateway in `mini_conductor/openai_server.py`. It exposes `/v1/models` and `/v1/chat/completions`, loads the conductor/harness artifacts from Cell 10, calls OpenRouter workers using `OPENROUTER_API_KEY`, and returns a normal Chat Completions response to Cursor.

See `CURSOR_CONDUCTOR.md` for setup details.

## Worker models and prices

The notebook uses these OpenRouter workers:

| Worker | IN $/1M | OUT $/1M |
|---|---:|---:|
| `minimax/minimax-m3` | 0.30 | 1.20 |
| `qwen/qwen3.7-plus` | 0.32 | 1.28 |
| `z-ai/glm-5.2` | 0.95 | 3.00 |
| `moonshotai/kimi-k2.7-code` | 0.74 | 3.50 |

The plan intentionally prefers MiniMax/Qwen for cheap first passes and reserves GLM/Kimi for hard reasoning/coding.

## SFT readiness: emitting valid JSON plans

The coordinator (`unsloth/Qwen3-8B-...`) is a hybrid-reasoning model that emits a `<think>...</think>` monologue by default. With greedy decoding the model would spend its whole token budget on reasoning and never reach the JSON, so Cell 8 reported `0/3 local smoke prompts emitted valid JSON plans`. The notebook fixes this by:

- Disabling thinking for both SFT and inference (`enable_thinking=False`) via a shared `render_conductor_prompt` helper, so the model emits JSON immediately.
- Training each SFT example on exactly the inference prompt plus the target JSON plus EOS (`format_sft_example`), eliminating train/inference skew and teaching the model to stop right after the JSON.
- Parsing JSON robustly: `<think>` blocks (including truncated ones) and surrounding prose are stripped, then the first well-formed JSON object is decoded, so stray braces in reasoning text cannot corrupt extraction.
- Running enough SFT steps (`SFT_MAX_STEPS = 60`) and deterministic coverage examples so math, multiple-choice, coding, and general routing patterns are all seen during the smoke run.
- Supporting both old and new TRL `SFTTrainer` APIs (`tokenizer` vs. `processing_class`) so SFT does not break when Colab installs a newer `trl` version.

If Cell 8 still does not reach `3/3`, increase `SFT_NUM_EXAMPLES`/`SFT_MAX_STEPS` and rerun Cells 7 and 8. The same prompt shape is mirrored in `mini_conductor/harness.py` and the exported `conductor_harness_loader.py` so the adapter behaves identically at serving time.

## Local validation already run

The notebook JSON was generated and parsed successfully. Python code cells compile, and non-GPU/non-OpenRouter cells were executed locally through the parser, workflow executor dry-run, and SFT data generator.

## Colab workflow note

This project uses the normal Google Colab workflow: upload/open `Mini_Conductor_OpenRouter_Unsloth.ipynb` directly in Colab and run the notebook cells. No MCP setup is required.
