# HARNESS_PLAN

## 1. Purpose

This document is the standalone plan for the **Mini-Conductor harness**.

The harness is the runtime layer that turns a trained local Conductor model into an actually usable multi-model agent. The trained Conductor does **not** call OpenRouter by itself. It emits a structured workflow plan. The harness parses that plan, validates it, executes worker calls through OpenRouter, manages context passing, tracks cost, and returns the final answer plus a trace.

This file consolidates harness details that are partially documented in:

```text
DESIGN.md
PLAN.md
Mini_Conductor_OpenRouter_Unsloth.ipynb
generate_colab_notebook.py
```

The implementation target is a Google Colab Pro+ notebook using normal notebook execution. No MCP setup is part of this harness plan.

---

## 2. What “harness” means in this project

In this project, the **harness** is the Python orchestration runtime around the Conductor model.

It is responsible for:

1. constructing the Conductor prompt,
2. calling the local trained coordinator model,
3. extracting and validating the JSON workflow plan,
4. selecting visible prior outputs for each step via `access`,
5. constructing role-specific worker prompts,
6. calling OpenRouter worker models,
7. caching worker calls,
8. tracking OpenRouter cost,
9. enforcing a hard API budget cap,
10. collecting a full execution trace,
11. returning the final answer.

Runtime shape:

```text
User question
  -> ask_conductor_local(...)
  -> JSON plan
  -> parse_and_validate_plan(...)
  -> execute_workflow(...)
  -> OpenRouter workers
  -> final answer + trace + cost
```

---

## 3. Existing implementation status

A first harness implementation already exists inside the generated notebook and generator script.

Primary implementation file:

```text
generate_colab_notebook.py
```

Generated notebook:

```text
Mini_Conductor_OpenRouter_Unsloth.ipynb
```

Important implemented functions:

```text
build_conductor_messages(question, workers)
ask_conductor_local(question, workers, max_new_tokens, temperature)
extract_json_object(text)
validate_plan(plan, workers, allow_empty=False)
parse_and_validate_plan(text, workers, allow_empty=False)
select_history(history, access)
build_worker_messages(question, step, visible_history)
call_openrouter_worker_uncached(worker, messages, temperature, max_tokens)
cached_worker_call(worker, messages, temperature, max_tokens)
execute_workflow(question, plan, workers, max_worker_tokens, dry_run=False)
estimate_call_cost(worker, input_tokens, output_tokens)
estimate_plan_cost(plan, workers, assumed_input_tokens, assumed_output_tokens)
```

Notebook cell layout (as generated):

```text
CELL 1  Install dependencies (restarts kernel once)
CELL 2  Configuration, Drive mount, budget settings
CELL 3  Imports, worker registry, schema, prompts, cost helpers, parser
CELL 4  OpenRouter cached caller and workflow executor (harness core)
CELL 5  Heuristic plans, prompt builders, SFT warm-start data
CELL 6  Load local coordinator model with Unsloth
CELL 7  SFT warm-start training
CELL 8  Ask the local Conductor and validate JSON plans
CELL 9  Optional GRPO-lite reward functions and training
CELL 10 Export coordinator artifacts for harness use (portable harness package)
CELL 11 Final demo: local plan + dry run, or real OpenRouter execution
```

Current notebook behavior:

```text
RUN_MODEL_LOAD = True
RUN_SFT = True
RUN_GRPO = False
RUN_OPENROUTER_SMOKE = False
RUN_OPENROUTER_DEMO = False
```

So the default notebook run is safe: it loads/trains locally and dry-runs the harness without OpenRouter spend.

---

## 4. Harness component design

### 4.1 Worker registry

The harness uses a worker registry supplied at runtime.

Configured workers:

| ID | Worker model | Role | IN $/1M | OUT $/1M |
|---:|---|---|---:|---:|
| 0 | `qwen/qwen3.7-plus` | low-cost general reasoning/math | 0.32 | 1.28 |
| 1 | `minimax/minimax-m3` | cheapest broad/long-context worker | 0.30 | 1.20 |
| 2 | `moonshotai/kimi-k2.7-code` | coding specialist | 0.74 | 3.50 |
| 3 | `z-ai/glm-5.2` | hard reasoning/planning/verifier | 0.95 | 3.00 |

Registry object shape:

```python
{
    "id": 0,
    "name": "qwen/qwen3.7-plus",
    "provider": "openrouter",
    "description": "Low-cost general reasoning worker...",
    "strengths": ["general reasoning", "math", "long context", "low cost"],
    "cost_tier": "low",
    "input_cost_per_million": 0.32,
    "output_cost_per_million": 1.28,
}
```

Design requirements:

- The worker list is passed to the Conductor on every call.
- The Conductor should route by metadata, not memorized IDs.
- During training data generation, worker order should be randomized to reduce fixed-ID overfitting.
- At runtime, IDs may be fixed for simplicity, but the metadata must remain present in the prompt.

---

### 4.2 Conductor prompt builder

Function:

```python
build_conductor_messages(question, workers)
```

Inputs:

```text
question: user task
workers: current worker registry
```

Outputs:

```text
OpenAI-style messages for local coordinator generation
```

Prompt contents:

1. system prompt describing the Conductor role,
2. JSON schema expectations,
3. role definitions,
4. cost policy,
5. worker registry with IDs, descriptions, strengths, and prices,
6. user question.

Design requirements:

- The prompt must strongly say: **return JSON only**.
- The prompt must remind the model that output tokens are expensive.
- The prompt must explicitly describe when to use MiniMax/Qwen vs Kimi/GLM.
- The prompt must include the current worker registry each time.

---

### 4.3 Local Conductor caller

Function:

```python
ask_conductor_local(question, workers, max_new_tokens=768, temperature=0.2)
```

Steps:

1. Build messages with `build_conductor_messages`.
2. Apply tokenizer chat template.
3. Generate with local Qwen3-8B LoRA/QLoRA model.
4. Decode new tokens only.
5. Parse and validate JSON plan.
6. Return `(plan, raw_text)`.

Recommended inference settings:

```text
temperature = 0.0 for deterministic planning
max_new_tokens = 512-768 for workflow plans
```

Design requirements:

- Use low/zero temperature for stable plan generation.
- Return both parsed plan and raw generation for debugging.
- Fail fast if the plan is invalid.
- Do not call OpenRouter in this function.

---

### 4.4 JSON extractor and plan validator

Functions:

```python
extract_json_object(text)
validate_plan(plan, workers, allow_empty=False)
parse_and_validate_plan(text, workers, allow_empty=False)
```

Validation rules:

- top-level object contains `steps`,
- `steps` is an array,
- maximum 5 steps,
- each step has:
  - `model_id`,
  - `role`,
  - `subtask`,
  - `access`,
- `role` is one of:
  - `thinker`,
  - `worker`,
  - `verifier`,
- `model_id` is in range,
- `access` only refers to earlier steps,
- `access` may be `[]`, `["all"]`, or explicit prior indices,
- `access` must not mix `"all"` and explicit indices.

Empty plans:

- invalid for ordinary execution,
- valid only for bounded recursive accept mode with `allow_empty=True`.

Design requirements:

- Invalid plans must be rejected before any worker call.
- The validator is the main protection against expensive malformed rollouts.
- Any validation failure should include a useful error message for logging.

---

### 4.5 Access-list history selector

Function:

```python
select_history(history, access)
```

Purpose:

Select which previous worker outputs become visible to the next step.

Behavior:

```text
access = []       -> no prior outputs
access = ["all"] -> all prior outputs
access = [0, 2]   -> only prior steps 0 and 2
```

Design requirements:

- Never expose future steps.
- Preserve step metadata in visible history:
  - step number,
  - model name,
  - role,
  - subtask,
  - output.
- Keep context concise because input tokens still cost money.

Future improvement:

- Add summarization/compression for long histories.
- Add a context token budget per worker call.

---

### 4.6 Worker prompt builder

Function:

```python
build_worker_messages(question, step, visible_history)
```

Inputs:

```text
question: original user question
step: current plan step
visible_history: prior outputs selected by access list
```

Outputs:

```text
OpenAI-style messages for OpenRouter worker call
```

Role prompts:

```text
thinker:
  Plan, decompose, identify pitfalls, propose strategy. Be concise.

worker:
  Use context and assigned subtask to make concrete progress.

verifier:
  Check correctness, edge cases, formatting, and provide corrected final answer.
```

Design requirements:

- Always include original question.
- Include only `visible_history`, not all history unless requested.
- Include the assigned subtask verbatim.
- Keep system prompts role-specific and short.

---

### 4.7 OpenRouter caller

Function:

```python
call_openrouter_worker_uncached(worker, messages, temperature=0.2, max_tokens=1024)
```

Responsibilities:

1. verify `OPENROUTER_API_KEY` is available,
2. estimate maximum possible call cost before making the request,
3. enforce `OPENROUTER_BUDGET_USD`,
4. call OpenRouter's OpenAI-compatible chat completions endpoint,
5. capture content and usage,
6. estimate actual cost from usage,
7. update session spend estimate,
8. retry transient failures.

Design requirements:

- Use low temperature for worker calls, usually `0.2`.
- Use `max_tokens=512` during training and `1024-2048` during evaluation/demo.
- Use `extra_headers` identifying the Colab app.
- Do not silently ignore budget failures.

---

### 4.8 Cache layer

Function:

```python
cached_worker_call(worker, messages, temperature=0.2, max_tokens=1024)
```

Cache key fields:

```text
worker name
messages
temperature
max_tokens
```

Current cache backend:

```python
diskcache.Cache(str(CACHE_DIR))
```

Persistent Colab path:

```text
/content/drive/MyDrive/mini_conductor/cache/openrouter_cache
```

Design requirements:

- Cache every successful worker call.
- Mark each trace step with `cache_hit`.
- Do not charge cached calls against the session spend estimate.
- Warn users that the cache may contain prompts and responses.

Future improvement:

- Add optional cache namespace/version.
- Add cache pruning.
- Add a flag to disable cache for fresh evals.

---

### 4.9 Workflow executor

Function:

```python
execute_workflow(question, plan, workers, max_worker_tokens=1024, dry_run=False)
```

Algorithm:

```text
validate plan
initialize empty history
for each step:
  get worker by model_id
  select visible history using access list
  build role-specific worker messages
  if dry_run:
    create fake output and zero cost
  else:
    call cached_worker_call
  append step record to history
return final output from last step and full trace
```

Trace record shape:

```python
{
    "step": 0,
    "model_id": 1,
    "worker_name": "minimax/minimax-m3",
    "role": "worker",
    "subtask": "Answer directly.",
    "access": [],
    "output": "...",
    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    "cost_usd": 0.0,
    "cache_hit": False,
}
```

Design requirements:

- Return both final answer and full trace.
- Support `dry_run=True` for safe demos and debugging.
- Execute steps sequentially for first version.
- Do not execute invalid plans.

Future improvement:

- Parallelize independent steps whose `access` is empty.
- Add per-step timeout controls.
- Add step-level retry policy.
- Add automatic verifier early-stop behavior.

---

## 4.10 Harness artifact export and portable loader (CELL 10)

The notebook does not only run the harness in-process. CELL 10 exports a **portable harness package** so the trained coordinator can be reloaded later (a fresh Colab session, a separate inference notebook, or a server) without rerunning training.

Artifacts are written to:

```text
HARNESS_DIR = ROOT / 'harness_artifacts'
# Colab default: /content/drive/MyDrive/mini_conductor/harness_artifacts
```

Exported artifacts:

```text
manifest.json                  # full harness specification for reload
latest_adapter.txt             # pointer to the most recent adapter dir
conductor_harness_loader.py    # standalone loader, no notebook required
merged_16bit/ (optional)       # full merged model if EXPORT_MERGED_16BIT=True
```

Adapter selection:

```text
find_latest_adapter() prefers checkpoints/grpo_lora, then checkpoints/sft_lora.
It raises if neither exists, so SFT (Cell 7) or GRPO (Cell 9) must run first.
```

`manifest.json` is the single source of truth for reload. It records:

```text
artifact_type, created_at
adapter_kind (grpo|sft), adapter_dir
base_model, fallback_base_model
max_seq_length, lora_rank
workers (full registry with IN/OUT prices)
coordinator_schema
conductor_system_prompt
role_prompts
load_mode
merged_16bit_dir (only when merged export is enabled)
```

Why the manifest embeds the worker registry, schema, and prompts:

- a reloaded coordinator must be prompted **exactly** as it was trained, otherwise plan quality and JSON validity drop,
- embedding the schema lets a downstream consumer validate plans without importing notebook code,
- embedding prices keeps cost-aware routing consistent at inference time.

Portable loader (`conductor_harness_loader.py`) provides:

```python
load_manifest(harness_dir)
load_conductor(harness_dir, load_in_4bit=True)   # returns (model, tokenizer, manifest)
build_conductor_messages(question, workers, manifest)
```

This loader is intentionally self-contained: it depends only on `json`, `pathlib`, and `unsloth`. A downstream inference service can therefore reload the coordinator and reuse the same execution harness (Sections 4.5-4.9) without the training notebook.

Optional merged export:

```text
EXPORT_MERGED_16BIT = False   # default
```

- When `True`, CELL 10 also writes a full 16-bit merged model via `save_pretrained_merged(..., "merged_16bit")` and records `merged_16bit_dir` in the manifest.
- Default is `False` because merging uses substantially more disk/RAM and is only needed for a standalone (non-LoRA) deployment, for example serving behind vLLM.

Design requirements:

- Export must run after a successful SFT or GRPO checkpoint exists.
- The manifest must stay in sync with the worker registry, schema, and prompts used in training.
- Reload paths must use `manifest["adapter_dir"]` (or the merged dir) rather than hardcoded paths.
- Secrets such as the OpenRouter key must never be written into the manifest or any artifact.

Note on `build_conductor_messages`: the in-notebook version takes `(question, workers)` and reads the module-level `CONDUCTOR_SYSTEM_PROMPT`; the exported loader version takes `(question, workers, manifest)` and reads the prompt from the manifest. Both must produce equivalent message structure so plans behave the same in-notebook and after reload.

---

## 5. Harness integration with training

### 5.1 SFT phase

During SFT, the harness does **not** call OpenRouter.

Harness-related responsibilities:

- generate target plans with valid worker IDs,
- randomize worker order,
- include worker registry in the Conductor prompt,
- validate target plans before adding them to SFT dataset.

SFT teaches the coordinator to produce valid plans before expensive worker-executing RL.

---

### 5.2 GRPO-lite phase

During GRPO-lite, the harness can be called inside `correctness_reward_func`.

Flow:

```text
Conductor completion
  -> parse_and_validate_plan
  -> execute_workflow(..., dry_run=False)
  -> grade final answer
  -> reward
```

Safety requirements:

- `RUN_GRPO=False` by default.
- `OPENROUTER_BUDGET_USD` must be set.
- worker `max_tokens` should be low, e.g. 512.
- GRPO should start with easy-to-grade tasks only.
- coding GRPO should wait until unit-test grading exists.

---

## 6. Harness operational modes

### 6.1 Dry-run mode

```python
execute_workflow(..., dry_run=True)
```

Use for:

- validating plan routing,
- demonstrating the harness without API spend,
- debugging trace shape.

Expected cost:

```text
$0
```

### 6.2 OpenRouter smoke mode

```python
RUN_OPENROUTER_SMOKE = True
```

Use for:

- testing API key,
- testing OpenRouter connectivity,
- validating cache and cost tracking.

Expected call:

```text
one short MiniMax call with max_tokens=16
```

### 6.3 Demo execution mode

```python
RUN_OPENROUTER_DEMO = True
```

Use for:

- executing one real workflow,
- inspecting trace,
- confirming cost is acceptable.

### 6.4 GRPO execution mode

```python
RUN_GRPO = True
```

Use only after dry-run and smoke/demo mode work.

---

## 7. Budget and cost controls

### 7.1 Exact worker pricing

| Worker | IN $/1M | OUT $/1M |
|---|---:|---:|
| `minimax/minimax-m3` | 0.30 | 1.20 |
| `qwen/qwen3.7-plus` | 0.32 | 1.28 |
| `moonshotai/kimi-k2.7-code` | 0.74 | 3.50 |
| `z-ai/glm-5.2` | 0.95 | 3.00 |

### 7.2 Cost estimate formula

```python
cost_usd = (input_tokens / 1_000_000) * input_cost_per_million \
         + (output_tokens / 1_000_000) * output_cost_per_million
```

### 7.3 Budget cap behavior

Before each uncached call:

```text
estimate prompt tokens
estimate max possible output cost from max_tokens
if current_spend + max_possible > cap:
  raise RuntimeError
```

After each call:

```text
read usage fields if available
compute actual cost
update spend estimate
save result to cache
```

### 7.4 Cost-control defaults

```text
OPENROUTER_BUDGET_USD = 5.00
WORKER_MAX_TOKENS_TRAIN = 512
WORKER_MAX_TOKENS_EVAL = 1024
```

---

## 8. Error handling plan

### Invalid Conductor output

Action:

- reject before worker calls,
- log raw output,
- during demo, optionally fall back to heuristic plan,
- during training, assign negative format/schema reward.

### OpenRouter API failure

Action:

- retry transient failures up to 3 times,
- exponential backoff,
- propagate persistent errors,
- do not cache failures by default.

### Budget exceeded

Action:

- raise `RuntimeError`,
- stop GRPO/demo cleanly,
- report current estimated spend.

### Missing API key

Action:

- raise clear error if an OpenRouter mode is enabled,
- avoid requiring a key for dry-run/SFT-only mode.

### Dataset/model errors

Action:

- use fallback examples if datasets fail,
- fallback coordinator model if 8B load fails,
- reduce sequence length/rank if memory fails.

---

## 9. Security and privacy plan

- Do not hardcode `OPENROUTER_API_KEY`.
- Use Colab secrets or `getpass`.
- Do not write API keys to Google Drive.
- Warn that `diskcache` stores prompts and responses.
- Do not share cache files if prompts/responses are sensitive.
- Keep worker prompts free of unnecessary secrets.

---

## 10. Testing plan

### 10.1 Local/no-GPU tests

Already validated locally:

- notebook JSON parses,
- Python code cells compile except shell install cell,
- parser and schema validation run,
- dry-run executor runs,
- SFT data generation runs.

### 10.2 Colab dry-run tests

Run default notebook settings:

```text
RUN_MODEL_LOAD=True
RUN_SFT=True
RUN_GRPO=False
RUN_OPENROUTER_SMOKE=False
RUN_OPENROUTER_DEMO=False
```

Expected:

- model loads,
- short SFT run completes,
- local plan generation validates,
- dry-run demo produces trace,
- no OpenRouter spend.

### 10.3 API smoke test

Set:

```python
RUN_OPENROUTER_SMOKE=True
```

Expected:

- one MiniMax call,
- nonzero usage/cost or fallback estimate,
- cache entry created.

### 10.4 Real workflow demo

Set:

```python
RUN_OPENROUTER_DEMO=True
```

Expected:

- local Conductor emits valid plan,
- harness executes each worker step,
- final answer returned,
- trace includes cost and cache-hit fields.

### 10.5 GRPO-lite smoke test

Set:

```python
RUN_GRPO=True
OPENROUTER_BUDGET_USD=5.00
GRPO_MAX_STEPS=20
GRPO_NUM_EXAMPLES=24
WORKER_MAX_TOKENS_TRAIN=512
```

Expected:

- budget guard prevents runaway spend,
- invalid plans receive negative reward,
- valid/correct cheap workflows receive better reward,
- LoRA checkpoint saved.

---

## 11. Evaluation plan for harness quality

Harness-specific metrics:

```text
plan parse success rate
schema validation success rate
worker call success rate
cache hit rate
average worker calls per question
average actual cost per question
average estimated cost per question
budget failures
retry count
latency per workflow
trace completeness
```

Model-quality metrics remain separate:

```text
final answer accuracy
role/model selection distribution
cost-adjusted accuracy
```

---

## 12. Future harness extensions

### 12.1 Parallel execution

If multiple steps have `access=[]`, they can be executed concurrently.

Need:

- dependency graph from access lists,
- async OpenRouter calls,
- concurrency limit,
- cost cap compatible with concurrent calls.

### 12.2 Context compression

For long traces, add a summarizer step or automatic truncation.

Need:

- token estimator,
- max context budget per worker,
- summarized prior-output cache.

### 12.3 Recursive refinement

Bounded recursion mode:

```text
initial workflow -> final answer -> Conductor reviews answer -> accept or revise
```

Need:

- `allow_empty=True` accept plan,
- maximum recursion depth,
- separate recursion budget.

### 12.4 Additional worker backends

Potential future backends:

```text
local vLLM server
Claude Code CLI
Codex CLI
other OpenAI-compatible endpoints
```

For this Colab-first harness, only OpenRouter is included initially.

### 12.5 Richer grading

For coding GRPO:

- add unit-test execution,
- sandbox generated code,
- grade pass/fail safely,
- avoid arbitrary unsafe execution.

---

## 13. Uncertainties / decisions still needed

1. **OpenRouter budget cap**
   - Default is `$5.00`, but user should confirm desired cap for the first real GRPO run.

2. **Primary task mix**
   - Current harness supports mixed examples but first GRPO should emphasize math/MCQ because grading is easier.

3. **Coding evaluation**
   - Kimi is configured as the coding specialist, but robust coding GRPO needs unit-test grading before heavy use.

4. **Exact Colab GPU**
   - If Colab allocates a weaker GPU, reduce coordinator size/settings or run parser/SFT smoke only.

5. **OpenRouter usage fields**
   - Some providers may not return complete token usage. The harness includes fallback estimates, but actual billing should be checked against OpenRouter dashboards.

6. **Cache sensitivity**
   - Prompts/responses are cached to Drive. This is useful but may be sensitive depending on user data.

---

## 14. Immediate implementation checklist

1. Keep `HARNESS_PLAN.md` as the harness specification.
2. Keep harness code in the notebook generator and generated notebook.
3. Open `Mini_Conductor_OpenRouter_Unsloth.ipynb` in Colab.
4. Run default dry mode.
5. Confirm local Conductor emits valid JSON.
6. Turn on `RUN_OPENROUTER_SMOKE=True` for one API call.
7. Turn on `RUN_OPENROUTER_DEMO=True` for one workflow (CELL 11).
8. Only then turn on `RUN_GRPO=True` with a budget cap.
9. Run CELL 10 to export the portable harness package (`manifest.json`, `latest_adapter.txt`, `conductor_harness_loader.py`) to `harness_artifacts/`.
10. To reuse the coordinator elsewhere, call `load_conductor(harness_dir)` from `conductor_harness_loader.py`, then drive Sections 4.5-4.9 of this harness.
