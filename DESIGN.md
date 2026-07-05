# DESIGN

## 1. Purpose

This document describes the overall design for the **Mini-Conductor OpenRouter + Unsloth** project.

The project goal is to build a Google Colab Pro+ notebook that trains and runs a practical, scaled-down version of the Sakana **Conductor** / **TRINITY** ideas:

```text
User question
  -> local trained Conductor/coordinator model
  -> JSON workflow plan
  -> Python harness executes workflow through OpenRouter worker models
  -> final answer returned to user
```

The Conductor model is not trained to directly solve every problem itself. Instead, it is trained to plan and route tasks to stronger worker models, using a cost-aware Thinker / Worker / Verifier orchestration pattern.

This design is intended to be runnable primarily inside **Google Colab Pro+**. Lambda.ai is a fallback for cases where Colab cannot provide sufficient GPU access or runtime stability.

No Google Colab MCP dependency is part of this design.

---

## 2. Design goals

### Primary goals

1. **Colab-first implementation**
   - Provide a `.ipynb` notebook that can be uploaded to Google Colab Pro+ and executed cell-by-cell.
   - Avoid requiring external services beyond Hugging Face downloads and OpenRouter API calls.

2. **Conductor-style coordination**
   - Train a local coordinator model to emit structured JSON workflow plans.
   - The plan selects worker models, roles, subtasks, and access to prior worker outputs.

3. **TRINITY-style roles**
   - Incorporate three roles:
     - `thinker`
     - `worker`
     - `verifier`

4. **OpenRouter worker execution**
   - Execute worker steps through OpenRouter models:
     - `z-ai/glm-5.2`
     - `moonshotai/kimi-k2.7-code`
     - `qwen/qwen3.7-plus`
     - `minimax/minimax-m3`

5. **Cost-aware routing**
   - The Conductor should prefer cheaper workers for simple tasks and reserve expensive models for coding, hard reasoning, or final verification.
   - The harness should track and cap OpenRouter spend.

6. **Safe staged rollout**
   - Start with SFT format warm-up.
   - Only run OpenRouter worker calls when explicitly enabled.
   - Only run GRPO after parser, SFT, and worker execution are validated.

7. **Reproducible artifacts**
   - Store notebooks, checkpoints, cache files, logs, evaluation outputs, and model adapters in persistent storage.

---

## 3. Non-goals

This project is **not** intended to reproduce the full Sakana results in the first run.

Non-goals for the first implementation:

1. Full paper-scale Conductor training.
   - The paper used roughly:

   ```text
   200 iterations * 4 questions * 64 rollouts = 51,200 workflow rollouts
   ```

   That is outside the first Colab prototype scope.

2. Full TRINITY hidden-state CMA-ES implementation.
   - TRINITY uses a Qwen3-0.6B backbone, hidden-state extraction, a lightweight trainable head, singular-value fine-tuning, and sep-CMA-ES.
   - This design starts with a more Colab-compatible generative JSON coordinator trained with SFT and optional GRPO-lite.

3. Guaranteed paper-level benchmark performance.
   - The first target is a working Mini-Conductor prototype, not state-of-the-art benchmark results.

4. Automatic Google Colab MCP orchestration.
   - The design uses a normal Colab notebook workflow.
   - MCP is intentionally excluded.

5. High-volume API spending.
   - The first GRPO runs should operate under a strict OpenRouter budget cap.

---

## 4. System overview

### Runtime architecture

```text
+-----------------------------+
| Google Colab Pro+ Notebook  |
+-----------------------------+
              |
              v
+-----------------------------+
| Local Conductor Model       |
| Qwen3-8B LoRA / QLoRA       |
+-----------------------------+
              |
              | emits JSON workflow
              v
+-----------------------------+
| Workflow Parser/Validator   |
| JSON schema + ID checks     |
+-----------------------------+
              |
              v
+-----------------------------+
| Workflow Executor/Harness   |
| access_list + role prompts  |
+-----------------------------+
              |
              v
+-----------------------------+
| OpenRouter Worker Models    |
| Qwen / MiniMax / Kimi / GLM |
+-----------------------------+
              |
              v
+-----------------------------+
| Final Answer + Trace + Cost |
+-----------------------------+
```

### Training architecture

```text
Synthetic / public dataset questions
       |
       v
Heuristic workflow generation
       |
       v
SFT warm-start dataset
       |
       v
Local Qwen3-8B coordinator LoRA training
       |
       v
JSON-valid Conductor
       |
       v
Optional GRPO-lite with real worker execution
       |
       v
Cost-aware empirical routing behavior
```

---

## 5. Core components

### 5.1 Local Conductor model

Recommended default:

```text
unsloth/Qwen3-8B-unsloth-bnb-4bit
```

Backup:

```text
unsloth/Qwen3-8B
unsloth/Qwen3-4B-Base
unsloth/Qwen2.5-7B
```

Rationale:

- The Sakana Conductor paper used Qwen2.5-7B.
- TRINITY used Qwen3-0.6B.
- Qwen3-8B is an 8B-class Qwen-family model and is a natural practical coordinator candidate.
- Unsloth 4-bit loading makes the model feasible on Colab Pro+ GPUs.

Recommended first-run settings:

```python
COORDINATOR_MODEL = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
```

Larger settings if GPU memory is stable:

```python
MAX_SEQ_LENGTH = 4096
LORA_RANK = 32
```

---

### 5.2 Worker registry

The worker registry contains OpenRouter model IDs, descriptions, strengths, and exact IN/OUT costs.

| Worker model | Role in system | IN $/1M | OUT $/1M |
|---|---|---:|---:|
| `minimax/minimax-m3` | cheapest long-context general worker/summarizer/verifier | 0.30 | 1.20 |
| `qwen/qwen3.7-plus` | low-cost general reasoning/math worker | 0.32 | 1.28 |
| `moonshotai/kimi-k2.7-code` | coding specialist | 0.74 | 3.50 |
| `z-ai/glm-5.2` | hard reasoning/planning/verification | 0.95 | 3.00 |

Rationale:

- MiniMax and Qwen are the cheapest requested models and should be default choices for low/medium complexity tasks.
- Kimi has the highest output cost but is coding-focused, so it should be reserved for code-heavy subtasks.
- GLM is comparatively expensive but suited to hard reasoning and long-horizon agent workflows.

---

### 5.3 Workflow JSON schema

The Conductor emits JSON only:

```json
{
  "steps": [
    {
      "model_id": 0,
      "role": "thinker",
      "subtask": "Create a concise solution plan.",
      "access": []
    }
  ]
}
```

Schema constraints:

- top-level object must contain `steps`.
- `steps` is an array with max length 5.
- each step must contain:
  - `model_id`: integer
  - `role`: one of `thinker`, `worker`, `verifier`
  - `subtask`: non-empty string
  - `access`: array of prior-step indices or `"all"`

Validation also checks:

- `model_id` is in range for the current worker pool.
- `access` only refers to previous steps.
- `access` does not mix `"all"` with explicit indices.

Rationale:

- JSON is easier to parse and validate than the original paper's Python-list format.
- Strict schema validation reduces wasted worker calls during GRPO.
- JSON can later be extended to support budgets, confidence, recursion, or tool requirements.

---

### 5.4 Role prompts

Each worker call receives a role-specific system prompt.

Roles:

```text
thinker:
  Plan, decompose, identify pitfalls, propose strategy.

worker:
  Solve, implement, calculate, draft, or make concrete progress.

verifier:
  Check correctness, edge cases, formatting, and produce corrected final answer.
```

Rationale:

- This borrows TRINITY's role abstraction.
- Roles make worker behavior more predictable.
- Roles provide simple structure for the Conductor to reason over.

---

### 5.5 Access-list context passing

Each step controls prior context visibility:

```json
"access": []
```

means no previous worker outputs are visible.

```json
"access": ["all"]
```

means all previous worker outputs are visible.

```json
"access": [0, 2]
```

means only previous steps 0 and 2 are visible.

Rationale:

- This mirrors the Conductor paper's `access_list` idea.
- It allows independent attempts, sequential refinement, tree-like workflows, and final aggregation.
- It also controls input-token cost by limiting context passed to later workers.

---

### 5.6 OpenRouter caller and cache

The harness uses OpenRouter's OpenAI-compatible API.

The caller must:

1. use `OPENROUTER_API_KEY` from environment or Colab secret,
2. cache calls by worker/model/messages/temperature/max_tokens,
3. estimate cost before uncached calls,
4. stop before exceeding a budget cap,
5. record usage, cost, and cache hits.

Persistent cache path:

```text
/content/drive/MyDrive/mini_conductor/cache/openrouter_cache
```

Rationale:

- GRPO may produce many repeated or similar calls.
- Caching prevents duplicate spend.
- Budget caps prevent runaway API cost.

---

### 5.7 Cost model

Cost formula:

```python
cost_usd = (input_tokens / 1_000_000) * input_cost_per_million \
         + (output_tokens / 1_000_000) * output_cost_per_million
```

Because output tokens are substantially more expensive than input tokens, training should use conservative worker `max_tokens`.

Recommended first-run worker max token limits:

```text
OpenRouter smoke: 16-64 tokens
Training worker calls: 512 tokens
Evaluation worker calls: 1024 tokens
Final demos: 1024-2048 tokens
```

Rationale:

- Worker outputs dominate spend.
- GRPO multiplies costs by `num_prompts * num_generations * workflow_steps`.
- Cost-aware training should reward short, cheap, correct workflows.

---

## 5.8 Inference / serving design (using the trained model as a Conductor)

Training produces a coordinator; this section defines how it is actually *used*. At inference time the trained model never calls OpenRouter itself — it only emits a plan, and the harness executes it. The end-to-end call path implemented in the notebook is:

```text
ask_conductor_local(question, workers)
  -> build_conductor_messages(question, workers)        # system prompt + worker registry as JSON
  -> tokenizer.apply_chat_template(..., add_generation_prompt=True)
  -> model.generate(...)                                # local Qwen3-8B LoRA, greedy at temp=0 for stability
  -> parse_and_validate_plan(decoded, workers)          # extract JSON, schema + ID + access checks
  -> execute_workflow(question, plan, workers)          # runs each step
       -> select_history(history, access)               # apply access-list visibility
       -> build_worker_messages(...)                    # role prompt + visible prior outputs
       -> cached_worker_call(worker, messages, ...)     # OpenRouter call (cached, budget-capped)
  -> (final_answer, trace)
```

Design points:

- **Separation of concerns.** The coordinator decides *who does what*; the harness performs the calls. This keeps the worker pool swappable without retraining and matches the Conductor paper's parse-then-execute model.
- **Deterministic planning at inference.** Plan generation defaults to greedy decoding (`temperature=0`) so plans are reproducible; worker calls use a low temperature (`0.2`).
- **Graceful degradation.** If the trained model is absent or emits an unparseable plan, the demo path falls back to `heuristic_plan_for_question(...)` so the harness is still exercisable. This is a usability decision, not a substitute for a valid model.
- **Workers supplied at call time.** The worker registry (ids, descriptions, strengths, IN/OUT costs) is injected into the prompt on every call via `build_conductor_messages`, so the same trained model adapts to a changed OpenRouter lineup without retraining.
- **Serving options.** For the first version, in-process `model.generate` is sufficient. A later option is to serve the merged coordinator behind an OpenAI-compatible endpoint (for example vLLM) and call it like any other model; the harness logic is unchanged.

## 5.9 Optional recursion mode

The design optionally supports a single Conductor self-review pass, mirroring the Conductor paper's recursive topology:

```text
execute initial workflow -> final_answer_v1
  -> re-prompt Conductor with final_answer_v1
       -> if acceptable: return {"steps": []}  (accept, no further work)
       -> else: emit a short (<=3 step) revision workflow and execute it
```

Rules:

- recursion is bounded by a fixed maximum number of passes to prevent infinite loops,
- the empty-plan (`{"steps": []}`) sentinel is the accept signal and is the one place an empty plan is valid,
- recursion is **off** in the first GRPO run and added only after the base execute path is stable.

Rationale:

- recursion is a cheap, optional test-time quality lever,
- keeping it disabled initially avoids extra OpenRouter spend and reward-shaping complexity.

---

## 6. Data design

### 6.1 SFT warm-start data

The SFT stage teaches:

- JSON format,
- valid worker IDs,
- valid roles,
- reasonable workflow patterns,
- cost-aware routing heuristics.

Data sources:

```text
HuggingFaceH4/MATH-500
cais/mmlu
openbmb/RLPR-Train-Dataset
livecodebench/code_generation_lite
fallback built-in synthetic examples
```

The current implementation includes fallback built-in examples so the notebook can run even if dataset downloads fail.

### 6.2 Heuristic target generation

Heuristic patterns:

#### General/easy

```json
{
  "steps": [
    {
      "model_id": "MiniMax ID",
      "role": "worker",
      "subtask": "Answer directly and concisely.",
      "access": []
    }
  ]
}
```

#### Math/reasoning

```json
{
  "steps": [
    {"model_id": "Qwen ID", "role": "thinker", "subtask": "Develop a concise plan.", "access": []},
    {"model_id": "MiniMax ID", "role": "worker", "subtask": "Solve using the plan.", "access": ["all"]},
    {"model_id": "Qwen ID", "role": "verifier", "subtask": "Verify and final answer.", "access": ["all"]}
  ]
}
```

#### Coding

```json
{
  "steps": [
    {"model_id": "GLM ID", "role": "thinker", "subtask": "Design algorithm and edge cases.", "access": []},
    {"model_id": "Kimi ID", "role": "worker", "subtask": "Implement solution.", "access": ["all"]},
    {"model_id": "Kimi ID", "role": "verifier", "subtask": "Review code and final formatting.", "access": ["all"]}
  ]
}
```

### 6.3 Worker randomization

During SFT data generation, worker order is randomized and IDs are re-assigned.

Rationale:

- Prevents memorizing fixed IDs.
- Forces the Conductor to read model metadata.
- Better matches the Conductor paper's randomized worker-pool idea.

---

## 7. Training design

### 7.1 Stage 1: SFT warm-start

Default first-run settings:

```text
SFT examples: 120
max steps: 25
learning rate: 2e-4
batch size: 1
gradient accumulation: 4
optimizer: adamw_8bit
```

Success criteria before moving to GRPO:

```text
>95% valid JSON on held-out prompts
valid worker IDs
valid role names
reasonable task-dependent workflows
low direct-answering tendency
```

Rationale:

- SFT is cheap and does not call OpenRouter workers.
- It minimizes expensive invalid GRPO rollouts.

---

### 7.2 Stage 2: GRPO-lite

Default disabled:

```python
RUN_GRPO = False
```

Enable only after SFT and harness pass smoke tests.

First GRPO-lite target:

```text
24 prompts
4 generations
20 max GRPO steps
512 worker output tokens
$5 OpenRouter budget cap
```

Reward functions:

1. `format_reward_func`
   - rewards parseable, schema-valid JSON.

2. `role_reward_func`
   - rewards reasonable role/model choices.

3. `cost_reward_func`
   - penalizes expensive workflows.

4. `correctness_reward_func`
   - executes workflow and grades final answer for easy-to-grade tasks.

Rationale:

- This is a safe first empirical routing stage.
- It starts with math and multiple-choice because grading is simpler.
- Coding GRPO should be added after unit-test grading is implemented.

---

## 8. Evaluation design

Baselines:

1. `minimax/minimax-m3` single-worker baseline.
2. `qwen/qwen3.7-plus` single-worker baseline.
3. `moonshotai/kimi-k2.7-code` coding-only baseline.
4. `z-ai/glm-5.2` reasoning/planning baseline.
5. Fixed workflow baseline.
6. Random router baseline.
7. Trained Mini-Conductor.

Metrics:

```text
valid JSON rate
schema-valid rate
accuracy/pass rate
average worker calls per question
average estimated cost
actual OpenRouter spend
cache hit rate
latency
role distribution
model selection distribution
```

Rationale:

- Accuracy alone is insufficient.
- The project goal is cost-aware orchestration, so cost and workflow length matter.

---

## 9. Runtime and storage design

### 9.1 Colab Pro+ runtime

Preferred GPUs:

```text
A100 40GB/80GB
H100
RTX 6000 Ada 48GB
```

Acceptable for smaller smoke tests:

```text
L4
V100
T4 only for data/parser/debugging or tiny tests
```

### 9.2 Persistent storage

Use Google Drive:

```text
/content/drive/MyDrive/mini_conductor/
/content/drive/MyDrive/mini_conductor/checkpoints/
/content/drive/MyDrive/mini_conductor/cache/
/content/drive/MyDrive/mini_conductor/evals/
```

Save:

```text
SFT LoRA adapter
GRPO LoRA adapter
OpenRouter cache
evaluation JSON/CSV
workflow traces
training config
```

Rationale:

- Colab local disk is temporary.
- Checkpoints must survive disconnections.

---

## 10. Operational modes

The notebook has explicit flags:

```python
RUN_MODEL_LOAD = True
RUN_SFT = True
RUN_GRPO = False
RUN_OPENROUTER_SMOKE = False
RUN_OPENROUTER_DEMO = False
```

### Default mode

Safe mode:

```text
load model
build data
run short SFT
validate plan generation
execute dry-run demo
no OpenRouter spend
```

### Smoke API mode

```python
RUN_OPENROUTER_SMOKE = True
```

Makes one cheap API call.

### Demo mode

```python
RUN_OPENROUTER_DEMO = True
```

Executes one workflow through OpenRouter.

### GRPO mode

```python
RUN_GRPO = True
```

Runs worker-executing GRPO-lite and can spend OpenRouter credits.

---

## 11. Failure modes and mitigations

### Invalid JSON generations

Mitigation:

- increase SFT examples,
- increase SFT steps,
- lower generation temperature for inference,
- add more schema examples,
- keep GRPO disabled until valid JSON is reliable.

### OpenRouter runaway spend

Mitigation:

- use `OPENROUTER_BUDGET_USD`,
- preflight cost estimates before uncached calls,
- cache worker calls,
- cap worker output tokens,
- prefer MiniMax/Qwen in prompt and rewards.

### Colab dependency conflicts

Mitigation:

- restart runtime after install cell,
- avoid vLLM initially,
- use standard `model.generate`,
- fallback to smaller coordinator model.

### GPU memory errors

Mitigation:

- use 4-bit model,
- reduce `MAX_SEQ_LENGTH`,
- reduce `LORA_RANK`,
- reduce SFT batch size,
- fallback to Qwen3-4B.

### Dataset download failures

Mitigation:

- use built-in fallback examples,
- keep first SFT smoke independent of external datasets.

### Weak GRPO signal

Mitigation:

- begin with easy-to-grade MATH/MMLU tasks,
- avoid coding GRPO until unit-test grading exists,
- use stronger format/schema rewards early,
- keep correctness reward dominant once workflows are valid.

---

## 12. Security and secret handling

- Do not hardcode OpenRouter API keys.
- Use Colab secrets or `getpass`.
- Do not save API keys to Drive.
- Be cautious when logging prompts and outputs because they may include sensitive user data.
- OpenRouter cache stores prompts and responses; do not share cache files if prompts are sensitive.

---

## 13. Implementation artifacts

Current expected files:

```text
DESIGN.md
DISCOVERY.md
PLAN.md
README_COLAB.md
generate_colab_notebook.py
Mini_Conductor_OpenRouter_Unsloth.ipynb
```

`Mini_Conductor_OpenRouter_Unsloth.ipynb` is the primary user-facing artifact for Colab.

---

## 14. Design decisions summary

### Decision: use JSON instead of Python lists

Reason:

- easier validation,
- less ambiguity,
- better for production harnessing,
- easier to extend.

### Decision: use Qwen3-8B coordinator

Reason:

- Qwen-family matches both Sakana papers,
- 8B scale is practical in Colab with Unsloth,
- stronger than a very small coordinator for prompt engineering.

### Decision: start with SFT before GRPO

Reason:

- SFT is cheap,
- prevents invalid expensive rollouts,
- establishes schema-following behavior.

### Decision: make GRPO optional and disabled by default

Reason:

- GRPO executes real worker calls,
- can spend OpenRouter credits,
- should only run after validation.

### Decision: prefer MiniMax/Qwen for default cheap routing

Reason:

- they are the cheapest requested workers,
- output tokens dominate cost,
- most training rollouts should avoid unnecessary GLM/Kimi calls.

### Decision: reserve Kimi for coding and GLM for hard reasoning

Reason:

- Kimi has coding specialization but high output cost,
- GLM is more expensive but suited for hard long-horizon reasoning.

### Decision: no Colab MCP dependency

Reason:

- user requested ignoring MCP,
- normal `.ipynb` upload/copy workflow is simpler and more reliable,
- design should not rely on interactive MCP setup.

---

## 15. Next steps after design

1. Verify `Mini_Conductor_OpenRouter_Unsloth.ipynb` is present and opens in Colab.
2. Run default safe mode with no OpenRouter calls.
3. Confirm valid JSON generation after SFT.
4. Enable one OpenRouter smoke call.
5. Enable one real OpenRouter demo workflow.
6. If cost and behavior look sane, enable GRPO-lite with a small budget cap.
7. Save LoRA adapters and evaluation outputs to Google Drive.
