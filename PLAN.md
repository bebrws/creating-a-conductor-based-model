# PLAN

This plan describes how to build a Colab Pro+ notebook that trains and uses a Mini-Conductor agent using your OpenRouter API key and these worker models:

```text
z-ai/glm-5.2
moonshotai/kimi-k2.7-code
qwen/qwen3.7-plus
minimax/minimax-m3
```

The preference is to keep everything inside a **Google Colab Pro+ notebook** and use Lambda.ai only if absolutely necessary.

---

## 0. High-level goal

Build a practical version of the Sakana Conductor/TRINITY ideas:

```text
Local trained coordinator model in Colab
  -> emits JSON workflow plan
  -> Python harness executes steps using OpenRouter worker models
  -> final answer is returned
```

The local coordinator should learn to select among your OpenRouter workers and assign roles:

```text
thinker
worker
verifier
```

The worker models are not built into the trained model. The trained model outputs a plan. A Python harness in the notebook calls OpenRouter.

---

## 1. Recommended architecture

Use a **Mini-Conductor** design:

```text
Conductor paper structure:
- model_id
- subtask
- access_list

TRINITY paper role structure:
- thinker
- worker
- verifier
```

The coordinator outputs JSON like:

```json
{
  "steps": [
    {
      "model_id": 0,
      "role": "thinker",
      "subtask": "Create a solution strategy and identify likely pitfalls.",
      "access": []
    },
    {
      "model_id": 1,
      "role": "worker",
      "subtask": "Solve the problem using the prior plan.",
      "access": ["all"]
    },
    {
      "model_id": 2,
      "role": "verifier",
      "subtask": "Check the solution for correctness and produce the final answer.",
      "access": ["all"]
    }
  ]
}
```

The harness then executes each step with the chosen OpenRouter model.

---

## 2. Recommended coordinator model

Use an 8B-class Qwen-family model because the papers used Qwen-family coordinators:

```text
Conductor paper: Qwen2.5-7B
TRINITY paper:   Qwen3-0.6B
```

Recommended Colab model:

```text
unsloth/Qwen3-8B-unsloth-bnb-4bit
```

Backup:

```text
unsloth/Qwen3-8B
```

URLs:

```text
https://huggingface.co/unsloth/Qwen3-8B-unsloth-bnb-4bit
https://huggingface.co/unsloth/Qwen3-8B
https://huggingface.co/Qwen/Qwen3-8B
```

If Colab memory is not enough, fallback coordinator choices:

```text
unsloth/Qwen3-4B-Base
unsloth/Qwen2.5-7B
```

For the first full version, use:

```python
COORDINATOR_MODEL = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
```

If you get A100/H100/RTX 6000 Ada and memory is stable:

```python
MAX_SEQ_LENGTH = 4096
LORA_RANK = 32
```

---

## 3. Recommended Colab Pro+ hardware policy

Use Colab Pro+ first.

Proceed with main 8B training only when Colab gives one of:

```text
A100 40GB/80GB
H100
RTX 6000 Ada 48GB
```

Use these for lighter work:

```text
L4: okay for smaller 4-bit tests and SFT smoke tests
V100: possible, but not ideal for 8B GRPO
T4: use for data prep, parser/reward debugging, very small smoke tests only
```

Save frequently to Google Drive and/or Hugging Face because Colab local disk is ephemeral.

Recommended persistent locations:

```text
/content/drive/MyDrive/mini_conductor/
/content/drive/MyDrive/mini_conductor/checkpoints/
/content/drive/MyDrive/mini_conductor/cache/
/content/drive/MyDrive/mini_conductor/evals/
```

---

## 4. Lambda.ai fallback policy

Use Lambda only if Colab Pro+ cannot provide a usable GPU or if sessions repeatedly disconnect.

Preferred Lambda fallback for ~$50:

```text
1x A100 40GB SXM4 at $1.99/hr -> about 25 hours
```

Alternative:

```text
1x H100 PCIe at $3.29/hr -> about 15 hours
```

Avoid using the 2x H100 option for development because ~$50 buys only about 6 hours.

Use 2x H100 only for a fully scripted final run.

---

## 5. OpenRouter worker registry

Use your `OPENROUTER_API_KEY` in Colab secrets or an environment variable.

Do not hardcode your key in the notebook.

```python
import os
from getpass import getpass

if "OPENROUTER_API_KEY" not in os.environ:
    os.environ["OPENROUTER_API_KEY"] = getpass("OpenRouter API key: ")
```

Worker registry:

```python
WORKERS = [
    {
        "id": 0,
        "name": "qwen/qwen3.7-plus",
        "provider": "openrouter",
        "description": "Low-cost general reasoning worker. Very long context. Good default first-pass solver and low-cost verifier.",
        "strengths": ["general reasoning", "math", "long context", "low cost", "structured outputs"],
        "cost_tier": "low",
        "input_cost_per_million": 0.32,
        "output_cost_per_million": 1.28,
    },
    {
        "id": 1,
        "name": "minimax/minimax-m3",
        "provider": "openrouter",
        "description": "Cheapest requested long-context multimodal worker. Good broad worker, summarizer, and inexpensive verifier.",
        "strengths": ["long context", "lowest cost", "general reasoning", "multimodal", "summarization"],
        "cost_tier": "lowest",
        "input_cost_per_million": 0.30,
        "output_cost_per_million": 1.20,
    },
    {
        "id": 2,
        "name": "moonshotai/kimi-k2.7-code",
        "provider": "openrouter",
        "description": "Coding-focused model. Use for code generation, debugging, software architecture, and implementation tasks.",
        "strengths": ["coding", "debugging", "software engineering", "implementation", "long context"],
        "cost_tier": "medium",
        "input_cost_per_million": 0.74,
        "output_cost_per_million": 3.50,
    },
    {
        "id": 3,
        "name": "z-ai/glm-5.2",
        "provider": "openrouter",
        "description": "Large-scale reasoning and long-horizon agent workflow model. Use for hard planning, hard verification, and complex project-level reasoning.",
        "strengths": ["hard reasoning", "agentic workflows", "planning", "verification", "long context"],
        "cost_tier": "medium_high",
        "input_cost_per_million": 0.95,
        "output_cost_per_million": 3.00,
    },
]
```

Suggested role defaults:

```text
qwen/qwen3.7-plus:
  cheap general thinker/worker/verifier

minimax/minimax-m3:
  cheap long-context worker, summarizer, low-cost verifier

moonshotai/kimi-k2.7-code:
  preferred coding worker

z-ai/glm-5.2:
  hard planner, hard verifier, complex multi-step reasoning
```

---

## 6. Cost-aware routing policy

### Exact OpenRouter IN/OUT prices to use

The plan should treat OpenRouter worker cost as **input-token cost + output-token cost**, using the following requested IN/OUT prices per 1M tokens:

| Worker model | IN $/1M tokens | OUT $/1M tokens | Cost notes |
|---|---:|---:|---|
| `minimax/minimax-m3` | `$0.30` | `$1.20` | Cheapest requested worker; good default for cheap long-context work and verification. |
| `qwen/qwen3.7-plus` | `$0.32` | `$1.28` | Nearly as cheap as MiniMax; good default general reasoning/math worker. |
| `z-ai/glm-5.2` | `$0.95` | `$3.00` | More expensive; reserve for hard planning, hard reasoning, and important verification. |
| `moonshotai/kimi-k2.7-code` | `$0.74` | `$3.50` | Coding specialist; expensive output, so cap output tokens and use mainly for code-heavy subtasks. |

Because output tokens are 3.75x-4.73x more expensive than input tokens for these models, the notebook should control **max output tokens** aggressively during GRPO. Prefer short worker outputs during training, e.g. `max_tokens=512-1024`, and allow longer outputs only during final evaluation/demo.

Cost formula:

```python
cost_usd = (input_tokens / 1_000_000) * input_cost_per_million \
         + (output_tokens / 1_000_000) * output_cost_per_million
```

Example per-call costs:

| Worker model | 1k IN + 1k OUT | 2k IN + 2k OUT | 5k IN + 2k OUT |
|---|---:|---:|---:|
| `minimax/minimax-m3` | `$0.00150` | `$0.00300` | `$0.00390` |
| `qwen/qwen3.7-plus` | `$0.00160` | `$0.00320` | `$0.00416` |
| `z-ai/glm-5.2` | `$0.00395` | `$0.00790` | `$0.01075` |
| `moonshotai/kimi-k2.7-code` | `$0.00424` | `$0.00848` | `$0.01070` |

Approximate workflow examples:

```text
Cheap 2-step workflow:
  MiniMax worker 2k IN / 1k OUT + Qwen verifier 2k IN / 1k OUT
  ≈ $0.00180 + $0.00192 = $0.00372

Cheap 3-step workflow:
  MiniMax thinker 2k/1k + Qwen worker 2k/1k + MiniMax verifier 3k/1k
  ≈ $0.00180 + $0.00192 + $0.00210 = $0.00582

Hard coding 3-step workflow:
  GLM thinker 2k/1k + Kimi code worker 3k/2k + GLM verifier 4k/1k
  ≈ $0.00490 + $0.00922 + $0.00680 = $0.02092
```

Training implication: a GRPO run with `100 prompts * 4 generations * 2 worker calls` could require about 800 worker calls. At `$0.003-$0.008` per cheap/medium call, this can be roughly `$2.40-$6.40` before retries, cache misses, and longer outputs. Hard Kimi/GLM-heavy workflows can push this higher, so the GRPO reward and prompt should strongly prefer MiniMax/Qwen unless the task needs Kimi/GLM.

Tell the Conductor:

```text
Prefer minimax/minimax-m3 and qwen/qwen3.7-plus for easy or medium tasks.
Prefer minimax/minimax-m3 when the task is long-context but not highly specialized, because it is the cheapest requested worker.
Prefer qwen/qwen3.7-plus for low-cost general reasoning and math.
Use moonshotai/kimi-k2.7-code for coding-heavy tasks, but cap output length because its output tokens cost $3.50/M.
Use z-ai/glm-5.2 when the task is hard, long-horizon, or needs strong planning/verification.
Use at most 3 steps unless the task is difficult.
Use 1 step for simple questions.
Use verifier steps for math, coding, high-stakes factual reasoning, or when prior steps disagree.
Avoid unnecessary GLM/Kimi calls during training unless the task type justifies them.
```

Approximate worker call cost examples:

For ~1k input + 1k output tokens:

```text
minimax/minimax-m3: ~$0.0015
qwen/qwen3.7-plus: ~$0.0016
z-ai/glm-5.2: ~$0.0040
moonshotai/kimi-k2.7-code: ~$0.0042
```

For ~2k input + 2k output tokens:

```text
minimax/minimax-m3: ~$0.0030
qwen/qwen3.7-plus: ~$0.0032
z-ai/glm-5.2: ~$0.0079
moonshotai/kimi-k2.7-code: ~$0.0085
```

These are not GPU/Colab costs. They are OpenRouter API costs and are separate from Colab Pro+.

---

## 7. Notebook structure

Create a Colab notebook named:

```text
Mini_Conductor_OpenRouter_Unsloth.ipynb
```

Notebook sections:

```text
1. Runtime and GPU check
2. Mount Google Drive
3. Install dependencies
4. Configure secrets / OpenRouter API key
5. Load coordinator model with Unsloth
6. Define worker registry
7. Define JSON workflow schema
8. Define Conductor system prompt
9. Define OpenRouter client and cached worker calls
10. Define workflow parser and validator
11. Define workflow executor
12. Build SFT warm-start dataset
13. SFT train the coordinator
14. Define graders
15. Define GRPO reward functions
16. Run GRPO-lite
17. Evaluate against baselines
18. Save LoRA / merged model / logs
19. Demo interactive Conductor call
20. Optional recursion mode
```

---

## 8. Dependency installation

Use a cell similar to:

```python
!pip install -q unsloth vllm trl datasets accelerate bitsandbytes peft transformers
!pip install -q openai jsonschema diskcache sympy pandas numpy tqdm tenacity
```

If `vllm` causes Colab dependency conflicts, skip vLLM for the first version and use standard `model.generate` for the local coordinator.

---

## 9. OpenRouter client

```python
from openai import OpenAI
import os

openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

Basic worker call:

```python
def call_openrouter_worker(model_name, messages, temperature=0.2, max_tokens=2048):
    response = openrouter.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
```

---

## 10. Caching requirement

Caching is mandatory because GRPO can call workers many times.

Use persistent Google Drive cache:

```python
import diskcache

CACHE_DIR = "/content/drive/MyDrive/mini_conductor/cache/openrouter_cache"
cache = diskcache.Cache(CACHE_DIR)
```

Cached call:

```python
import hashlib, json

def stable_hash(obj):
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cached_worker_call(worker, messages, temperature=0.2, max_tokens=2048):
    key = stable_hash({
        "worker": worker["name"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    })
    if key in cache:
        return cache[key]
    out = call_openrouter_worker(worker["name"], messages, temperature, max_tokens)
    cache[key] = out
    return out
```

---

## 11. JSON workflow schema

Use strict JSON, not Python list text.

```python
COORDINATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "model_id": {"type": "integer"},
                    "role": {"type": "string", "enum": ["thinker", "worker", "verifier"]},
                    "subtask": {"type": "string"},
                    "access": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "string", "enum": ["all"]}
                            ]
                        }
                    }
                },
                "required": ["model_id", "role", "subtask", "access"]
            }
        }
    },
    "required": ["steps"]
}
```

---

## 12. Conductor system prompt

Use this as the base prompt:

```text
You are a Conductor model. Your job is not to solve the user's problem directly.
Your job is to design a short workflow for worker language models.

You are given:
1. a user question
2. a list of available worker models with ids, strengths, and cost tiers

Return only valid JSON matching this schema:
{
  "steps": [
    {
      "model_id": 0,
      "role": "thinker|worker|verifier",
      "subtask": "natural language instruction for that worker",
      "access": []
    }
  ]
}

Roles:
- thinker: decomposes the problem, plans, identifies pitfalls, or critiques.
- worker: performs concrete solving, coding, calculation, or drafting.
- verifier: checks correctness, edge cases, and final formatting.

Rules:
- Use at most 5 steps.
- Use fewer steps for easy questions.
- Prefer cheaper workers for simple tasks.
- Use the coding worker for coding-heavy tasks.
- Use the strongest reasoning worker for hard planning or final verification.
- The final step should produce the final answer.
- The access field controls prior context: [] means no prior worker outputs, ["all"] means all previous outputs, [0, 2] means only selected prior steps.
- Return JSON only. No markdown. No explanations outside JSON.
```

---

## 13. Workflow executor

Role prompts:

```python
ROLE_PROMPTS = {
    "thinker": "You are a planning agent. Decompose the task, identify pitfalls, and propose a strategy. Do not over-focus on final formatting unless asked.",
    "worker": "You are a solving agent. Use the provided context and assigned subtask to make concrete progress toward the final answer.",
    "verifier": "You are a verifier. Check correctness, edge cases, and formatting. If correct, produce the final answer. If incorrect, explain the correction and produce the corrected final answer.",
}
```

Executor:

```python
def select_history(history, access):
    if "all" in access:
        return history
    selected = []
    for idx in access:
        if isinstance(idx, int) and 0 <= idx < len(history):
            selected.append(history[idx])
    return selected


def execute_workflow(question, plan, workers, max_worker_tokens=2048):
    history = []
    for i, step in enumerate(plan["steps"]):
        worker = workers[step["model_id"]]
        role = step["role"]
        visible = select_history(history, step.get("access", []))
        context = "\n\n".join(
            f"Step {h['step']} | {h['worker_name']} | {h['role']}\nSubtask: {h['subtask']}\nOutput:\n{h['output']}"
            for h in visible
        )
        messages = [
            {"role": "system", "content": ROLE_PROMPTS[role]},
            {"role": "user", "content": f"Original question:\n{question}\n\nVisible prior work:\n{context}\n\nAssigned subtask:\n{step['subtask']}"},
        ]
        output = cached_worker_call(worker, messages, temperature=0.2, max_tokens=max_worker_tokens)
        history.append({
            "step": i,
            "model_id": step["model_id"],
            "worker_name": worker["name"],
            "role": role,
            "subtask": step["subtask"],
            "output": output,
        })
    return history[-1]["output"] if history else "", history
```

---

## 14. SFT warm-start plan

Start with SFT before GRPO.

Purpose:

```text
Teach valid JSON format.
Teach reasonable model/role/subtask/access patterns.
Reduce invalid generations before expensive worker-executing GRPO.
```

Create 1k-5k synthetic workflow examples.

Dataset sources for questions:

```text
HuggingFaceH4/MATH-500
cais/mmlu
openbmb/RLPR-Train-Dataset
livecodebench/code_generation_lite
```

For the first run, use a smaller mix:

```text
200 MATH/MATH500-style
200 MMLU
200 RLPR/WebInstruct-style
100 coding
```

Generate target workflow heuristically by task type.

Examples:

### Easy/general question

```json
{
  "steps": [
    {
      "model_id": 0,
      "role": "worker",
      "subtask": "Answer the question directly and concisely.",
      "access": []
    }
  ]
}
```

### Math/reasoning

```json
{
  "steps": [
    {
      "model_id": 0,
      "role": "thinker",
      "subtask": "Develop a solution plan and identify pitfalls.",
      "access": []
    },
    {
      "model_id": 1,
      "role": "worker",
      "subtask": "Solve the problem using the plan.",
      "access": ["all"]
    },
    {
      "model_id": 3,
      "role": "verifier",
      "subtask": "Verify the solution and provide the final answer.",
      "access": ["all"]
    }
  ]
}
```

### Coding

```json
{
  "steps": [
    {
      "model_id": 3,
      "role": "thinker",
      "subtask": "Design the algorithm and identify edge cases.",
      "access": []
    },
    {
      "model_id": 2,
      "role": "worker",
      "subtask": "Implement the solution and explain key complexity considerations.",
      "access": ["all"]
    },
    {
      "model_id": 2,
      "role": "verifier",
      "subtask": "Review the implementation for bugs, edge cases, and final formatting.",
      "access": ["all"]
    }
  ]
}
```

Important: during SFT, randomize worker order and model IDs in the prompt, then adjust target IDs accordingly. This reduces overfitting to fixed IDs.

---

## 15. SFT training settings

Conservative Colab settings:

```python
max_seq_length = 2048
lora_rank = 16
per_device_train_batch_size = 1 or 2
gradient_accumulation_steps = 4
max_steps = 100 to 300
learning_rate = 2e-4
optim = "adamw_8bit"
```

Expected target:

```text
>95% valid JSON on held-out prompts
valid model IDs
valid role strings
reasonable access lists
```

If this fails, do not start GRPO yet. Increase SFT data quality and train longer.

---

## 16. GRPO-lite plan

After SFT works, run a small GRPO phase.

Purpose:

```text
Teach empirical routing behavior against the actual OpenRouter workers.
Teach cost-aware model selection.
Teach when verification helps.
```

Start small:

```text
100-300 train questions
num_generations = 4
max_steps = 50-150
max_prompt_length = 1024
max_completion_length = 1024
worker max_tokens = 1024 or 2048
```

Recommended first GRPO config:

```python
GRPO_NUM_GENERATIONS = 4
GRPO_MAX_STEPS = 75
GRPO_MAX_PROMPT_LENGTH = 1024
GRPO_MAX_COMPLETION_LENGTH = 1024
WORKER_MAX_TOKENS = 1024
```

Scale only after successful smoke test.

---

## 17. GRPO reward functions

Use multiple rewards:

```text
format_reward
schema_reward
role_workflow_reward
cost_reward
correctness_reward
```

### Format/schema reward

Reward valid JSON and valid schema.

```text
+2 for parseable JSON
+2 for schema-valid workflow
-2 for invalid JSON
-2 for invalid schema
```

### Role workflow reward

Reward reasonable role patterns:

```text
+0.5 if at least one worker step exists
+0.5 if hard math/coding includes verifier
+0.5 if coding tasks use Kimi as worker or verifier
+0.5 if hard reasoning uses GLM as thinker/verifier or Qwen/MiniMax as lower-cost first pass
```

Keep this reward small so it does not overpower correctness.

### Cost reward

Use estimated cost:

```python
def estimate_plan_cost(plan, workers, assumed_input_tokens=1500, assumed_output_tokens=1000):
    total = 0.0
    for step in plan["steps"]:
        w = workers[step["model_id"]]
        total += (assumed_input_tokens / 1_000_000) * w["input_cost_per_million"]
        total += (assumed_output_tokens / 1_000_000) * w["output_cost_per_million"]
    return total
```

For actual accounting, use OpenRouter response usage fields when available instead of assumptions:

```python
def estimate_call_cost_from_usage(worker, usage):
    input_tokens = usage.get("prompt_tokens", 0) or 0
    output_tokens = usage.get("completion_tokens", 0) or 0
    return (
        (input_tokens / 1_000_000) * worker["input_cost_per_million"]
        + (output_tokens / 1_000_000) * worker["output_cost_per_million"]
    )
```

The cost reward should reflect the exact IN/OUT table above and should penalize output-heavy workflows more strongly because OUT tokens dominate cost.

Reward example:

```text
cost_reward = -min(1.0, estimated_cost / 0.02)
```

### Correctness reward

Actually execute the workflow and grade the final answer.

Use task-specific graders:

```text
MMLU: exact final letter match
MATH: normalized exact/sympy equivalence where possible
Coding: unit tests if available; otherwise hold coding GRPO until executor is reliable
RLPR/general: exact labels where available or LLM judge for small eval only
```

Start GRPO primarily on MMLU and math because grading is easier. Add coding once the code grader is ready.

---

## 18. OpenRouter budget control

Set a hard budget for the notebook run.

Example:

```python
OPENROUTER_BUDGET_USD = 10.00
```

Track estimated spend:

```python
openrouter_spend_estimate = 0.0
```

Before every uncached worker call, estimate the maximum possible spend from the requested `max_tokens` and stop before exceeding the cap:

```python
def max_possible_call_cost(worker, estimated_input_tokens, max_output_tokens):
    return (
        (estimated_input_tokens / 1_000_000) * worker["input_cost_per_million"]
        + (max_output_tokens / 1_000_000) * worker["output_cost_per_million"]
    )

projected = openrouter_spend_estimate + max_possible_call_cost(
    worker,
    estimated_input_tokens=estimated_prompt_tokens,
    max_output_tokens=max_tokens,
)
if projected > OPENROUTER_BUDGET_USD:
    raise RuntimeError("OpenRouter budget would be exceeded")
```

Start with a very small API budget for smoke tests:

```text
$1-$3 for smoke tests
$5-$10 for first GRPO-lite run
```

Because the selected models are relatively inexpensive, meaningful small experiments should be possible with careful max token limits and caching.

---

## 19. Evaluation plan

Evaluate against baselines.

Baselines:

```text
1. Cheapest single worker: minimax/minimax-m3
2. Low-cost single worker: qwen/qwen3.7-plus
3. Coding worker only: moonshotai/kimi-k2.7-code
4. Strong reasoning/planning worker only: z-ai/glm-5.2
5. Fixed workflow: cheap thinker -> task-specific worker -> verifier
6. Random router
7. Trained Mini-Conductor
```

Metrics:

```text
accuracy/pass rate
valid JSON rate
average worker calls per question
estimated OpenRouter cost per question
token usage
role distribution
model selection distribution
latency
cache hit rate
```

Initial eval size:

```text
50 questions total
- 20 math
- 20 MMLU/general MCQ
- 10 coding or general reasoning
```

Larger eval after success:

```text
200 questions total
```

---

## 20. Recursive mode

After the base workflow works, add optional recursive refinement.

Prompt the Conductor after the first workflow execution:

```text
Here is the final response from the previous workflow:
{final_response}

If it is already correct, return {"steps": []}.
If it needs improvement, return a new workflow of up to 3 steps to verify or revise it.
```

Do not include recursive mode in the first GRPO run. Add it later after base execution is stable.

---

## 21. Save strategy

Save often.

After SFT:

```text
/content/drive/MyDrive/mini_conductor/checkpoints/sft_lora/
```

After GRPO:

```text
/content/drive/MyDrive/mini_conductor/checkpoints/grpo_lora/
```

Also save:

```text
worker cache
training config JSON
eval results CSV/JSONL
sample traces
model card notes
```

Optionally push LoRA to Hugging Face.

---

## 22. Colab run schedule

### Phase A: T4/L4/V100 acceptable

```text
- Build notebook skeleton
- Load datasets
- Build SFT synthetic workflow examples
- Test parser/schema/reward functions
- Test OpenRouter call with one prompt
- Run tiny 10-example SFT smoke test
```

### Phase B: A100/H100/RTX 6000 Ada preferred

```text
- Load Qwen3-8B 4-bit coordinator
- Run SFT warm-start
- Validate JSON
- Run small GRPO smoke test with 10-20 prompts
- Run GRPO-lite
- Evaluate
- Save adapter
```

### Phase C: Optional Lambda fallback

Use Lambda only if:

```text
- Colab cannot allocate A100/H100/RTX 6000 Ada for main run
- Colab disconnects repeatedly
- dependency/runtime issues waste too much time
```

Preferred Lambda fallback:

```text
1x A100 40GB SXM4, $1.99/hr, about 25 hours for $50
```

---

## 23. Expected outcomes

A realistic Colab Pro+ first version should produce:

```text
- working notebook
- Qwen3-8B LoRA coordinator
- valid JSON workflow generation
- OpenRouter harness for specified worker models
- cached worker calls
- SFT warm-start
- small GRPO-lite run
- small eval report
- saved LoRA adapter
```

Do not expect paper-level Conductor performance from the first Colab run.

The paper-like Conductor used:

```text
2x H100 80GB
51,200 workflow rollouts
up to 5 worker calls per rollout
multiple high-end workers
```

The Colab version is a practical scaled-down prototype.

---

## 24. Information required / questions

Please provide or decide the following before the notebook is finalized:

1. **OpenRouter budget cap**: What maximum dollar amount should the notebook be allowed to spend on OpenRouter calls during GRPO/evaluation?
   - Suggested first value: `$5-$10`.

2. **Primary task focus**: Should the first trained Conductor optimize mainly for:
   - coding,
   - math/reasoning,
   - general QA,
   - mixed tasks?

3. **Dataset preference**: Do you want the first run to include coding datasets immediately, or start with easier-to-grade MATH/MMLU and add coding later?
   - Recommendation: start with MATH/MMLU + small RLPR, add coding after the harness works.

4. **Hugging Face token**: Will you push the resulting LoRA adapter to Hugging Face, or only save to Google Drive?

5. **Colab GPU availability**: When you start, what GPU does Colab allocate? The notebook should choose settings based on actual GPU.

6. **Strict JSON vs Python lists**: I recommend strict JSON, but the original Conductor used Python lists. Confirm JSON is acceptable.

7. **Use of Claude Code/Codex**: You previously mentioned Claude Code and Codex are installed locally and can be used for research. Since this plan is Colab-centered, should they be excluded from the notebook worker pool for now? Recommendation: exclude them from Colab training and use only OpenRouter workers.

---

## 25. Immediate next steps

1. Confirm the questions in section 24.
2. Create `Mini_Conductor_OpenRouter_Unsloth.ipynb` from the Unsloth Qwen GRPO notebook.
3. Implement worker registry and OpenRouter cached caller.
4. Implement JSON schema parser/validator.
5. Generate SFT warm-start data.
6. Run a tiny SFT smoke test.
7. Run a tiny GRPO smoke test with strict budget cap.
8. Scale only after logs and costs look sane.
