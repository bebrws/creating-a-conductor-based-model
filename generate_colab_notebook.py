import json
from pathlib import Path

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})

def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text})

md("""# Mini-Conductor OpenRouter + Unsloth Notebook\n\nCopy/paste these cells into Google Colab Pro+ and run top-to-bottom.\n\nDefault behavior is safe:\n\n- loads an 8B Qwen-family coordinator with Unsloth,\n- builds a synthetic SFT warm-start dataset,\n- runs a short SFT smoke training run,\n- validates JSON workflow generation,\n- does **not** spend OpenRouter credits unless you explicitly set `RUN_OPENROUTER_SMOKE=True`, `RUN_OPENROUTER_DEMO=True`, or `RUN_GRPO=True`.\n\nWorker models configured:\n\n- `z-ai/glm-5.2` — $0.95 / $3.00 per 1M IN/OUT\n- `moonshotai/kimi-k2.7-code` — $0.74 / $3.50 per 1M IN/OUT\n- `qwen/qwen3.7-plus` — $0.32 / $1.28 per 1M IN/OUT\n- `minimax/minimax-m3` — $0.30 / $1.20 per 1M IN/OUT\n""")

code("""# CELL 1: Install dependencies\n# Runtime -> Change runtime type -> GPU. Prefer A100/H100/RTX 6000 Ada for the 8B run.\n# Run this install cell once. It restarts the kernel after installation because Unsloth,\n# PyTorch, pandas, and NumPy use compiled extensions that cannot be reloaded safely\n# after pip changes package versions mid-session.\n\nimport os\nfrom pathlib import Path\n\n_DEPS_MARKER = Path('/tmp/mini_conductor_deps_installed')\n_CONSTRAINTS = Path('/tmp/mini_conductor_constraints.txt')\n_CONSTRAINTS.write_text(\n    'numpy==2.0.2\\n'\n    'pandas==2.2.2\\n'\n    'cuda-python==12.9.4\\n'\n    'cuda-bindings==12.9.4\\n'\n)\n\nif not _DEPS_MARKER.exists():\n    !pip install -q -U -c /tmp/mini_conductor_constraints.txt numpy pandas cuda-python cuda-bindings unsloth trl datasets accelerate bitsandbytes peft transformers openai huggingface_hub jsonschema diskcache sympy tqdm tenacity\n    _DEPS_MARKER.write_text('installed')\n    print('Dependencies installed. Restarting kernel now. After restart, rerun from Cell 2.')\n    os.kill(os.getpid(), 9)\nelse:\n    print('Dependency install already completed for this runtime. Continue from Cell 2.')\n""")

code(r'''# CELL 2: Configuration, Drive mount, and budget settings

import os
from datetime import datetime
from getpass import getpass
from pathlib import Path

# -----------------------------
# Runtime mode flags
# -----------------------------
RUN_MODEL_LOAD = True
RUN_SFT = True                 # safe short SFT smoke run by default
RUN_GRPO = False               # turn on only after Cell 8 reports 3/3 valid local plans
RUN_OPENROUTER_SMOKE = False   # one cheap API call if True
RUN_OPENROUTER_DEMO = False    # executes a workflow with OpenRouter if True
EXPORT_MERGED_16BIT = False    # optional full merged model export; uses much more disk/RAM
PUSH_TO_HF = False             # uploads adapter + harness metadata after Cell 10 if True
PROJECT_SLUG = "mini-conductor-qwen3-router-sft-grpo-32k"  # stable Drive/HF namespace for this run family
# Per-run tag so each run writes to its own subfolder and its own HF path, and never
# overwrites a previous run. Defaults to a timestamp. Pin it to a fixed string (or set
# the RUN_TAG env var) only when you intentionally want to resume/overwrite one run.
RUN_TAG = os.environ.get("RUN_TAG") or datetime.now().strftime("run-%Y%m%d-%H%M%S")
HF_REPO_ID = f"bebrws/{PROJECT_SLUG}"
HF_PRIVATE_REPO = True

# -----------------------------
# Coordinator model
# -----------------------------
COORDINATOR_MODEL = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
FALLBACK_COORDINATOR_MODEL = "unsloth/Qwen3-4B-Base"
# Qwen3-8B natively supports 32k tokens. Training examples stay mostly short, so
# VRAM/time only grow for the long-context examples mixed in below.
MAX_SEQ_LENGTH = 32768
LORA_RANK = 16

# SFT settings. 500 examples / 250 steps (~2 epochs at effective batch 4) locks in
# both the JSON format and the routing patterns; drop to 120/60 for a quick smoke run.
SFT_NUM_EXAMPLES = 500
SFT_MAX_STEPS = 250
SFT_LEARNING_RATE = 2e-4
# Long-context training mix: this fraction of full-prompt SFT examples buries the
# real question at the end of synthetic agent-harness context (file dumps, tool
# lists, logs), teaching the conductor to route on the question and ignore the
# noise — the shape of real coding-agent (Pi/Cursor) requests. Lengths are in
# characters (~4 chars/token); 96k chars ≈ 24k tokens, inside MAX_SEQ_LENGTH.
SFT_LONG_CONTEXT_FRACTION = 0.25
SFT_LONG_CONTEXT_MIN_CHARS = 8000
SFT_LONG_CONTEXT_MAX_CHARS = 96000

# GRPO-lite settings. Keep conservative until costs/logs look sane.
GRPO_NUM_EXAMPLES = 24
GRPO_NUM_GENERATIONS = 4
GRPO_PER_DEVICE_TRAIN_BATCH_SIZE = 4  # must be a multiple of GRPO_NUM_GENERATIONS
GRPO_MAX_STEPS = 20
GRPO_MAX_PROMPT_LENGTH = 768
GRPO_MAX_COMPLETION_LENGTH = 256      # plans should be compact JSON, not long prose
GRPO_WARMUP_STEPS = 2
REQUIRE_GRPO_READY = True             # require Cell 8 to pass before spending GRPO credits
GRPO_MIN_VALID_PLANS = 3              # Cell 8 has 3 smoke prompts
WORKER_MAX_TOKENS_TRAIN = 512
WORKER_MAX_TOKENS_EVAL = 1024

# OpenRouter budget. API costs are separate from Colab Pro+.
OPENROUTER_BUDGET_USD = 5.00

# -----------------------------
# Storage
# -----------------------------
try:
    from google.colab import drive  # type: ignore
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive') / PROJECT_SLUG / RUN_TAG
except Exception:
    ROOT = Path('.') / PROJECT_SLUG / RUN_TAG

CHECKPOINT_DIR = ROOT / 'checkpoints'
CACHE_DIR = ROOT / 'cache' / 'openrouter_cache'
EVAL_DIR = ROOT / 'evals'
HARNESS_DIR = ROOT / 'harness_artifacts'
for p in [ROOT, CHECKPOINT_DIR, CACHE_DIR, EVAL_DIR, HARNESS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------
# API key
# -----------------------------
if (RUN_OPENROUTER_SMOKE or RUN_OPENROUTER_DEMO or RUN_GRPO) and not os.getenv('OPENROUTER_API_KEY'):
    os.environ['OPENROUTER_API_KEY'] = getpass('OpenRouter API key: ')
if PUSH_TO_HF and not os.getenv('HF_TOKEN'):
    os.environ['HF_TOKEN'] = getpass('Hugging Face write token: ')

print('ROOT:', ROOT)
print('PROJECT_SLUG:', PROJECT_SLUG, '| RUN_TAG:', RUN_TAG)
print('RUN_MODEL_LOAD:', RUN_MODEL_LOAD, 'RUN_SFT:', RUN_SFT, 'RUN_GRPO:', RUN_GRPO)
print('OpenRouter calls enabled:', RUN_OPENROUTER_SMOKE or RUN_OPENROUTER_DEMO or RUN_GRPO)
''')

code(r'''# CELL 3: Imports, worker registry, schema, prompts, cost helpers, and parser

import json
import re
import time
import math
import random
import hashlib
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from jsonschema import Draft202012Validator

random.seed(3407)
np.random.seed(3407)

# -----------------------------
# OpenRouter worker registry with exact IN/OUT prices per 1M tokens
# -----------------------------
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

PRICE_TABLE = pd.DataFrame([
    {
        "model": w["name"],
        "IN $/1M": w["input_cost_per_million"],
        "OUT $/1M": w["output_cost_per_million"],
        "cost_tier": w["cost_tier"],
    }
    for w in WORKERS
])
display(PRICE_TABLE)

# -----------------------------
# Workflow schema
# -----------------------------
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
                    "subtask": {"type": "string", "minLength": 3},
                    "access": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "string", "enum": ["all"]},
                            ]
                        },
                    },
                },
                "required": ["model_id", "role", "subtask", "access"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}
SCHEMA_VALIDATOR = Draft202012Validator(COORDINATOR_SCHEMA)

# -----------------------------
# Prompts
# -----------------------------
CONDUCTOR_SYSTEM_PROMPT = """You are a Conductor model. Your job is not to solve the user's problem directly.
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

Cost policy:
- minimax/minimax-m3 costs $0.30 IN / $1.20 OUT per 1M tokens. It is the cheapest requested worker.
- qwen/qwen3.7-plus costs $0.32 IN / $1.28 OUT per 1M tokens. It is a low-cost general reasoning/math worker.
- moonshotai/kimi-k2.7-code costs $0.74 IN / $3.50 OUT per 1M tokens. Use it mainly for code-heavy subtasks.
- z-ai/glm-5.2 costs $0.95 IN / $3.00 OUT per 1M tokens. Use it for hard planning, hard reasoning, and important verification.
- Output tokens are expensive, so avoid unnecessary long outputs and unnecessary expensive workers.

Rules:
- Use at most 5 steps.
- Use fewer steps for easy questions.
- Prefer MiniMax and Qwen for easy/medium tasks.
- Use Kimi for coding-heavy tasks.
- Use GLM for hard planning, hard long-horizon reasoning, or important verification.
- Use 1 step for simple questions.
- Use verifier steps for math, coding, high-stakes factual reasoning, or when prior steps disagree.
- The final step should produce the final answer.
- The access field controls prior context: [] means no prior worker outputs, ["all"] means all previous outputs, [0, 2] means only selected prior steps.
- Return JSON only. No markdown. No explanations outside JSON.
- Do not write chain-of-thought, reasoning prose, or <think> tags.
"""

ROLE_PROMPTS = {
    "thinker": "You are a planning agent. Decompose the task, identify pitfalls, and propose a strategy. Be concise.",
    "worker": "You are a solving agent. Use the provided context and assigned subtask to make concrete progress toward the final answer.",
    "verifier": "You are a verifier. Check correctness, edge cases, and formatting. If correct, produce the final answer. If incorrect, explain the correction and produce the corrected final answer.",
}

# -----------------------------
# Cost helpers
# -----------------------------
def estimate_call_cost(worker: Dict[str, Any], input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * worker["input_cost_per_million"]
        + (output_tokens / 1_000_000) * worker["output_cost_per_million"]
    )


def estimate_plan_cost(plan: Dict[str, Any], workers: List[Dict[str, Any]], assumed_input_tokens=1500, assumed_output_tokens=1000) -> float:
    total = 0.0
    for step in plan.get("steps", []):
        mid = step.get("model_id", -1)
        if 0 <= mid < len(workers):
            total += estimate_call_cost(workers[mid], assumed_input_tokens, assumed_output_tokens)
    return total

# -----------------------------
# JSON extraction and validation
# -----------------------------
def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def strip_reasoning_blocks(text: str) -> str:
    # Qwen3 and similar hybrid-reasoning models can wrap a <think>...</think>
    # monologue around (or before) the JSON. Prefer JSON outside reasoning blocks,
    # but parse_and_validate_plan can still fall back to the original text if a
    # valid plan only appears inside a reasoning block.
    text = re.sub(r"<think\b[^>]*>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think\b[^>]*>.*$", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think\b[^>]*>", " ", text, flags=re.IGNORECASE)
    return text


def iter_json_objects(text: str):
    # Scan cleaned text first, then the original fenced-stripped text as a fallback.
    # raw_decode stops at the end of the object, so trailing prose is ignored and
    # braces in surrounding text cannot corrupt extraction. Trying all candidates
    # lets parse_and_validate_plan skip non-plan dicts and return the first valid
    # workflow object.
    base = strip_code_fences(text)
    variants = [strip_reasoning_blocks(base), base]
    decoder = json.JSONDecoder()
    seen = set()
    for candidate_text in variants:
        for match in re.finditer(r"\{", candidate_text):
            try:
                obj, _ = decoder.raw_decode(candidate_text[match.start():])
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            key = json.dumps(obj, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            yield obj


def extract_json_object(text: str) -> Dict[str, Any]:
    first = None
    for obj in iter_json_objects(text):
        if first is None:
            first = obj
        if "steps" in obj:
            return obj
    if first is not None:
        return first
    raise ValueError("No JSON object found")


def validate_plan(plan: Dict[str, Any], workers: List[Dict[str, Any]], allow_empty: bool = False) -> Dict[str, Any]:
    errors = sorted(SCHEMA_VALIDATOR.iter_errors(plan), key=lambda e: e.path)
    if errors:
        raise ValueError("Schema errors: " + "; ".join(e.message for e in errors[:5]))
    steps = plan.get("steps", [])
    if not allow_empty and len(steps) == 0:
        raise ValueError("Plan has no steps")
    for i, step in enumerate(steps):
        mid = step["model_id"]
        if not (0 <= mid < len(workers)):
            raise ValueError(f"Invalid model_id {mid}")
        access = step.get("access", [])
        if "all" in access and len(access) > 1:
            raise ValueError("access cannot mix 'all' with indices")
        for a in access:
            if isinstance(a, int) and not (0 <= a < i):
                raise ValueError(f"Invalid access index {a} at step {i}")
    return plan


def parse_and_validate_plan(text: str, workers: List[Dict[str, Any]], allow_empty: bool = False) -> Dict[str, Any]:
    last_error = None
    for obj in iter_json_objects(text):
        try:
            return validate_plan(obj, workers, allow_empty=allow_empty)
        except Exception as e:
            last_error = e
    if last_error is not None:
        raise ValueError(f"No valid JSON plan found; last candidate error: {last_error}") from last_error
    raise ValueError("No JSON object found")

# Smoke test parser/cost
_example = {"steps": [{"model_id": 1, "role": "worker", "subtask": "Answer directly.", "access": []}]}
validate_plan(_example, WORKERS)
print("Parser/cost smoke OK. Example estimated cost:", estimate_plan_cost(_example, WORKERS))
''')

code(r'''# CELL 4: OpenRouter cached caller and workflow executor

import diskcache
from tenacity import retry, stop_after_attempt, wait_exponential

cache = diskcache.Cache(str(CACHE_DIR))
openrouter_spend_estimate = 0.0

def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def get_openrouter_client():
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set. Set RUN_OPENROUTER_* or RUN_GRPO only when you are ready to spend API credits.")
    from openai import OpenAI
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def usage_to_dict(usage_obj: Any) -> Dict[str, int]:
    if usage_obj is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(usage_obj, "model_dump"):
        d = usage_obj.model_dump()
    elif isinstance(usage_obj, dict):
        d = usage_obj
    else:
        d = {k: getattr(usage_obj, k, 0) for k in ["prompt_tokens", "completion_tokens", "total_tokens"]}
    return {
        "prompt_tokens": int(d.get("prompt_tokens") or 0),
        "completion_tokens": int(d.get("completion_tokens") or 0),
        "total_tokens": int(d.get("total_tokens") or 0),
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_openrouter_worker_uncached(worker: Dict[str, Any], messages: List[Dict[str, str]], temperature=0.2, max_tokens=1024) -> Dict[str, Any]:
    global openrouter_spend_estimate

    # Conservative pre-flight budget guard. The input token estimate is approximate.
    estimated_input_tokens = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
    max_possible = estimate_call_cost(worker, estimated_input_tokens, max_tokens)
    if openrouter_spend_estimate + max_possible > OPENROUTER_BUDGET_USD:
        raise RuntimeError(
            f"OpenRouter budget would be exceeded. current=${openrouter_spend_estimate:.4f}, "
            f"max_possible=${max_possible:.4f}, cap=${OPENROUTER_BUDGET_USD:.2f}"
        )

    client = get_openrouter_client()
    response = client.chat.completions.create(
        model=worker["name"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers={
            "HTTP-Referer": "https://colab.research.google.com/",
            "X-Title": PROJECT_SLUG,
        },
    )
    content = response.choices[0].message.content or ""
    usage = usage_to_dict(getattr(response, "usage", None))
    actual_cost = estimate_call_cost(worker, usage["prompt_tokens"], usage["completion_tokens"])
    if actual_cost == 0.0:
        # Fallback estimate when provider does not return usage.
        actual_cost = estimate_call_cost(worker, estimated_input_tokens, max(1, len(content) // 4))
    openrouter_spend_estimate += actual_cost
    return {"content": content, "usage": usage, "cost_usd": actual_cost}


def cached_worker_call(worker: Dict[str, Any], messages: List[Dict[str, str]], temperature=0.2, max_tokens=1024) -> Dict[str, Any]:
    key = stable_hash({
        "worker": worker["name"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    })
    if key in cache:
        cached = cache[key]
        cached["cache_hit"] = True
        return cached
    result = call_openrouter_worker_uncached(worker, messages, temperature=temperature, max_tokens=max_tokens)
    result["cache_hit"] = False
    cache[key] = result
    return result


def select_history(history: List[Dict[str, Any]], access: List[Any]) -> List[Dict[str, Any]]:
    if "all" in access:
        return history
    selected = []
    for idx in access:
        if isinstance(idx, int) and 0 <= idx < len(history):
            selected.append(history[idx])
    return selected


def build_worker_messages(question: str, step: Dict[str, Any], visible_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context = "\n\n".join(
        f"Step {h['step']} | {h['worker_name']} | {h['role']}\nSubtask: {h['subtask']}\nOutput:\n{h['output']}"
        for h in visible_history
    )
    role = step["role"]
    return [
        {"role": "system", "content": ROLE_PROMPTS[role]},
        {"role": "user", "content": f"Original question:\n{question}\n\nVisible prior work:\n{context}\n\nAssigned subtask:\n{step['subtask']}"},
    ]


def execute_workflow(question: str, plan: Dict[str, Any], workers: List[Dict[str, Any]], max_worker_tokens=1024, dry_run=False) -> Tuple[str, List[Dict[str, Any]]]:
    plan = validate_plan(plan, workers)
    history = []
    for i, step in enumerate(plan["steps"]):
        worker = workers[step["model_id"]]
        visible = select_history(history, step.get("access", []))
        messages = build_worker_messages(question, step, visible)
        if dry_run:
            output = f"[DRY RUN] {worker['name']} as {step['role']} would do: {step['subtask']}"
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            cost_usd = 0.0
            cache_hit = False
        else:
            result = cached_worker_call(worker, messages, temperature=0.2, max_tokens=max_worker_tokens)
            output = result["content"]
            usage = result.get("usage", {})
            cost_usd = result.get("cost_usd", 0.0)
            cache_hit = result.get("cache_hit", False)
        history.append({
            "step": i,
            "model_id": step["model_id"],
            "worker_name": worker["name"],
            "role": step["role"],
            "subtask": step["subtask"],
            "access": step.get("access", []),
            "output": output,
            "usage": usage,
            "cost_usd": cost_usd,
            "cache_hit": cache_hit,
        })
    return (history[-1]["output"] if history else ""), history

# Optional OpenRouter smoke test: one cheap MiniMax call.
if RUN_OPENROUTER_SMOKE:
    smoke_messages = [{"role": "user", "content": "Reply with exactly: OpenRouter smoke OK"}]
    smoke = cached_worker_call(WORKERS[1], smoke_messages, max_tokens=16)
    print(smoke)
else:
    print("OpenRouter smoke skipped. Set RUN_OPENROUTER_SMOKE=True to test one cheap API call.")
''')

code(r'''# CELL 5: Heuristic plans, prompt builders, and SFT warm-start data

def worker_summary_for_prompt(workers: List[Dict[str, Any]]) -> str:
    compact = []
    for w in workers:
        compact.append({
            "id": w["id"],
            "name": w["name"],
            "description": w["description"],
            "strengths": w["strengths"],
            "cost_tier": w["cost_tier"],
            "input_cost_per_million": w["input_cost_per_million"],
            "output_cost_per_million": w["output_cost_per_million"],
        })
    return json.dumps(compact, indent=2)


def grpo_worker_summary_for_prompt(workers: List[Dict[str, Any]]) -> str:
    compact = [
        {
            "id": w["id"],
            "name": w["name"],
            "strengths": w["strengths"][:3],
            "in": w["input_cost_per_million"],
            "out": w["output_cost_per_million"],
        }
        for w in workers
    ]
    return json.dumps(compact, separators=(",", ":"))


def build_conductor_messages(question: str, workers: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": CONDUCTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"User question:\n{question}\n\nAvailable worker models:\n{worker_summary_for_prompt(workers)}\n\nReturn only valid JSON."},
    ]


def apply_chat_template_no_think(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    # Qwen3 (and other hybrid-reasoning models) emit a <think>...</think> monologue
    # by default. At greedy inference that burns the whole token budget before any
    # JSON is produced, which is why the readiness check saw 0 valid plans. Disable
    # thinking so the coordinator emits JSON immediately. Fall back cleanly for
    # tokenizers/templates that do not support the enable_thinking kwarg.
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def render_conductor_prompt(tokenizer, question: str, workers: List[Dict[str, Any]]) -> str:
    # The exact prompt string the model conditions on at inference time.
    return apply_chat_template_no_think(
        tokenizer, build_conductor_messages(question, workers), add_generation_prompt=True
    )


def format_sft_example(tokenizer, question: str, workers: List[Dict[str, Any]], plan: Dict[str, Any], prompt_style: str = "full") -> str:
    # Train on (inference prompt) + (target JSON) + EOS. Rendering the prompt with
    # add_generation_prompt=True makes the SFT input byte-for-byte identical to what
    # ask_conductor_local feeds the model, so there is zero train/inference skew, and
    # the trailing EOS teaches the model to stop right after the JSON object.
    # prompt_style="grpo" renders the compact Cell 9 GRPO prompt instead, so the GRPO
    # stage starts from a prompt format the SFT warm start has already taught.
    if prompt_style == "grpo":
        prompt = apply_chat_template_no_think(
            tokenizer, build_grpo_conductor_messages(question, workers), add_generation_prompt=True
        )
    else:
        prompt = render_conductor_prompt(tokenizer, question, workers)
    target = json.dumps(plan, ensure_ascii=False)
    eos = tokenizer.eos_token or "<|im_end|>"
    return prompt + target + eos


GRPO_CONDUCTOR_SYSTEM_PROMPT = """Return only compact valid JSON for a worker workflow.
Schema: {"steps":[{"model_id":0,"role":"thinker|worker|verifier","subtask":"...","access":[]}]}
Use <=3 steps. Use MiniMax/Qwen for cheap general work, Kimi for coding, GLM for hard reasoning/verification.
Do not write chain-of-thought, reasoning prose, markdown, or <think> tags."""


def build_grpo_conductor_messages(question: str, workers: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": GRPO_CONDUCTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question:\n{question}\n\nWorkers:{grpo_worker_summary_for_prompt(workers)}\n\nJSON only."},
    ]


def worker_id_by_name(workers: List[Dict[str, Any]], name: str) -> int:
    for w in workers:
        if w["name"] == name:
            return int(w["id"])
    raise KeyError(name)


def classify_task(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["python", "function", "algorithm", "leetcode", "code", "implement", "bug", "class "]):
        return "coding"
    if any(k in q for k in ["solve", "calculate", "prove", "equation", "integer", "probability", "geometry", "algebra", "sum"]):
        return "math"
    if "answer choices" in q or re.search(r"\b[a-d][\).]", q):
        return "multiple_choice"
    return "general"


def heuristic_plan_for_question(question: str, workers: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Resolve IDs after any worker order randomization.
    qwen = worker_id_by_name(workers, "qwen/qwen3.7-plus")
    mini = worker_id_by_name(workers, "minimax/minimax-m3")
    kimi = worker_id_by_name(workers, "moonshotai/kimi-k2.7-code")
    glm = worker_id_by_name(workers, "z-ai/glm-5.2")
    task = classify_task(question)

    if task == "coding":
        return {"steps": [
            {"model_id": glm, "role": "thinker", "subtask": "Design the algorithm, identify edge cases, and specify the implementation approach. Be concise.", "access": []},
            {"model_id": kimi, "role": "worker", "subtask": "Implement the solution using the plan. Focus on correctness and edge cases.", "access": ["all"]},
            {"model_id": kimi, "role": "verifier", "subtask": "Review the implementation for bugs, edge cases, and final formatting. Return the corrected final answer.", "access": ["all"]},
        ]}
    if task == "math":
        return {"steps": [
            {"model_id": qwen, "role": "thinker", "subtask": "Develop a concise solution strategy and identify pitfalls.", "access": []},
            {"model_id": mini, "role": "worker", "subtask": "Solve the problem using the plan. Keep the reasoning concise.", "access": ["all"]},
            {"model_id": qwen, "role": "verifier", "subtask": "Verify the result and provide the final answer in the requested format.", "access": ["all"]},
        ]}
    if task == "multiple_choice":
        return {"steps": [
            {"model_id": qwen, "role": "worker", "subtask": "Answer the multiple-choice question and provide the final option only if possible.", "access": []},
            {"model_id": mini, "role": "verifier", "subtask": "Check the prior answer and provide the final option only.", "access": ["all"]},
        ]}
    return {"steps": [
        {"model_id": mini, "role": "worker", "subtask": "Answer the question directly and concisely.", "access": []}
    ]}


def shuffled_workers() -> List[Dict[str, Any]]:
    ws = [dict(w) for w in WORKERS]
    random.shuffle(ws)
    for i, w in enumerate(ws):
        w["id"] = i
    return ws

# -----------------------------
# Synthetic agent-harness context for long-context training
# -----------------------------
AGENT_CONTEXT_SNIPPETS = [
    'def process_batch(items, *, chunk_size=64, retries=3):\n    """Process items in chunks with retry semantics."""\n    results = []\n    for start in range(0, len(items), chunk_size):\n        chunk = items[start:start + chunk_size]\n        for attempt in range(retries):\n            try:\n                results.extend(handle_chunk(chunk))\n                break\n            except TransientError:\n                if attempt == retries - 1:\n                    raise\n    return results\n',
    '## Configuration\n\nThe service reads settings from `config.yaml` at startup. Supported keys:\n\n- `database.url`: Postgres connection string\n- `cache.ttl_seconds`: entry lifetime (default 300)\n- `workers.pool_size`: concurrent worker count\n- `logging.level`: DEBUG | INFO | WARNING\n\nRestart the service after editing configuration; hot reload is not supported.\n',
    '{\n  "name": "workspace-tools",\n  "tools": [\n    {"name": "read", "description": "Read file contents"},\n    {"name": "bash", "description": "Execute shell commands"},\n    {"name": "edit", "description": "Precise text replacement in files"},\n    {"name": "write", "description": "Create or overwrite files"}\n  ],\n  "limits": {"max_file_bytes": 1048576, "timeout_ms": 120000}\n}\n',
    "2026-07-01T14:22:07Z INFO  api.request path=/v1/orders status=200 dur_ms=41\n2026-07-01T14:22:09Z WARN  cache.miss key=orders:daily rebuilding\n2026-07-01T14:22:11Z INFO  api.request path=/v1/orders/778 status=404 dur_ms=8\n2026-07-01T14:22:15Z ERROR db.pool connection reset, retrying (attempt 1/3)\n2026-07-01T14:22:16Z INFO  db.pool connection restored\n",
    'export async function fetchWithBackoff(url: string, opts: RequestInit = {}, maxAttempts = 4): Promise<Response> {\n\tlet delay = 250;\n\tfor (let attempt = 1; attempt <= maxAttempts; attempt++) {\n\t\tconst res = await fetch(url, opts);\n\t\tif (res.ok || attempt === maxAttempts) return res;\n\t\tawait new Promise((r) => setTimeout(r, delay));\n\t\tdelay *= 2;\n\t}\n\tthrow new Error("unreachable");\n}\n',
    'class OrderRepository:\n    def __init__(self, session_factory):\n        self._session_factory = session_factory\n\n    def find_by_status(self, status, limit=100):\n        with self._session_factory() as session:\n            return (\n                session.query(Order)\n                .filter(Order.status == status)\n                .order_by(Order.created_at.desc())\n                .limit(limit)\n                .all()\n            )\n',
]


def synth_agent_context(target_chars: int, seed_idx: int) -> str:
    """Deterministic filler resembling a coding-agent workspace dump."""
    parts: List[str] = []
    total = 0
    i = 0
    while total < target_chars:
        snippet = AGENT_CONTEXT_SNIPPETS[(seed_idx + i) % len(AGENT_CONTEXT_SNIPPETS)]
        block = f"FILE workspace/module_{seed_idx % 97}_{i}.txt:\n{snippet}\n"
        parts.append(block)
        total += len(block)
        i += 1
    return "".join(parts)[:target_chars]


def wrap_question_with_agent_context(question: str, target_chars: int, seed_idx: int) -> str:
    """Bury the real question at the end of long agent-harness context, mirroring
    how coding agents (Pi/Cursor) send a large system/workspace preamble with the
    actual user request last."""
    context_budget = max(0, target_chars - len(question) - 400)
    context = synth_agent_context(context_budget, seed_idx)
    return (
        "SYSTEM:\nYou are assisting inside a coding-agent harness. The workspace context "
        "below may be long and mostly irrelevant; the actual user request is at the end.\n\n"
        + context
        + "\n\nUSER QUESTION:\n"
        + question
    )

# A small built-in fallback dataset so the notebook can run even if HF dataset loading fails.
FALLBACK_QUESTIONS = [
    {"question": "Solve: If 3x + 5 = 20, what is x?", "answer": "5", "task_type": "math"},
    {"question": "A store discounts a $80 item by 25%. What is the sale price?", "answer": "60", "task_type": "math"},
    {"question": "Answer Choices: A. Paris B. Madrid C. Rome D. Berlin. What is the capital of France?", "answer": "A", "task_type": "multiple_choice"},
    {"question": "Answer Choices: A. Oxygen B. Carbon dioxide C. Nitrogen D. Hydrogen. Which gas do plants primarily absorb for photosynthesis?", "answer": "B", "task_type": "multiple_choice"},
    {"question": "Write a Python function that returns True if a string is a palindrome, ignoring case.", "answer": "", "task_type": "coding"},
    {"question": "Implement a function to return the maximum element in a list of integers.", "answer": "", "task_type": "coding"},
    {"question": "Explain the difference between supervised and reinforcement learning in two sentences.", "answer": "", "task_type": "general"},
]


def try_load_hf_questions(limit_total: int = 100) -> List[Dict[str, str]]:
    rows = []
    try:
        from datasets import load_dataset
        # MATH-500 is easy to load and grade approximately.
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for r in ds.select(range(min(len(ds), limit_total // 3))):
            rows.append({"question": str(r.get("problem", "")), "answer": str(r.get("answer", "")), "task_type": "math"})
    except Exception as e:
        print("MATH-500 load skipped/failed:", repr(e))
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
        for r in ds.select(range(min(len(ds), limit_total // 3))):
            choices = r.get("choices", [])
            question = str(r.get("question", "")) + "\nAnswer Choices: " + " ".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
            ans = r.get("answer", "")
            if isinstance(ans, int): ans = chr(65 + ans)
            rows.append({"question": question, "answer": str(ans), "task_type": "multiple_choice"})
    except Exception as e:
        print("MMLU load skipped/failed:", repr(e))
    # Fill remaining with fallback rows.
    while len(rows) < limit_total:
        rows.append(random.choice(FALLBACK_QUESTIONS))
    return rows[:limit_total]


def make_sft_records(n: int = 120) -> List[Dict[str, Any]]:
    questions = try_load_hf_questions(max(20, n // 2))
    # Always include the built-in examples first so the SFT smoke run sees each
    # routing pattern that Cell 8 checks (math, multiple-choice, coding, general),
    # even if external HF datasets load successfully and contain no coding tasks.
    coverage = FALLBACK_QUESTIONS[:min(len(FALLBACK_QUESTIONS), n)]
    source = (questions if questions else FALLBACK_QUESTIONS)
    sampled = [random.choice(source) for _ in range(max(0, n - len(coverage)))]
    records = []
    for idx, item in enumerate((coverage + sampled)[:n]):
        ws = shuffled_workers()
        # Plan and task type are always derived from the CORE question, even when the
        # prompt below wraps it in long agent context — filler code snippets must not
        # flip classify_task() to "coding" for a math question.
        core_question = item["question"]
        plan = heuristic_plan_for_question(core_question, ws)
        validate_plan(plan, ws)
        # Every 4th record uses the compact GRPO prompt from Cell 9. GRPO otherwise
        # starts from a prompt format the policy has never seen during SFT, and the
        # first rollouts degenerate to invalid plans.
        prompt_style = "grpo" if idx % 4 == 3 else "full"
        prompt_question = core_question
        if (
            prompt_style == "full"
            and MAX_SEQ_LENGTH >= 8192
            and SFT_LONG_CONTEXT_FRACTION > 0
            and random.random() < SFT_LONG_CONTEXT_FRACTION
        ):
            target_chars = random.randint(SFT_LONG_CONTEXT_MIN_CHARS, SFT_LONG_CONTEXT_MAX_CHARS)
            prompt_question = wrap_question_with_agent_context(core_question, target_chars, idx)
        builder = build_grpo_conductor_messages if prompt_style == "grpo" else build_conductor_messages
        messages = builder(prompt_question, ws) + [{"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)}]
        records.append({
            "messages": messages,
            "prompt_style": prompt_style,
            "question": prompt_question,
            "core_question": core_question,
            "answer": item.get("answer", ""),
            "task_type": item.get("task_type", classify_task(core_question)),
            "workers": ws,
            "plan": plan,
        })
    return records

sft_records_preview = make_sft_records(5)
print("SFT preview target:")
print(json.dumps(sft_records_preview[0]["plan"], indent=2))
''')

code(r'''# CELL 6: Load local coordinator model with Unsloth

model = None
tokenizer = None

def ensure_chat_template(tokenizer):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    return tokenizer

if RUN_MODEL_LOAD:
    try:
        from unsloth import FastLanguageModel
    except RuntimeError as e:
        if "numpy was upgraded mid-session" in str(e):
            raise RuntimeError(
                "NumPy was changed after this kernel started. Restart the runtime/kernel, "
                "skip Cell 1 if it says dependencies are already installed, then rerun from Cell 2."
            ) from e
        raise
    import torch

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=COORDINATOR_MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            max_lora_rank=LORA_RANK,
        )
        print("Loaded", COORDINATOR_MODEL)
    except Exception as e:
        print("Primary model failed, trying fallback:", repr(e))
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=FALLBACK_COORDINATOR_MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            max_lora_rank=LORA_RANK,
        )
        print("Loaded fallback", FALLBACK_COORDINATOR_MODEL)

    tokenizer = ensure_chat_template(tokenizer)
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    print("Model ready for LoRA training")
else:
    print("Model load skipped")
''')

code(r'''# CELL 7: SFT warm-start training

if RUN_SFT:
    assert model is not None and tokenizer is not None, "Set RUN_MODEL_LOAD=True before SFT"
    # Re-enable training mode in case Cell 8/11 already switched the model to Unsloth
    # fast-inference mode; otherwise a rerun of this cell trains in inference mode.
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_training(model)
    except Exception:
        pass
    globals()["_CONDUCTOR_FAST_INFERENCE_READY"] = False
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    sft_records = make_sft_records(SFT_NUM_EXAMPLES)
    # Render each example as (inference prompt) + (target JSON) + EOS so the model is
    # trained on exactly the prompt ask_conductor_local will feed it, with thinking
    # disabled. This removes the train/inference skew that caused 0 valid plans.
    texts = [format_sft_example(tokenizer, r["question"], r["workers"], r["plan"], r.get("prompt_style", "full")) for r in sft_records]
    sft_dataset = Dataset.from_list([{"text": t} for t in texts])
    print(sft_dataset)
    print("SFT example text (last 600 chars):")
    print(texts[0][-600:])

    import inspect
    # TRL truncates every example to max_length, and its default is 1024 tokens. The
    # rendered conductor prompt alone (system prompt + worker registry JSON) is ~1100+
    # tokens, so with the default the assistant JSON target and EOS are cut off and the
    # model never trains on the plan at all. Budget the full MAX_SEQ_LENGTH. TRL renamed
    # max_seq_length -> max_length across versions; support both.
    _sft_params = inspect.signature(SFTConfig.__init__).parameters
    _sft_len_kwargs = (
        {"max_length": MAX_SEQ_LENGTH}
        if "max_length" in _sft_params
        else {"max_seq_length": MAX_SEQ_LENGTH}
    )
    sft_args = SFTConfig(
        dataset_text_field="text",
        **_sft_len_kwargs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=SFT_MAX_STEPS,
        learning_rate=SFT_LEARNING_RATE,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        output_dir=str(CHECKPOINT_DIR / "sft_outputs"),
    )
    # TRL renamed the tokenizer argument to processing_class. Support both so the
    # notebook works across Colab's frequently changing trl versions.
    trainer_kwargs = dict(model=model, train_dataset=sft_dataset, args=sft_args)
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    # Mask the prompt out of the loss so the ~150 target-JSON tokens carry the training
    # signal. Without this, ~90% of each example's loss is spent teaching the model to
    # reproduce the static system prompt and worker registry instead of emitting plans.
    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        print("Prompt tokens masked from loss (train_on_responses_only).")
    except Exception as e:
        print("train_on_responses_only unavailable; training on full text:", repr(e))
    trainer.train()
    sft_save_dir = CHECKPOINT_DIR / "sft_lora"
    model.save_pretrained(str(sft_save_dir))
    tokenizer.save_pretrained(str(sft_save_dir))
    print("Saved SFT LoRA to", sft_save_dir)
else:
    print("SFT skipped")
''')

code(r'''# CELL 8: Ask the local Conductor and validate JSON plans

import torch


def ask_conductor_local(question: str, workers: List[Dict[str, Any]], max_new_tokens=256, temperature=0.2) -> Tuple[Dict[str, Any], str]:
    assert model is not None and tokenizer is not None, "Coordinator model not loaded"
    if not globals().get("_CONDUCTOR_FAST_INFERENCE_READY", False):
        try:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
        except Exception:
            model.eval()
        globals()["_CONDUCTOR_FAST_INFERENCE_READY"] = True
    # Render with thinking disabled and add_generation_prompt=True so this prompt is
    # byte-for-byte the SFT training prompt (see format_sft_example in Cell 5).
    prompt = render_conductor_prompt(tokenizer, question, workers)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
    )
    if temperature > 0:
        generate_kwargs.update(temperature=temperature, top_p=0.9)
    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    try:
        plan = parse_and_validate_plan(decoded, workers)
    except Exception as e:
        # Attach the raw generation to the error; the caller's `raw` variable is never
        # assigned when this raises, so this is the only way to see what the model said.
        raise ValueError(f"{e}\n--- RAW MODEL OUTPUT ---\n{decoded}") from e
    return plan, decoded

# Validate on a few sample questions. If this fails, increase SFT examples/steps.
GRPO_READY = False
GRPO_VALID_PLAN_COUNT = 0
GRPO_VALIDATION_TOTAL = 0
if model is not None and tokenizer is not None:
    test_questions = [
        "Solve: If 12 apples are split equally among 4 people, how many apples does each person get?",
        "Write a Python function to reverse a string.",
        "Answer Choices: A. Mercury B. Venus C. Mars D. Jupiter. Which planet is known as the Red Planet?",
    ]
    if MAX_SEQ_LENGTH >= 8192:
        # Long-context smoke: the real question buried after ~20k chars of agent-harness
        # noise, mirroring coding-agent requests. The GRPO gate still only requires
        # GRPO_MIN_VALID_PLANS of these to pass.
        test_questions.append(
            wrap_question_with_agent_context("Write a Python function to reverse a string.", 20000, 0)
        )
    valid_plan_count = 0
    GRPO_VALIDATION_TOTAL = len(test_questions)
    for q in test_questions:
        q_preview = q[:160].replace("\n", " ")
        print("\nQUESTION:", q_preview + (" ..." if len(q) > 160 else ""), f"[{len(q)} chars]")
        raw = None
        try:
            plan, raw = ask_conductor_local(q, WORKERS, max_new_tokens=320, temperature=0.0)
            print(json.dumps(plan, indent=2))
            print("Estimated plan cost:", estimate_plan_cost(plan, WORKERS))
            valid_plan_count += 1
        except Exception as e:
            print("Plan generation/parse failed:", repr(e))
            if raw is not None:
                print("RAW:", raw)
    GRPO_VALID_PLAN_COUNT = valid_plan_count
    GRPO_READY = valid_plan_count >= GRPO_MIN_VALID_PLANS
    print(f"GRPO readiness: {valid_plan_count}/{len(test_questions)} valid local plans")
else:
    print("Model is not loaded; skipping local Conductor validation")
''')

code(r'''# CELL 9: Optional GRPO-lite reward functions and training
# This section can spend OpenRouter credits because correctness_reward executes worker workflows.
# Keep RUN_GRPO=False until SFT output is stable and you set a budget cap.


def normalize_text_answer(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[^a-z0-9\.\-\/ ]", "", x)
    return x.strip()


def extract_choice(text: str) -> Optional[str]:
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else None


def simple_grade(final_answer: str, expected: str, task_type: str) -> bool:
    if not expected:
        return False
    if task_type == "multiple_choice":
        return extract_choice(final_answer) == str(expected).strip().upper()[:1]
    na = normalize_text_answer(final_answer)
    ne = normalize_text_answer(expected)
    if ne and ne in na:
        return True
    # numeric fallback
    nums_a = re.findall(r"-?\d+(?:\.\d+)?", na)
    nums_e = re.findall(r"-?\d+(?:\.\d+)?", ne)
    return bool(nums_a and nums_e and nums_a[-1] == nums_e[-1])


def completion_to_text(completion: Any) -> str:
    # TRL GRPO usually passes completion as list of messages: [{"role":"assistant","content":"..."}]
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)


def format_reward_func(completions, **kwargs) -> List[float]:
    scores = []
    for c in completions:
        text = completion_to_text(c)
        try:
            parse_and_validate_plan(text, WORKERS)
            scores.append(2.0)
        except Exception:
            scores.append(-2.0)
    return scores


def cost_reward_func(completions, **kwargs) -> List[float]:
    scores = []
    for c in completions:
        text = completion_to_text(c)
        try:
            plan = parse_and_validate_plan(text, WORKERS)
            est = estimate_plan_cost(plan, WORKERS, assumed_input_tokens=1500, assumed_output_tokens=700)
            scores.append(-min(1.0, est / 0.02))
        except Exception:
            scores.append(-1.0)
    return scores


def role_reward_func(completions, task_type=None, **kwargs) -> List[float]:
    task_type = task_type or ["general"] * len(completions)
    scores = []
    for c, tt in zip(completions, task_type):
        text = completion_to_text(c)
        try:
            plan = parse_and_validate_plan(text, WORKERS)
            roles = [s["role"] for s in plan["steps"]]
            model_ids = [s["model_id"] for s in plan["steps"]]
            names = [WORKERS[i]["name"] for i in model_ids]
            score = 0.0
            if len(plan["steps"]) <= 3: score += 0.2
            if tt in ["math", "multiple_choice", "coding"] and "verifier" in roles: score += 0.3
            if tt == "coding" and "moonshotai/kimi-k2.7-code" in names: score += 0.4
            if tt != "coding" and any(n in names for n in ["minimax/minimax-m3", "qwen/qwen3.7-plus"]): score += 0.3
            scores.append(score)
        except Exception:
            scores.append(0.0)
    return scores


def correctness_reward_func(completions, question=None, answer=None, task_type=None, **kwargs) -> List[float]:
    question = question or [""] * len(completions)
    answer = answer or [""] * len(completions)
    task_type = task_type or ["general"] * len(completions)
    scores = []
    for c, q, ans, tt in zip(completions, question, answer, task_type):
        text = completion_to_text(c)
        try:
            plan = parse_and_validate_plan(text, WORKERS)
            if tt == "coding":
                # Avoid expensive/unsafe coding grading in first GRPO pass.
                scores.append(0.0)
                continue
            final, trace = execute_workflow(q, plan, WORKERS, max_worker_tokens=WORKER_MAX_TOKENS_TRAIN, dry_run=False)
            scores.append(4.0 if simple_grade(final, ans, tt) else -0.5)
        except Exception as e:
            scores.append(-1.0)
    return scores


def make_grpo_dataset(n: int = 24):
    from datasets import Dataset
    rows = []
    # Prefer easy-to-grade fallback/math/MCQ examples for the first GRPO pass.
    base = [x for x in (try_load_hf_questions(60) + FALLBACK_QUESTIONS) if x.get("task_type") in ["math", "multiple_choice"] and x.get("answer")]
    for item in base[:n]:
        # Pre-render the prompt to a plain string with thinking disabled. If the raw
        # message list were passed, GRPOTrainer would apply the chat template itself
        # with default settings (enable_thinking=True on Qwen3), and every rollout
        # would burn its GRPO_MAX_COMPLETION_LENGTH budget inside <think> before
        # emitting any JSON, so every reward would collapse to the failure value.
        prompt = apply_chat_template_no_think(
            tokenizer, build_grpo_conductor_messages(item["question"], WORKERS), add_generation_prompt=True
        )
        rows.append({
            "prompt": prompt,
            "question": item["question"],
            "answer": item.get("answer", ""),
            "task_type": item.get("task_type", classify_task(item["question"])),
        })
    return Dataset.from_list(rows)

if RUN_GRPO:
    assert model is not None and tokenizer is not None, "Set RUN_MODEL_LOAD=True before GRPO"
    assert os.getenv("OPENROUTER_API_KEY"), "Set OPENROUTER_API_KEY before GRPO"
    # Re-enable training mode in case Cell 8 switched the model to Unsloth fast
    # inference; GRPO must not train a model patched for inference.
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_training(model)
    except Exception:
        pass
    globals()["_CONDUCTOR_FAST_INFERENCE_READY"] = False
    assert GRPO_MAX_PROMPT_LENGTH + GRPO_MAX_COMPLETION_LENGTH <= MAX_SEQ_LENGTH - 64, (
        "GRPO prompt + completion budget must fit inside MAX_SEQ_LENGTH with margin. "
        f"Got prompt={GRPO_MAX_PROMPT_LENGTH}, completion={GRPO_MAX_COMPLETION_LENGTH}, max_seq={MAX_SEQ_LENGTH}."
    )
    valid = globals().get("GRPO_VALID_PLAN_COUNT", None)
    total = globals().get("GRPO_VALIDATION_TOTAL", None)
    if not globals().get("GRPO_READY", False):
        if not total:
            message = (
                "GRPO readiness check has not run in this kernel. Rerun Cell 8 after copying the latest "
                "Cell 2/5/8/9 changes, then rerun Cell 9."
            )
        else:
            action = (
                "GRPO will stop because REQUIRE_GRPO_READY=True."
                if REQUIRE_GRPO_READY
                else "GRPO will still run because REQUIRE_GRPO_READY=False."
            )
            message = (
                f"GRPO readiness check did not pass: {valid}/{total} local smoke prompts emitted valid JSON plans; "
                f"strict requirement is {GRPO_MIN_VALID_PLANS}. {action}"
            )
        if REQUIRE_GRPO_READY:
            raise RuntimeError(message + " Set REQUIRE_GRPO_READY=False only if you intentionally want GRPO to bootstrap from invalid plans.")
        print("WARNING:", message)
    from trl import GRPOConfig, GRPOTrainer

    grpo_dataset = make_grpo_dataset(GRPO_NUM_EXAMPLES)
    print(grpo_dataset)

    grpo_args = GRPOConfig(
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.001,
        warmup_steps=GRPO_WARMUP_STEPS,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=GRPO_PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_generations=GRPO_NUM_GENERATIONS,
        max_prompt_length=GRPO_MAX_PROMPT_LENGTH,
        max_completion_length=GRPO_MAX_COMPLETION_LENGTH,
        max_steps=GRPO_MAX_STEPS,
        save_steps=max(10, GRPO_MAX_STEPS // 2),
        report_to="none",
        output_dir=str(CHECKPOINT_DIR / "grpo_outputs"),
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, role_reward_func, cost_reward_func, correctness_reward_func],
        args=grpo_args,
        train_dataset=grpo_dataset,
    )
    grpo_trainer.train()
    grpo_save_dir = CHECKPOINT_DIR / "grpo_lora"
    model.save_pretrained(str(grpo_save_dir))
    tokenizer.save_pretrained(str(grpo_save_dir))
    print("Saved GRPO LoRA to", grpo_save_dir)
    print("Estimated OpenRouter spend:", openrouter_spend_estimate)
else:
    print("GRPO skipped. Set RUN_GRPO=True only after SFT/harness smoke tests pass and you set a budget.")
''')

code(r'''# CELL 10: Export coordinator artifacts for harness use

import json
import shutil
from datetime import datetime, timezone


def find_latest_adapter() -> Tuple[str, Path]:
    candidates = [
        ("grpo", CHECKPOINT_DIR / "grpo_lora"),
        ("sft", CHECKPOINT_DIR / "sft_lora"),
    ]
    for name, path in candidates:
        if path.exists() and (path / "adapter_config.json").exists():
            return name, path
    raise FileNotFoundError(
        "No LoRA adapter found. Run Cell 7 with RUN_SFT=True first, "
        "or run Cell 9 with RUN_GRPO=True after SFT."
    )


adapter_kind, adapter_dir = find_latest_adapter()
latest_adapter_file = HARNESS_DIR / "latest_adapter.txt"
manifest_path = HARNESS_DIR / "manifest.json"
loader_path = HARNESS_DIR / "conductor_harness_loader.py"

latest_adapter_file.write_text(str(adapter_dir), encoding="utf-8")

manifest = {
    "artifact_type": "mini_conductor_lora_adapter",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "project_slug": PROJECT_SLUG,
    "run_tag": RUN_TAG,
    "adapter_kind": adapter_kind,
    "adapter_dir": str(adapter_dir),
    "base_model": COORDINATOR_MODEL,
    "fallback_base_model": FALLBACK_COORDINATOR_MODEL,
    "max_seq_length": MAX_SEQ_LENGTH,
    "lora_rank": LORA_RANK,
    "workers": WORKERS,
    "coordinator_schema": COORDINATOR_SCHEMA,
    "conductor_system_prompt": CONDUCTOR_SYSTEM_PROMPT,
    "role_prompts": ROLE_PROMPTS,
    "load_mode": "unsloth_lora_adapter",
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

loader_code = r"""import json
from pathlib import Path


def load_manifest(harness_dir):
    harness_dir = Path(harness_dir)
    return json.loads((harness_dir / "manifest.json").read_text(encoding="utf-8"))


def load_conductor(harness_dir, load_in_4bit=True):
    manifest = load_manifest(harness_dir)
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=manifest["adapter_dir"],
        max_seq_length=manifest["max_seq_length"],
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer, manifest


def _worker_summary_for_prompt(workers):
    # Must match the notebook's worker_summary_for_prompt (Cell 5) so the loader
    # prompt is identical to the SFT/inference prompt the adapter was trained on.
    compact = []
    for w in workers:
        compact.append({
            "id": w["id"],
            "name": w["name"],
            "description": w["description"],
            "strengths": w["strengths"],
            "cost_tier": w["cost_tier"],
            "input_cost_per_million": w["input_cost_per_million"],
            "output_cost_per_million": w["output_cost_per_million"],
        })
    return json.dumps(compact, indent=2)


def build_conductor_messages(question, workers, manifest):
    worker_text = _worker_summary_for_prompt(workers)
    return [
        {"role": "system", "content": manifest["conductor_system_prompt"]},
        {"role": "user", "content": f"User question:\n{question}\n\nAvailable worker models:\n{worker_text}\n\nReturn only valid JSON."},
    ]
"""
loader_path.write_text(loader_code, encoding="utf-8")

print("Harness artifacts written:")
print("  adapter kind:", adapter_kind)
print("  adapter dir:", adapter_dir)
print("  latest adapter pointer:", latest_adapter_file)
print("  manifest:", manifest_path)
print("  loader:", loader_path)

if EXPORT_MERGED_16BIT:
    assert model is not None and tokenizer is not None, "Load/train the model before merged export"
    merged_dir = HARNESS_DIR / "merged_16bit"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    manifest["merged_16bit_dir"] = str(merged_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Merged 16-bit model written:", merged_dir)
else:
    print("Merged model export skipped. Set EXPORT_MERGED_16BIT=True in Cell 2 only if you need a full merged model.")

if PUSH_TO_HF:
    if not HF_REPO_ID:
        raise ValueError("Set HF_REPO_ID in Cell 2 before PUSH_TO_HF=True")
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is not set in the environment")
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=os.environ["HF_TOKEN"])
    create_repo(HF_REPO_ID, private=HF_PRIVATE_REPO, exist_ok=True, token=os.environ["HF_TOKEN"])
    # Namespace every upload under runs/<RUN_TAG>/ so reruns do not overwrite a previous
    # run's files in the same repo (mirrors the per-run Drive layout above).
    hf_run_prefix = f"runs/{RUN_TAG}"
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=HF_REPO_ID,
        path_in_repo=f"{hf_run_prefix}/adapter",
        commit_message=f"Upload {adapter_kind} Mini-Conductor LoRA adapter ({RUN_TAG})",
    )
    manifest["hf_repo_id"] = HF_REPO_ID
    manifest["hf_run_tag"] = RUN_TAG
    manifest["hf_adapter_path"] = f"{hf_run_prefix}/adapter"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    api.upload_folder(
        folder_path=str(HARNESS_DIR),
        repo_id=HF_REPO_ID,
        path_in_repo=f"{hf_run_prefix}/harness_artifacts",
        commit_message=f"Upload Mini-Conductor harness metadata ({RUN_TAG})",
    )
    if EXPORT_MERGED_16BIT and "merged_16bit_dir" in manifest:
        api.upload_folder(
            folder_path=manifest["merged_16bit_dir"],
            repo_id=HF_REPO_ID,
            path_in_repo=f"{hf_run_prefix}/merged_16bit",
            commit_message=f"Upload merged 16-bit Mini-Conductor model ({RUN_TAG})",
        )
    print("Uploaded adapter and harness artifacts to Hugging Face:", HF_REPO_ID, "under", hf_run_prefix)
else:
    print("Hugging Face upload skipped. Set PUSH_TO_HF=True and HF_REPO_ID in Cell 2 to upload.")
''')

code(r'''# CELL 11: Final demo: local plan + dry run, or real OpenRouter execution if enabled

demo_question = "Write a Python function that returns the length of the longest substring without repeating characters. Include edge cases."

print("DEMO QUESTION:", demo_question)

try:
    if model is not None and tokenizer is not None:
        demo_plan, raw = ask_conductor_local(demo_question, WORKERS, temperature=0.0)
    else:
        demo_plan = heuristic_plan_for_question(demo_question, WORKERS)
        raw = json.dumps(demo_plan)
except Exception as e:
    print("Local conductor failed; using heuristic fallback:", repr(e))
    demo_plan = heuristic_plan_for_question(demo_question, WORKERS)
    raw = json.dumps(demo_plan)

print("\nPLAN:")
print(json.dumps(demo_plan, indent=2))
print("Estimated plan cost (rough):", estimate_plan_cost(demo_plan, WORKERS, assumed_input_tokens=2000, assumed_output_tokens=1000))

final, trace = execute_workflow(
    demo_question,
    demo_plan,
    WORKERS,
    max_worker_tokens=WORKER_MAX_TOKENS_EVAL,
    dry_run=not RUN_OPENROUTER_DEMO,
)

print("\nTRACE:")
for h in trace:
    print("=" * 80)
    print(f"Step {h['step']} | {h['worker_name']} | {h['role']} | cost=${h['cost_usd']:.6f} | cache={h['cache_hit']}")
    print("Subtask:", h['subtask'])
    print("Output preview:", h['output'][:1000])

print("\nFINAL:")
print(final)
print("\nEstimated OpenRouter spend this session:", openrouter_spend_estimate)
print("\nSet RUN_OPENROUTER_DEMO=True and rerun cells 2, 4, and 11 to execute this workflow against OpenRouter.")
''')

code(r'''# CELL 12: Standalone Hugging Face upload
# Runnable on its own after any training run (only needs Cell 2 for config): uploads
# the newest adapter (GRPO if present, else SFT) plus harness metadata to HF_REPO_ID
# under runs/<RUN_TAG>/, and also mirrors it to a stable top-level latest/ path so
# downstream loaders do not need to know the run tag.

import os
from getpass import getpass

if not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = getpass("Hugging Face write token: ")

from huggingface_hub import HfApi, create_repo

_candidates = [("grpo", CHECKPOINT_DIR / "grpo_lora"), ("sft", CHECKPOINT_DIR / "sft_lora")]
adapter_kind = adapter_dir = None
for _name, _path in _candidates:
    if _path.exists() and (_path / "adapter_config.json").exists():
        adapter_kind, adapter_dir = _name, _path
        break
if adapter_dir is None:
    raise FileNotFoundError(f"No LoRA adapter found under {CHECKPOINT_DIR}. Run Cell 7 (SFT) or Cell 9 (GRPO) first.")

api = HfApi(token=os.environ["HF_TOKEN"])
create_repo(HF_REPO_ID, private=HF_PRIVATE_REPO, exist_ok=True, token=os.environ["HF_TOKEN"])

hf_run_prefix = f"runs/{RUN_TAG}"
api.upload_folder(
    folder_path=str(adapter_dir),
    repo_id=HF_REPO_ID,
    path_in_repo=f"{hf_run_prefix}/adapter",
    commit_message=f"Upload {adapter_kind} Mini-Conductor LoRA adapter ({RUN_TAG})",
)
api.upload_folder(
    folder_path=str(adapter_dir),
    repo_id=HF_REPO_ID,
    path_in_repo="latest/adapter",
    commit_message=f"Mirror {adapter_kind} adapter from {RUN_TAG} to latest/",
)
if HARNESS_DIR.exists() and any(HARNESS_DIR.iterdir()):
    api.upload_folder(
        folder_path=str(HARNESS_DIR),
        repo_id=HF_REPO_ID,
        path_in_repo=f"{hf_run_prefix}/harness_artifacts",
        commit_message=f"Upload Mini-Conductor harness metadata ({RUN_TAG})",
    )

print(f"Uploaded {adapter_kind} adapter to https://huggingface.co/{HF_REPO_ID}")
print(f"  run path:     {hf_run_prefix}/adapter")
print(f"  stable path:  latest/adapter")
print(f"  private repo: {HF_PRIVATE_REPO}")
''')

# Write ipynb
nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path("Mini_Conductor_OpenRouter_Unsloth.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Wrote Mini_Conductor_OpenRouter_Unsloth.ipynb")
