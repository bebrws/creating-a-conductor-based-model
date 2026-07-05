from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("mini_conductor.harness")


def log_preview(text: Any, limit: int | None = None) -> str:
    if limit is None:
        try:
            limit = int(os.getenv("CONDUCTOR_LOG_PREVIEW_CHARS", "500"))
        except Exception:
            limit = 500
    value = str(text or "").replace("\r", " ").strip()
    value = re.sub(r"\s+", " ", value)
    if len(value) > limit:
        return value[:limit] + f"... <truncated {len(value) - limit} chars>"
    return value


def summarize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": m.get("role"),
            "chars": len(str(m.get("content", ""))),
            "preview": log_preview(m.get("content")),
        }
        for m in messages
    ]


ROLE_PROMPT_FALLBACKS = {
    "thinker": "You are a planning agent. Decompose the task, identify pitfalls, and propose a strategy. Be concise.",
    "worker": "You are a solving agent. Use the provided context and assigned subtask to make concrete progress toward the final answer.",
    "verifier": "You are a verifier. Check correctness, edge cases, and formatting. If correct, produce the final answer. If incorrect, explain the correction and produce the corrected final answer.",
}


@dataclass
class HarnessConfig:
    harness_dir: Path
    execute: bool
    conductor_mode: str
    max_worker_tokens: int
    max_conductor_tokens: int
    temperature: float
    budget_usd: float
    cache_dir: Path | None


def stable_hash(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(harness_dir: Path) -> dict[str, Any]:
    manifest_path = harness_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return load_json(manifest_path)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def strip_reasoning_blocks(text: str) -> str:
    # Hybrid-reasoning coordinators (e.g. Qwen3) can wrap a <think>...</think>
    # monologue around or before the JSON. Prefer JSON outside reasoning blocks,
    # but parse_and_validate_plan can still fall back to the original text if a
    # valid plan only appears inside a reasoning block.
    text = re.sub(r"<think\b[^>]*>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think\b[^>]*>.*$", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think\b[^>]*>", " ", text, flags=re.IGNORECASE)
    return text


def iter_json_objects(text: str):
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


def extract_json_object(text: str) -> dict[str, Any]:
    first = None
    for obj in iter_json_objects(text):
        if first is None:
            first = obj
        if "steps" in obj:
            return obj
    if first is not None:
        return first
    raise ValueError("No JSON object found in conductor output")


def validate_plan(plan: dict[str, Any], workers: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Plan must contain a non-empty steps array")
    if len(steps) > 5:
        raise ValueError("Plan may contain at most 5 steps")

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"Step {i} must be an object")
        for key in ["model_id", "role", "subtask", "access"]:
            if key not in step:
                raise ValueError(f"Step {i} missing required key {key!r}")
        model_id = step["model_id"]
        if not isinstance(model_id, int) or not (0 <= model_id < len(workers)):
            raise ValueError(f"Step {i} has invalid model_id {model_id!r}")
        if step["role"] not in {"thinker", "worker", "verifier"}:
            raise ValueError(f"Step {i} has invalid role {step['role']!r}")
        if not isinstance(step["subtask"], str) or len(step["subtask"].strip()) < 3:
            raise ValueError(f"Step {i} has invalid subtask")
        access = step["access"]
        if not isinstance(access, list):
            raise ValueError(f"Step {i} access must be a list")
        if "all" in access and len(access) > 1:
            raise ValueError(f"Step {i} access cannot mix 'all' with indices")
        for item in access:
            if item == "all":
                continue
            if not isinstance(item, int) or not (0 <= item < i):
                raise ValueError(f"Step {i} has invalid access index {item!r}")
    return plan


def parse_and_validate_plan(text: str, workers: list[dict[str, Any]]) -> dict[str, Any]:
    last_error = None
    for obj in iter_json_objects(text):
        try:
            return validate_plan(obj, workers)
        except Exception as e:
            last_error = e
    if last_error is not None:
        raise ValueError(f"No valid JSON plan found; last candidate error: {last_error}") from last_error
    raise ValueError("No JSON object found in conductor output")


def estimate_call_cost(worker: dict[str, Any], input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000) * float(worker["input_cost_per_million"])
        + (output_tokens / 1_000_000) * float(worker["output_cost_per_million"])
    )


def usage_to_dict(usage_obj: Any) -> dict[str, int]:
    if usage_obj is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(usage_obj, "model_dump"):
        data = usage_obj.model_dump()
    elif isinstance(usage_obj, dict):
        data = usage_obj
    else:
        data = {k: getattr(usage_obj, k, 0) for k in ["prompt_tokens", "completion_tokens", "total_tokens"]}
    return {
        "prompt_tokens": int(data.get("prompt_tokens") or 0),
        "completion_tokens": int(data.get("completion_tokens") or 0),
        "total_tokens": int(data.get("total_tokens") or 0),
    }


def worker_summary_for_prompt(workers: list[dict[str, Any]]) -> str:
    # Must match the notebook's worker_summary_for_prompt (Cell 5) exactly so the
    # harness prompt is identical to the SFT/inference prompt the adapter was
    # trained on. Any drift here degrades plan quality.
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


def build_conductor_messages(question: str, workers: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": manifest["conductor_system_prompt"]},
        {
            "role": "user",
            "content": (
                f"User question:\n{question}\n\n"
                f"Available worker models:\n{worker_summary_for_prompt(workers)}\n\n"
                "Return only valid JSON."
            ),
        },
    ]


def heuristic_plan_for_question(question: str, workers: list[dict[str, Any]]) -> dict[str, Any]:
    q = question.lower()
    by_name = {w["name"]: i for i, w in enumerate(workers)}
    cheap = by_name.get("minimax/minimax-m3", 0)
    general = by_name.get("qwen/qwen3.7-plus", cheap)
    coding = by_name.get("moonshotai/kimi-k2.7-code", general)
    hard = by_name.get("z-ai/glm-5.2", general)

    is_coding = any(x in q for x in ["code", "python", "function", "debug", "bug", "refactor", "implement"])
    is_hard = any(x in q for x in ["prove", "architecture", "design", "plan", "strategy", "analyze", "complex"])
    needs_verify = is_coding or is_hard or any(x in q for x in ["math", "calculate", "correct", "verify"])

    if is_coding:
        steps = [{"model_id": coding, "role": "worker", "subtask": "Solve the coding task. Include edge cases when relevant.", "access": []}]
    elif is_hard:
        steps = [{"model_id": hard, "role": "thinker", "subtask": "Plan and solve the task carefully.", "access": []}]
    else:
        steps = [{"model_id": cheap, "role": "worker", "subtask": "Answer the user's question directly. Return only the final answer.", "access": []}]

    if needs_verify:
        verifier = hard if is_hard else general
        steps.append({"model_id": verifier, "role": "verifier", "subtask": "Check the previous answer. Return only the final answer.", "access": ["all"]})
    return {"steps": steps}


def load_local_conductor(manifest: dict[str, Any], load_in_4bit: bool = True) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=manifest["adapter_dir"],
        max_seq_length=int(manifest["max_seq_length"]),
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def ask_local_conductor(
    question: str,
    workers: list[dict[str, Any]],
    manifest: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
) -> tuple[dict[str, Any], str]:
    import torch

    model, tokenizer = load_local_conductor(manifest)
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        model.eval()
    messages = build_conductor_messages(question, workers, manifest)
    # Disable hybrid-reasoning thinking so the adapter emits JSON immediately,
    # matching how the notebook trained and validated it. Fall back for templates
    # that do not accept the enable_thinking kwarg.
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_and_validate_plan(raw, workers), raw


def select_history(history: list[dict[str, Any]], access: list[Any]) -> list[dict[str, Any]]:
    if "all" in access:
        return history
    return [history[i] for i in access if isinstance(i, int) and 0 <= i < len(history)]


def build_worker_messages(
    question: str,
    step: dict[str, Any],
    visible_history: list[dict[str, Any]],
    role_prompts: dict[str, str],
) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"Step {h['step']} | {h['worker_name']} | {h['role']}\nSubtask: {h['subtask']}\nOutput:\n{h['output']}"
        for h in visible_history
    )
    role = step["role"]
    return [
        {"role": "system", "content": role_prompts.get(role, ROLE_PROMPT_FALLBACKS[role])},
        {
            "role": "user",
            "content": f"Original question:\n{question}\n\nVisible prior work:\n{context}\n\nAssigned subtask:\n{step['subtask']}",
        },
    ]


def read_cache(cache_dir: Path | None, key: str) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    return load_json(path)


def write_cache(cache_dir: Path | None, key: str, value: dict[str, Any]) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")


def call_openrouter_worker(
    worker: dict[str, Any],
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    cache_dir: Path | None,
) -> dict[str, Any]:
    key = stable_hash({"worker": worker["name"], "messages": messages, "max_tokens": max_tokens, "temperature": temperature})
    cached = read_cache(cache_dir, key)
    if cached is not None:
        cached["cache_hit"] = True
        logger.info(
            "OpenRouter cache hit | model=%s | max_tokens=%s | temperature=%s | response_preview=%s",
            worker["name"],
            max_tokens,
            temperature,
            log_preview(cached.get("content")),
        )
        return cached

    estimated_input_tokens = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
    logger.info(
        "OpenRouter request | model=%s | max_tokens=%s | temperature=%s | estimated_input_tokens=%s | messages=%s",
        worker["name"],
        max_tokens,
        temperature,
        estimated_input_tokens,
        json.dumps(summarize_messages(messages), ensure_ascii=False),
    )

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    response = client.chat.completions.create(
        model=worker["name"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers={
            "HTTP-Referer": "https://colab.research.google.com/",
            "X-Title": "Mini-Conductor Harness",
        },
    )
    content = response.choices[0].message.content or ""
    usage = usage_to_dict(getattr(response, "usage", None))
    cost = estimate_call_cost(worker, usage["prompt_tokens"], usage["completion_tokens"])
    if cost == 0.0:
        cost = estimate_call_cost(worker, estimated_input_tokens, max(1, len(content) // 4))
    result = {"content": content, "usage": usage, "cost_usd": cost, "cache_hit": False}
    logger.info(
        "OpenRouter response | model=%s | usage=%s | cost_usd=%.6f | response_chars=%s | response_preview=%s",
        worker["name"],
        json.dumps(usage, sort_keys=True),
        cost,
        len(content),
        log_preview(content),
    )
    write_cache(cache_dir, key, result)
    return result


def execute_workflow(
    question: str,
    plan: dict[str, Any],
    manifest: dict[str, Any],
    config: HarnessConfig,
) -> tuple[str, list[dict[str, Any]], float]:
    workers = manifest["workers"]
    role_prompts = manifest.get("role_prompts") or ROLE_PROMPT_FALLBACKS
    validate_plan(plan, workers)
    history: list[dict[str, Any]] = []
    spend = 0.0

    for i, step in enumerate(plan["steps"]):
        worker = workers[step["model_id"]]
        logger.info(
            "Conductor decision | step=%s | model_id=%s | model=%s | role=%s | access=%s | subtask=%s",
            i,
            step["model_id"],
            worker["name"],
            step["role"],
            json.dumps(step.get("access", []), ensure_ascii=False),
            log_preview(step.get("subtask"), limit=800),
        )
        visible = select_history(history, step.get("access", []))
        messages = build_worker_messages(question, step, visible, role_prompts)
        if not config.execute:
            output = f"[DRY RUN] {worker['name']} as {step['role']} would do: {step['subtask']}"
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            cost_usd = 0.0
            cache_hit = False
            logger.info(
                "Dry-run worker step | step=%s | model=%s | role=%s | output_preview=%s",
                i,
                worker["name"],
                step["role"],
                log_preview(output),
            )
        else:
            estimated_input_tokens = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
            max_possible = estimate_call_cost(worker, estimated_input_tokens, config.max_worker_tokens)
            if spend + max_possible > config.budget_usd:
                raise RuntimeError(
                    f"Budget guard stopped before step {i}. current=${spend:.4f}, "
                    f"max_possible=${max_possible:.4f}, cap=${config.budget_usd:.2f}"
                )
            result = call_openrouter_worker(
                worker,
                messages,
                max_tokens=config.max_worker_tokens,
                temperature=config.temperature,
                cache_dir=config.cache_dir,
            )
            output = result["content"]
            usage = result["usage"]
            cost_usd = float(result["cost_usd"])
            cache_hit = bool(result["cache_hit"])
            spend += cost_usd
            logger.info(
                "Worker step complete | step=%s | model=%s | role=%s | cost_usd=%.6f | cache_hit=%s | output_preview=%s",
                i,
                worker["name"],
                step["role"],
                cost_usd,
                cache_hit,
                log_preview(output),
            )

        history.append(
            {
                "step": i,
                "worker_name": worker["name"],
                "role": step["role"],
                "subtask": step["subtask"],
                "access": step.get("access", []),
                "output": output,
                "usage": usage,
                "cost_usd": cost_usd,
                "cache_hit": cache_hit,
            }
        )
    return history[-1]["output"], history, spend


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Mini-Conductor LoRA adapter plus OpenRouter worker harness.")
    parser.add_argument("--harness-dir", required=True, help="Directory containing manifest.json from notebook Cell 10.")
    parser.add_argument("--question", required=True, help="User question/task to run.")
    parser.add_argument("--execute", action="store_true", help="Actually call OpenRouter workers. Omit for dry run.")
    parser.add_argument("--conductor-mode", choices=["local", "heuristic"], default="local")
    parser.add_argument("--max-worker-tokens", type=int, default=1024)
    parser.add_argument("--max-conductor-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--cache-dir", default=None, help="Optional JSON cache directory for OpenRouter worker calls.")
    parser.add_argument("--plan-json", default=None, help="Optional path to save the conductor plan JSON.")
    parser.add_argument("--trace-json", default=None, help="Optional path to save execution trace JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    config = HarnessConfig(
        harness_dir=Path(args.harness_dir),
        execute=args.execute,
        conductor_mode=args.conductor_mode,
        max_worker_tokens=args.max_worker_tokens,
        max_conductor_tokens=args.max_conductor_tokens,
        temperature=args.temperature,
        budget_usd=args.budget_usd,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    manifest = load_manifest(config.harness_dir)
    workers = manifest["workers"]

    if config.conductor_mode == "heuristic":
        plan = heuristic_plan_for_question(args.question, workers)
        raw = json.dumps(plan)
    else:
        plan, raw = ask_local_conductor(
            args.question,
            workers,
            manifest,
            max_new_tokens=config.max_conductor_tokens,
            temperature=0.0,
        )

    if args.plan_json:
        Path(args.plan_json).write_text(json.dumps(plan, indent=2), encoding="utf-8")

    final, trace, spend = execute_workflow(args.question, plan, manifest, config)

    if args.trace_json:
        Path(args.trace_json).write_text(json.dumps({"plan": plan, "trace": trace, "spend_usd": spend}, indent=2), encoding="utf-8")

    print("PLAN:")
    print(json.dumps(plan, indent=2))
    if config.conductor_mode == "local":
        print("\nRAW_CONDUCTOR_OUTPUT:")
        print(raw)
    print("\nTRACE:")
    for item in trace:
        print("=" * 80)
        print(
            f"Step {item['step']} | {item['worker_name']} | {item['role']} | "
            f"cost=${item['cost_usd']:.6f} | cache={item['cache_hit']}"
        )
        print("Subtask:", item["subtask"])
        print("Output preview:", item["output"][:1000])
    print("\nFINAL:")
    print(final)
    print(f"\nEstimated OpenRouter spend: ${spend:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
