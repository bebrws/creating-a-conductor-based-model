import argparse
import dataclasses
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .harness import (
    HarnessConfig,
    build_conductor_messages,
    execute_workflow,
    heuristic_plan_for_question,
    load_manifest,
    log_preview,
    parse_and_validate_plan,
)

logger = logging.getLogger("mini_conductor.openai_server")


DEFAULT_MODEL_NAME = "mini-conductor-qwen3-router-sft-grpo"


def cached_snapshot(repo_id: str, allow_patterns: list[str], refresh: bool = False) -> Path:
    """snapshot_download, but offline-first: if every matching file is already in the
    local HF cache, return it without any network traffic (instant restarts). Only
    hit the network when files are missing, or when refresh=True (--hf-refresh) to
    pick up newly uploaded runs."""
    from huggingface_hub import snapshot_download

    if not refresh:
        try:
            return Path(snapshot_download(repo_id, allow_patterns=allow_patterns, local_files_only=True))
        except Exception:
            logger.info("HF cache incomplete for %s; downloading from the Hub", repo_id)
    return Path(snapshot_download(repo_id, allow_patterns=allow_patterns))


def resolve_hf_conductor(hf_model_id: str, hf_run: str | None = None, refresh: bool = False) -> dict[str, Any]:
    """Download conductor artifacts (LoRA adapter + manifest) from a HF Hub repo.

    Supports the layouts produced by notebook Cells 10/12:
      runs/<run-tag>/adapter/            (+ runs/<run-tag>/harness_artifacts/manifest.json)
      latest/adapter/
      adapter/  or  adapter files at the repo root

    The multi-GB merged_16bit model is intentionally NOT downloaded here; it is
    fetched lazily only when the merged load path is actually used.
    """
    local_root = cached_snapshot(
        hf_model_id,
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer*",
            "chat_template.jinja",
            "special_tokens_map.json",
            "adapter/*",
            "latest/adapter/*",
            "runs/*/adapter/*",
            "manifest.json",
            "harness_artifacts/manifest.json",
            "latest/harness_artifacts/manifest.json",
            "runs/*/harness_artifacts/manifest.json",
        ],
        refresh=refresh,
    )

    candidates: list[Path] = []
    if hf_run:
        candidates.append(local_root / "runs" / hf_run / "adapter")
    candidates.append(local_root / "latest" / "adapter")
    runs_root = local_root / "runs"
    if runs_root.exists():
        # Run tags are timestamps, so lexicographic order is chronological.
        candidates.extend(rd / "adapter" for rd in sorted(runs_root.iterdir(), reverse=True))
    candidates.append(local_root / "adapter")
    candidates.append(local_root)

    adapter_dir = next((c for c in candidates if (c / "adapter_config.json").exists()), None)
    if adapter_dir is None:
        raise FileNotFoundError(
            f"No adapter_config.json found in Hugging Face repo {hf_model_id}"
            + (f" for run {hf_run}" if hf_run else "")
            + f". Searched: {[str(c.relative_to(local_root)) or '.' for c in candidates]}"
        )

    manifest_candidates = [
        adapter_dir.parent / "harness_artifacts" / "manifest.json",
        local_root / "harness_artifacts" / "manifest.json",
        local_root / "manifest.json",
    ]
    manifest_path = next((m for m in manifest_candidates if m.exists()), None)

    run_prefix = str(adapter_dir.parent.relative_to(local_root))
    if run_prefix == ".":
        run_prefix = ""
    return {
        "local_root": local_root,
        "adapter_dir": adapter_dir,
        "manifest_path": manifest_path,
        "run_prefix": run_prefix,
    }


def content_to_text(content: Any) -> str:
    """Convert OpenAI chat message content into plain text.

    Cursor normally sends string content, but OpenAI-compatible clients can send
    lists of multimodal parts. We keep text parts and ignore non-text payloads.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {"text", "input_text"} and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(str(item["content"]))
        return "\n".join(parts)
    return str(content)


def messages_to_question(messages: list[dict[str, Any]]) -> str:
    """Flatten Cursor/OpenAI chat messages into one task for the conductor.

    We preserve the conversation roles because Cursor may include instructions,
    files, selected code, and prior assistant context. The conductor plans over a
    clipped copy of this text, while workers receive the full text.
    """
    blocks: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").upper()
        text = content_to_text(message.get("content")).strip()
        if text:
            blocks.append(f"{role}:\n{text}")
    if not blocks:
        return ""
    return "\n\n".join(blocks)


def truncate_for_planning(text: str, max_chars: int) -> str:
    """Keep conductor prompt short while preserving full text for workers."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    return (
        text[:head]
        + f"\n\n[... {omitted} characters omitted from conductor planning prompt; workers receive the full request ...]\n\n"
        + text[-tail:]
    )


def rough_token_count(text: str) -> int:
    return max(1, len(text) // 4)


class ConductorOpenAIService:
    def __init__(
        self,
        harness_dir: Path | None,
        execute: bool,
        conductor_mode: str,
        max_worker_tokens: int,
        max_conductor_tokens: int,
        worker_temperature: float,
        budget_usd: float,
        cache_dir: Path | None,
        model_name: str,
        max_planning_chars: int | None,
        include_trace: bool,
        hf_model_id: str | None = None,
        hf_run: str | None = None,
        conductor_load: str = "auto",
        hf_refresh: bool = False,
    ) -> None:
        self.hf_model_id = hf_model_id
        self.conductor_load = conductor_load
        self._hf: dict[str, Any] | None = None
        if hf_model_id:
            self._hf = resolve_hf_conductor(hf_model_id, hf_run, refresh=hf_refresh)
            logger.info(
                "Resolved Hugging Face conductor | repo=%s | adapter_dir=%s | manifest=%s",
                hf_model_id,
                self._hf["adapter_dir"],
                self._hf["manifest_path"],
            )
            if self._hf["manifest_path"] is not None:
                self.manifest = json.loads(Path(self._hf["manifest_path"]).read_text(encoding="utf-8"))
            elif harness_dir is not None:
                self.manifest = load_manifest(harness_dir)
            else:
                raise FileNotFoundError(
                    f"Repo {hf_model_id} contains no manifest.json and no --harness-dir was given."
                )
            # The manifest's adapter_dir is a path on the training machine (e.g.
            # /content/drive/...); point it at the local snapshot instead.
            self.manifest["adapter_dir"] = str(self._hf["adapter_dir"])
            harness_dir = harness_dir or Path(self._hf["local_root"])
        else:
            assert harness_dir is not None
            self.manifest = load_manifest(harness_dir)
        self.harness_dir = harness_dir
        self.workers = self.manifest["workers"]
        self.config = HarnessConfig(
            harness_dir=harness_dir,
            execute=execute,
            conductor_mode=conductor_mode,
            max_worker_tokens=max_worker_tokens,
            max_conductor_tokens=max_conductor_tokens,
            temperature=worker_temperature,
            budget_usd=budget_usd,
            cache_dir=cache_dir,
        )
        self.model_name = model_name
        if max_planning_chars is None:
            # Fit the planning prompt to the conductor's context window: reserve room
            # for the system prompt + worker registry (~1.5k tokens) and generation,
            # then convert tokens to chars conservatively (~3 chars/token). A 2048-ctx
            # model keeps the old 8000-char floor; a 32k model gets ~91k chars.
            budget_tokens = int(self.manifest.get("max_seq_length", 2048)) - max_conductor_tokens - 2048
            max_planning_chars = max(8000, budget_tokens * 3)
            logger.info("max_planning_chars auto-sized from manifest: %s", max_planning_chars)
        self.max_planning_chars = max_planning_chars
        self.include_trace = include_trace
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _resolve_load_mode(self) -> str:
        if self.conductor_load != "auto":
            return self.conductor_load
        # Unsloth + bitsandbytes 4-bit need CUDA. On CUDA-less machines (e.g. Apple
        # Silicon) fall back to the merged 16-bit model via plain transformers.
        try:
            import torch

            if torch.cuda.is_available():
                import unsloth  # noqa: F401

                return "unsloth"
        except Exception:
            pass
        return "merged"

    def _ensure_merged_model(self) -> Path:
        """Locate (downloading if needed) the merged_16bit model directory."""
        merged_local = self.manifest.get("merged_16bit_dir")
        if merged_local and Path(merged_local).exists():
            return Path(merged_local)
        if self._hf is None or self.hf_model_id is None:
            raise RuntimeError(
                "Merged model load requested but no merged_16bit directory exists locally "
                "and no --hf-model-id was provided to download it from."
            )
        prefix = "harness_artifacts/merged_16bit"
        if self._hf["run_prefix"]:
            prefix = f"{self._hf['run_prefix']}/{prefix}"
        logger.info(
            "Resolving merged 16-bit conductor from %s (%s/*) — served from local cache when complete, ~16GB download otherwise",
            self.hf_model_id,
            prefix,
        )
        local_root = cached_snapshot(self.hf_model_id, allow_patterns=[f"{prefix}/*"], refresh=False)
        merged_dir = local_root / prefix
        if not (merged_dir / "config.json").exists():
            raise RuntimeError(
                f"Repo {self.hf_model_id} has no merged model under {prefix}/. "
                "Re-export with EXPORT_MERGED_16BIT=True (notebook Cell 10), or run this "
                "server on a CUDA machine with --conductor-load unsloth to use the LoRA adapter."
            )
        return merged_dir

    def _load_local_conductor_once(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return self._model, self._tokenizer
            mode = self._resolve_load_mode()
            logger.info("Loading local conductor | load_mode=%s", mode)
            if mode == "unsloth":
                from unsloth import FastLanguageModel

                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.manifest["adapter_dir"],
                    max_seq_length=int(self.manifest["max_seq_length"]),
                    load_in_4bit=True,
                )
                try:
                    FastLanguageModel.for_inference(model)
                except Exception:
                    model.eval()
            elif mode == "merged":
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                merged_dir = self._ensure_merged_model()
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                logger.info("Loading merged conductor from %s onto %s (bfloat16)", merged_dir, device)
                tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        str(merged_dir), dtype=torch.bfloat16, low_cpu_mem_usage=True
                    )
                except TypeError:
                    # transformers < 5 uses torch_dtype instead of dtype
                    model = AutoModelForCausalLM.from_pretrained(
                        str(merged_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
                    )
                model.to(device)
                model.eval()
            else:
                raise ValueError(f"Unknown conductor load mode: {mode}")
            self._model = model
            self._tokenizer = tokenizer
            return model, tokenizer

    def _ask_local_conductor(self, question: str) -> tuple[dict[str, Any], str]:
        import torch

        model, tokenizer = self._load_local_conductor_once()
        messages = build_conductor_messages(question, self.workers, self.manifest)
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        device = getattr(model, "device", None)
        if device is None:
            device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=self.config.max_conductor_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
        with self._lock:
            with torch.no_grad():
                output_ids = model.generate(**generate_kwargs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return parse_and_validate_plan(raw, self.workers), raw

    def plan(self, full_question: str) -> tuple[dict[str, Any], str, str]:
        planning_question = truncate_for_planning(full_question, self.max_planning_chars)
        logger.info(
            "Conductor planning start | mode=%s | full_question_chars=%s | planning_question_chars=%s | planning_preview=%s",
            self.config.conductor_mode,
            len(full_question),
            len(planning_question),
            log_preview(planning_question, limit=800),
        )
        if self.config.conductor_mode == "heuristic":
            plan = heuristic_plan_for_question(planning_question, self.workers)
            raw = json.dumps(plan)
        else:
            plan, raw = self._ask_local_conductor(planning_question)
        logger.info("Conductor raw plan | raw_preview=%s", log_preview(raw, limit=1200))
        logger.info("Conductor parsed plan | plan=%s", json.dumps(plan, ensure_ascii=False))
        return plan, raw, planning_question

    def run_chat(self, messages: list[dict[str, Any]], max_worker_tokens: int | None = None) -> dict[str, Any]:
        full_question = messages_to_question(messages)
        if not full_question.strip():
            raise ValueError("No text content found in chat messages")

        # Honor the client's max_tokens (Cursor/OpenAI clients send it) as a per-request
        # cap on worker output, without mutating shared service config.
        config = self.config
        if max_worker_tokens and max_worker_tokens > 0:
            config = dataclasses.replace(self.config, max_worker_tokens=int(max_worker_tokens))

        plan, raw_plan, planning_question = self.plan(full_question)
        final, trace, spend = execute_workflow(full_question, plan, self.manifest, config)
        logger.info(
            "Conductor request complete | steps=%s | spend_usd=%.6f | final_chars=%s | final_preview=%s",
            len(plan.get("steps", [])),
            spend,
            len(final),
            log_preview(final, limit=800),
        )

        if self.include_trace:
            final = final.rstrip() + "\n\n---\n\nConductor plan:\n```json\n" + json.dumps(plan, indent=2) + "\n```"
            final += f"\n\nEstimated OpenRouter spend: ${spend:.6f}"

        return {
            "final": final,
            "plan": plan,
            "raw_plan": raw_plan,
            "trace": trace,
            "spend_usd": spend,
            "planning_question_chars": len(planning_question),
            "full_question_chars": len(full_question),
        }


def build_openai_response(model_name: str, result: dict[str, Any], request_messages: list[dict[str, Any]]) -> dict[str, Any]:
    content = result["final"]
    prompt_text = messages_to_question(request_messages)
    prompt_tokens = rough_token_count(prompt_text)
    completion_tokens = rough_token_count(content)
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def sse_chunk(data: dict[str, Any]) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


def openai_error_response(status_code: int, message: str, err_type: str, code: str | None = None):
    # OpenAI-compatible clients (including Cursor) expect errors shaped as
    # {"error": {"message", "type", "param", "code"}}. FastAPI's default
    # {"detail": ...} body is parsed poorly by those clients and surfaces as a
    # confusing or empty error, so we always emit the OpenAI envelope.
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": err_type, "param": None, "code": code}},
    )


class AuthError(Exception):
    pass


def create_app(service: ConductorOpenAIService, server_api_key: str | None = None):
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from starlette.exceptions import HTTPException as StarletteHTTPException
    except ImportError as e:
        raise RuntimeError(
            "fastapi and uvicorn are required for the OpenAI-compatible server. "
            "Install them with: pip install fastapi 'uvicorn[standard]'"
        ) from e

    app = FastAPI(title="Mini-Conductor OpenAI-compatible Gateway")

    @app.exception_handler(StarletteHTTPException)
    async def _http_exc_handler(request: Request, exc: StarletteHTTPException):
        # Map framework HTTP errors (e.g. 404/405 for wrong paths) to OpenAI shape.
        err_type = "authentication_error" if exc.status_code == 401 else "invalid_request_error"
        return openai_error_response(exc.status_code, str(exc.detail), err_type)

    def check_auth(request: Request) -> None:
        if not server_api_key:
            return
        expected = f"Bearer {server_api_key}"
        if request.headers.get("authorization") != expected:
            raise AuthError("Invalid conductor server API key")

    @app.get("/healthz")
    async def healthz():
        return {
            "ok": True,
            "model": service.model_name,
            "conductor_mode": service.config.conductor_mode,
            "execute": service.config.execute,
        }

    @app.get("/v1/models")
    @app.get("/models")
    async def models(request: Request):
        try:
            check_auth(request)
        except AuthError as e:
            return openai_error_response(401, str(e), "authentication_error")
        return {
            "object": "list",
            "data": [
                {
                    "id": service.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local-mini-conductor",
                }
            ],
        }

    async def completions_impl(request: Request):
        try:
            check_auth(request)
        except AuthError as e:
            return openai_error_response(401, str(e), "authentication_error")

        try:
            payload = await request.json()
        except Exception:
            return openai_error_response(400, "Request body must be valid JSON", "invalid_request_error")

        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            return openai_error_response(400, "'messages' must be a list", "invalid_request_error")

        requested_model = str(payload.get("model") or service.model_name)
        stream = bool(payload.get("stream", False))
        max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens")
        logger.info(
            "OpenAI request | path=%s | model=%s | stream=%s | messages=%s | max_tokens=%s",
            request.url.path,
            requested_model,
            stream,
            len(messages),
            max_tokens,
        )

        try:
            # run_chat does blocking work (model download/load, generation, worker
            # HTTP calls). Run it in a worker thread so the event loop — and with it
            # /healthz and concurrent requests — stays responsive.
            import asyncio

            result = await asyncio.to_thread(service.run_chat, messages, max_tokens)
            response = build_openai_response(requested_model, result, messages)
        except ValueError as e:
            # Bad input (e.g. empty prompt) is a 400, not a server error.
            return openai_error_response(400, str(e), "invalid_request_error")
        except Exception as e:
            # Orchestration/worker/budget failures map to a server error so Cursor
            # shows the real reason (e.g. budget guard, missing OPENROUTER_API_KEY).
            return openai_error_response(500, str(e), "server_error")

        if not stream:
            return JSONResponse(response)

        content = response["choices"][0]["message"]["content"]
        created = response["created"]
        response_id = response["id"]

        def event_stream():
            yield sse_chunk(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": requested_model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
            )
            # Cursor is fine with chunked text; this service emits coarse chunks rather
            # than true token streaming because worker orchestration completes first.
            chunk_size = 1600
            for i in range(0, len(content), chunk_size):
                yield sse_chunk(
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": requested_model,
                        "choices": [
                            {"index": 0, "delta": {"content": content[i : i + chunk_size]}, "finish_reason": None}
                        ],
                    }
                )
            yield sse_chunk(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": requested_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        return await completions_impl(request)

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Mini-Conductor behind an OpenAI-compatible HTTP API for Cursor.")
    parser.add_argument("--harness-dir", default=os.getenv("CONDUCTOR_HARNESS_DIR"), help="Directory containing manifest.json from notebook Cell 10. Optional when --hf-model-id is set.")
    parser.add_argument("--hf-model-id", default=os.getenv("CONDUCTOR_HF_MODEL_ID"), help="Hugging Face repo to load the conductor from (e.g. bebrws/mini-conductor-qwen3-router-sft-grpo). Adapter and manifest are downloaded automatically.")
    parser.add_argument("--hf-run", default=os.getenv("CONDUCTOR_HF_RUN"), help="Specific runs/<tag> inside the HF repo. Defaults to the newest run.")
    parser.add_argument("--hf-refresh", action="store_true", default=os.getenv("CONDUCTOR_HF_REFRESH", "0") == "1", help="Check the Hub for newly uploaded runs at startup. Without this flag, restarts serve entirely from the local cache (no network).")
    parser.add_argument("--conductor-load", choices=["auto", "unsloth", "merged"], default=os.getenv("CONDUCTOR_LOAD", "auto"), help="How to load the local conductor: unsloth 4-bit LoRA (CUDA), merged 16-bit via transformers (works on Apple Silicon/CPU), or auto-detect.")
    parser.add_argument("--host", default=os.getenv("CONDUCTOR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CONDUCTOR_PORT", "8008")))
    parser.add_argument("--model-name", default=os.getenv("CONDUCTOR_MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--conductor-mode", choices=["local", "heuristic"], default=os.getenv("CONDUCTOR_MODE", "local"))
    parser.add_argument("--execute", action="store_true", default=os.getenv("CONDUCTOR_EXECUTE", "0") == "1", help="Actually call OpenRouter workers. Required for useful Cursor answers.")
    parser.add_argument("--max-worker-tokens", type=int, default=int(os.getenv("CONDUCTOR_MAX_WORKER_TOKENS", "2048")))
    parser.add_argument("--max-conductor-tokens", type=int, default=int(os.getenv("CONDUCTOR_MAX_CONDUCTOR_TOKENS", "320")))
    parser.add_argument("--worker-temperature", type=float, default=float(os.getenv("CONDUCTOR_WORKER_TEMPERATURE", "0.2")))
    parser.add_argument("--budget-usd", type=float, default=float(os.getenv("CONDUCTOR_BUDGET_USD", "5.0")), help="Per-request max budget guard.")
    parser.add_argument("--cache-dir", default=os.getenv("CONDUCTOR_CACHE_DIR"))
    parser.add_argument("--max-planning-chars", type=int, default=int(os.environ["CONDUCTOR_MAX_PLANNING_CHARS"]) if os.getenv("CONDUCTOR_MAX_PLANNING_CHARS") else None, help="Max chars of the request shown to the conductor for planning. Default: auto-sized from the model's max_seq_length (8000 for 2k-ctx models, ~91k for 32k).")
    parser.add_argument("--include-trace", action="store_true", default=os.getenv("CONDUCTOR_INCLUDE_TRACE", "0") == "1")
    parser.add_argument("--server-api-key", default=os.getenv("CONDUCTOR_SERVER_API_KEY"), help="Optional bearer token required from Cursor. If unset, no local auth is enforced.")
    parser.add_argument("--log-level", default=os.getenv("CONDUCTOR_LOG_LEVEL", "INFO"), help="Python logging level for conductor/server logs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.harness_dir and not args.hf_model_id:
        raise SystemExit(
            "Set --hf-model-id (or CONDUCTOR_HF_MODEL_ID) to load from Hugging Face, "
            "or --harness-dir/CONDUCTOR_HARNESS_DIR for a local Cell 10 harness_artifacts directory."
        )
    if args.execute and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY must be set when --execute is used.")

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info(
        "Starting Mini-Conductor OpenAI gateway | harness_dir=%s | hf_model_id=%s | mode=%s | load=%s | execute=%s | budget_usd=%s | cache_dir=%s",
        args.harness_dir,
        args.hf_model_id,
        args.conductor_mode,
        args.conductor_load,
        bool(args.execute),
        args.budget_usd,
        args.cache_dir,
    )

    service = ConductorOpenAIService(
        harness_dir=Path(args.harness_dir) if args.harness_dir else None,
        hf_model_id=args.hf_model_id,
        hf_run=args.hf_run,
        conductor_load=args.conductor_load,
        hf_refresh=bool(args.hf_refresh),
        execute=bool(args.execute),
        conductor_mode=args.conductor_mode,
        max_worker_tokens=args.max_worker_tokens,
        max_conductor_tokens=args.max_conductor_tokens,
        worker_temperature=args.worker_temperature,
        budget_usd=args.budget_usd,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        model_name=args.model_name,
        max_planning_chars=args.max_planning_chars,
        include_trace=bool(args.include_trace),
    )

    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit("Install server dependencies first: pip install fastapi 'uvicorn[standard]'") from e

    app = create_app(service, server_api_key=args.server_api_key)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
