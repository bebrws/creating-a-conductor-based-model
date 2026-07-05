# Using Mini-Conductor from Pi or Cursor

This repo includes an OpenAI-compatible gateway that lets Pi (the pi coding agent) or Cursor talk to your trained conductor model as if it were a normal OpenAI Chat Completions model.

The gateway does this:

1. Pi/Cursor sends `/v1/chat/completions` to the local gateway.
2. The gateway loads the trained conductor — either directly from Hugging Face (`--hf-model-id bebrws/mini-conductor-qwen3-router-sft-grpo-32k`) or from a local Cell 10 `harness_artifacts` directory.
3. The local conductor emits a JSON worker plan.
4. The gateway validates the plan.
5. The gateway calls the selected OpenRouter workers using `OPENROUTER_API_KEY` from the environment.
6. The gateway returns the final worker response in OpenAI-compatible format.

## Why use a gateway instead of configuring Cursor to call the LoRA directly?

Cursor can call OpenAI-compatible HTTP endpoints, but your conductor is a local LoRA adapter plus orchestration code. It is not a normal hosted chat model by itself. The gateway is the layer that turns:

```text
Cursor prompt -> conductor plan -> OpenRouter worker calls -> final answer
```

into a standard OpenAI-compatible API.

## Important: Cursor usually cannot reach `localhost`

This is the part that surprises most people, so read it before configuring Cursor.

When you set a custom OpenAI base URL, Cursor generally routes requests **through Cursor's own backend servers**, not directly from the editor process on your laptop. Those servers are on the public internet, so a base URL of `http://127.0.0.1:8008/v1` (or `http://localhost:...`) is **not reachable** and the model will fail to connect.

You have three realistic options:

1. **Public HTTPS tunnel to your gateway (most common).** Run the gateway locally/on your GPU box, then expose it with a tunnel and give Cursor the tunnel's HTTPS URL:

   ```bash
   # examples — pick one
   cloudflared tunnel --url http://127.0.0.1:8008
   # or
   ngrok http 8008
   ```

   Then in Cursor set the Base URL to the tunnel URL plus `/v1`, e.g. `https://your-subdomain.trycloudflare.com/v1`.
   Because the tunnel is public, you **must** set `CONDUCTOR_SERVER_API_KEY` (see below) so only you can call it — it can spend your OpenRouter credits.

2. **Host the gateway on a public VM** (ideally a GPU VM for `--conductor-mode local`). Bind it to `0.0.0.0`, put it behind HTTPS, and give Cursor that HTTPS URL.

3. **Try direct localhost only if your Cursor version supports it.** Some setups/clients will hit `127.0.0.1` directly. If that works for you, great — but assume option 1 is needed unless proven otherwise.

In all cases the gateway needs HTTPS in front of it for Cursor; tunnels (option 1) provide this automatically, which is why it's the easiest path.

## Requirements

Install the server deps (includes torch/transformers for local conductor inference):

```bash
pip install -r requirements-cursor-service.txt
```

`--conductor-mode local` now works on two kinds of machines:

- **CUDA GPU box**: loads the 4-bit LoRA adapter with Unsloth (`--conductor-load unsloth`).
- **Apple Silicon / CPU** (no CUDA): loads the merged 16-bit model with plain transformers on MPS/CPU (`--conductor-load merged`). The first start downloads ~16GB from Hugging Face; later starts reuse the cache.

`--conductor-load auto` (the default) picks the right one for the machine.

If you want to test orchestration without loading any model, use `--conductor-mode heuristic`.

## Start the gateway (loads the model from Hugging Face)

Set your OpenRouter key in the shell that starts the service, plus an auth token for the gateway itself:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export CONDUCTOR_SERVER_API_KEY="cursor-local-secret"
```

The trained conductor lives at `bebrws/mini-conductor-qwen3-router-sft-grpo-32k` on Hugging Face (a private repo — make sure you're logged in via `huggingface-cli login` or have `HF_TOKEN` set). Start the gateway pointing at it:

```bash
python3 -m mini_conductor.openai_server \
  --hf-model-id bebrws/mini-conductor-qwen3-router-sft-grpo-32k \
  --conductor-mode local \
  --execute \
  --cache-dir ./openrouter_cache \
  --budget-usd 5.00 \
  --host 127.0.0.1 \
  --port 8008
```

The gateway resolves the newest `runs/<tag>/` in the repo automatically (pin one with `--hf-run run-20260705-174015`), downloads the adapter + manifest, and — on machines without CUDA — the merged 16-bit model as well.

Everything is cached in `~/.cache/huggingface/hub`, and restarts are **offline-first**: if the files are already cached, the server starts instantly with no network traffic and no re-download. Pass `--hf-refresh` (or `CONDUCTOR_HF_REFRESH=1`) only when you've uploaded a new run to the repo and want the server to pick it up.

Alternatively, the old local path still works: `--harness-dir /path/to/harness_artifacts` instead of `--hf-model-id`.

For a no-model smoke test (no download, workers dry-run):

```bash
python3 -m mini_conductor.openai_server \
  --hf-model-id bebrws/mini-conductor-qwen3-router-sft-grpo-32k \
  --conductor-mode heuristic \
  --cache-dir ./openrouter_cache \
  --budget-usd 5.00 \
  --host 127.0.0.1 \
  --port 8008
```

## Test with curl

Models endpoint:

```bash
curl http://127.0.0.1:8008/v1/models \
  -H "Authorization: Bearer ${CONDUCTOR_SERVER_API_KEY:-dummy}"
```

Chat completions endpoint:

```bash
curl http://127.0.0.1:8008/v1/chat/completions \
  -H "Authorization: Bearer ${CONDUCTOR_SERVER_API_KEY:-dummy}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mini-conductor-qwen3-router-sft-grpo-32k",
    "messages": [
      {"role": "user", "content": "Write a Python function to reverse a string and explain edge cases."}
    ],
    "stream": false
  }'
```

Streaming is also supported:

```json
"stream": true
```

The service emits coarse chunks after orchestration completes. It does not stream individual worker tokens live yet.

## Configure Pi (pi coding agent)

Unlike Cursor, Pi runs entirely on your machine, so it can reach `http://127.0.0.1:8008` directly — no tunnel needed.

This repo ships a Pi provider extension at `pi-local-conductor-provider/` that registers the gateway as an OpenAI-compatible provider named `mini-conductor` with one model: `mini-conductor-qwen3-router-sft-grpo-32k`.

One-time setup:

1. **Register the extension** in `~/.pi/agent/settings.json` by adding this to the `packages` array (already present on this machine):

   ```json
   {
     "source": "/Users/bbarrows/repos/large-llm-training/pi-local-conductor-provider",
     "extensions": ["+./index.ts"]
   }
   ```

2. **Store the gateway key** in `~/.pi/agent/auth.json` so the extension can authenticate to the gateway (already present on this machine). The value must match `CONDUCTOR_SERVER_API_KEY`:

   ```json
   "mini-conductor": { "key": "cursor-local-secret" }
   ```

Each session:

1. Start the gateway as shown above (`--hf-model-id ... --conductor-mode local --execute`).
2. Launch Pi and switch to the conductor model with `/model` → provider `Mini Conductor Local` → `Mini Conductor SFT-GRPO 32k (local)`, or start Pi with it directly:

   ```bash
   pi --provider mini-conductor --model mini-conductor-qwen3-router-sft-grpo-32k
   ```

3. Ask a question. The gateway logs show the conductor's plan and each OpenRouter worker call; Pi receives the final answer.

Caveats for Pi specifically:

- The gateway returns plain assistant text and does **not** implement OpenAI `tool_calls`, so Pi cannot use it as a full autonomous coding agent (no bash/edit tool execution). Use it for chat/ask-style questions, or run Pi with `--no-tools` / one-shot `-p` prompts. Keep a tool-capable provider (e.g. your OpenRouter default) for normal agentic coding.
- The first request after startup is slow on Apple Silicon: the merged model loads into memory (~16GB) and planning runs on MPS. Subsequent requests reuse the loaded model.
- Every executed request spends OpenRouter credits (bounded by `--budget-usd` per request).

## Configure Cursor

Cursor's UI changes over time, but the configuration you want is an OpenAI-compatible/custom OpenAI endpoint. In Cursor: Settings → Models → enable a custom OpenAI API key / override the OpenAI Base URL, then add a custom model.

- Provider/type: OpenAI-compatible or custom OpenAI ("Override OpenAI Base URL")
- Base URL: your **public HTTPS** gateway URL plus `/v1` (see the localhost warning above), e.g. `https://your-subdomain.trycloudflare.com/v1`
- API key: the value of `CONDUCTOR_SERVER_API_KEY` if you set one; otherwise any non-empty dummy string
- Model name: add a custom model named exactly `mini-conductor-qwen3-router-sft-grpo-32k`

Then select `mini-conductor-qwen3-router-sft-grpo-32k` as the model in Cursor.

Note on Cursor model verification: when you add a custom OpenAI model, Cursor often "verifies" it by sending a tiny chat request. That triggers a real conductor + OpenRouter run. If you want verification to succeed without spending credits or needing a GPU, first verify with `--conductor-mode heuristic` (and you can omit `--execute` so it dry-runs), then switch to `local` + `--execute` for real use.

If Cursor only accepts a base URL without `/v1`, this gateway also exposes `/chat/completions` and `/models`, so a base URL without the `/v1` suffix can work with clients that append paths differently.

## Logging

The server logs conductor decisions and OpenRouter calls at `INFO` level by default:

- incoming OpenAI-compatible request summary
- conductor planning prompt preview
- raw and parsed conductor plan
- each selected worker model, role, subtask, and access setting
- each OpenRouter request summary
- each OpenRouter response usage/cost/preview
- final answer summary and total spend

Control log verbosity/preview size with:

```bash
export CONDUCTOR_LOG_LEVEL=INFO
export CONDUCTOR_LOG_PREVIEW_CHARS=500
```

or start the server with:

```bash
python3 -m mini_conductor.openai_server --log-level INFO ...
```

## Behavior notes and limitations

- Errors are returned in OpenAI format (`{"error": {"message", "type", ...}}`) so Cursor shows a readable message. Bad requests return `400`; auth failures return `401`; orchestration/budget/worker failures return `500` with the real reason.
- The client's `max_tokens` (or `max_completion_tokens`) is honored as a per-request cap on worker output. If omitted, the server default (`--max-worker-tokens`) is used.
- Cursor will only see the final answer. The conductor plan and trace are logged by the service/harness, not exposed to Cursor unless you start with `--include-trace`.
- Cursor's agent/edit features expect strong tool-calling and a large context window. This gateway returns plain assistant text and does not implement OpenAI tool/function calling, so it is best used in Cursor's chat/ask flows rather than as a full autonomous agent model.
- The same limitation applies to Pi as a coding agent: if you set this conductor as Pi's default provider, Pi may show text like `tools: [{"name":"bash",...}]` because the gateway does not yet return real OpenAI `tool_calls` objects for Pi to execute. For Pi smoke tests, use `--no-tools` or direct `-p` prompts; for normal Pi coding-agent work, keep a regular tool-capable provider as default.
- The local LoRA conductor needs a compatible runtime. For an 8B Qwen3 LoRA, a GPU box is the practical path.
- The gateway uses a per-request budget guard. Raise or lower `--budget-usd` as needed.
- For large Cursor context, the conductor only sees a clipped planning copy by default (`--max-planning-chars 8000`), while OpenRouter workers receive the full request text.
- Do not expose this service publicly without `CONDUCTOR_SERVER_API_KEY` and network-level protection. It can spend your OpenRouter credits.

## Environment variables

All major options can be configured through environment variables:

```bash
export CONDUCTOR_HARNESS_DIR="/path/to/harness_artifacts"
export CONDUCTOR_MODE="local"          # or heuristic
export CONDUCTOR_EXECUTE="1"           # actually call OpenRouter workers
export CONDUCTOR_CACHE_DIR="./openrouter_cache"
export CONDUCTOR_BUDGET_USD="5.0"
export CONDUCTOR_PORT="8008"
export CONDUCTOR_SERVER_API_KEY="cursor-local-secret"
export OPENROUTER_API_KEY="sk-or-..."
```

Then run:

```bash
python3 -m mini_conductor.openai_server
```
