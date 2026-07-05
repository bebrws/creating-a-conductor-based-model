  # Loads the conductor straight from Hugging Face (no Drive download needed).
  # On this Mac (no CUDA) it auto-selects the merged 16-bit model on MPS;
  # the first start downloads ~16GB to the HF cache, later starts reuse it.
  export CONDUCTOR_HF_MODEL_ID="bebrws/mini-conductor-qwen3-router-sft-grpo-32k"
  export CONDUCTOR_SERVER_API_KEY="cursor-local-secret"
  export OPENROUTER_API_KEY="sk-or-..."   # required with --execute

  python3 -m mini_conductor.openai_server \
    --hf-model-id "$CONDUCTOR_HF_MODEL_ID" \
    --conductor-mode local \
    --execute \
    --cache-dir ./openrouter_cache \
    --budget-usd 5.00 \
    --host 127.0.0.1 \
    --port 8008

  # No-model smoke test (no HF download, no GPU, workers dry-run): swap the two
  # mode flags for:  --conductor-mode heuristic   and drop --execute




# My personal notes:

export OPENROUTER_API_KEY="sk-or-..."          # your real key
export CONDUCTOR_SERVER_API_KEY="cursor-local-secret"
python3 -m mini_conductor.openai_server \
  --hf-model-id bebrws/mini-conductor-qwen3-router-sft-grpo-32k \
  --conductor-mode local \
  --execute \
  --cache-dir ./openrouter_cache \
  --budget-usd 5.00 \
  --host 127.0.0.1 \
  --port 8008

# then in another shell:

pi --provider mini-conductor --model mini-conductor-qwen3-router-sft-grpo-32k