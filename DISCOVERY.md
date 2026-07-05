# DISCOVERY

This file records the discovery/research performed for building a Conductor-style agent based on the two Sakana white papers in this directory, Unsloth notebooks, Colab/compute options, OpenRouter worker models, and the proposed training/deployment approach.

## Current directory inputs

Working directory:

```text
~/repos/large-llm-training
```

PDFs found:

```text
Sakana-White-Paper-Orchestrat-Agents-Natrual-Language.pdf
SakanaWhitePaper-Trinity-LLM-Coordinator.pdf
```

Text extraction was performed with `pypdf` via `uv run --with pypdf`. Extracted files were written under:

```text
/tmp/large-llm-training-analysis/
```

Extracted PDF metadata:

```text
Sakana-White-Paper-Orchestrat-Agents-Natrual-Language.pdf: 39 pages
SakanaWhitePaper-Trinity-LLM-Coordinator.pdf: 30 pages
```

---

## Paper 1: Learning to Orchestrate Agents in Natural Language with the Conductor

File:

```text
Sakana-White-Paper-Orchestrat-Agents-Natrual-Language.pdf
```

Title discovered in PDF metadata:

```text
Learning to Orchestrate Agents in Natural Language with the Conductor
```

ArXiv ID found in metadata:

```text
https://arxiv.org/abs/2512.04388v5
```

### Core idea

The paper introduces an RL-trained **Conductor** model. The Conductor is not primarily a problem solver. It is trained to create a workflow for a pool of worker LLMs.

The Conductor outputs a complete orchestration strategy, including:

```python
model_id = [2, 0]
subtasks = [
    "Develop an efficient algorithm...",
    "Implement the algorithm described by the previous agent in Python"
]
access_list = [[], ["all"]]
```

The external harness then executes those subtasks by calling the selected worker LLMs.

### Important paper details

Discovered details:

- Base coordinator model: **Qwen2.5-7B**.
- The Conductor is trained using **GRPO**.
- It outputs workflows of up to **5 workflow steps**.
- It coordinates a pool of open-source and proprietary worker models.
- It learns natural-language prompt engineering, model selection, verification, refinement, and communication topology.
- It can be extended with:
  - randomized worker pools,
  - recursive Conductor calls,
  - adaptive test-time scaling.

### Workflow schema in paper

The prompt asks the Conductor to output three Python lists:

```text
model id
subtasks
access list
```

The access list determines what previous worker outputs are visible to the current worker.

Examples:

```python
access_list = [[], ["all"]]
```

means:

- step 0 sees no prior worker output,
- step 1 sees all previous worker outputs.

### Reward structure

The Conductor reward is determined by two progressive conditions:

1. **Format condition**: reward is `0` if the Conductor output cannot be parsed into Python lists of subtasks, worker IDs, and access lists.
2. **Correctness condition**: reward is `1` if executing the well-formatted workflow produces a correct final answer, and `0.5` otherwise.

This means worker choice directly matters during RL/GRPO because the final reward depends on the actual worker-model outputs.

### Training setup from appendix

Important extracted training details:

```text
Base model: Qwen2.5-7B
Max completion length: 1024
Training iterations: 200
Questions per iteration: 4
Rollouts per question: 64
Effective rollouts: 200 * 4 * 64 = 51,200 workflow rollouts
Sampling temperature: 1.0
Optimizer: AdamW
Base learning rate: 1e-6
Warmup ratio: 0.03
Reference model synchronization: disabled
KL divergence penalty: 0
Compute: 2 NVIDIA H100 80GB GPUs
```

Worker settings from the paper:

```text
Worker max completion tokens: 4096
Worker temperature: 0.2
Closed-source reasoning budgets minimized:
- Gemini 2.5 Pro: 128 tokens
- Claude Sonnet 4: 0
- GPT-5: minimal
Qwen3-32B decoding:
- top_p = 0.8
- top_k = 20
- presence_penalty = 1.0
```

### Recursion training

The paper trains a recursive Conductor variant by fine-tuning the trained Conductor:

```text
Recursive fine-tune iterations: 20
Filtered subset size: 350 samples
Subset composition: 175 LiveCodeBench + 175 RLPR
Rollouts per sample: 64
Batch size: 256
KL penalty: 0
Initial non-recursive reward discount: 0.25
```

### Datasets used

The Conductor paper says the training data is built from four reasoning domains:

```text
MATH500
MMLU
RLPR
LiveCodeBench V1
```

Evaluation included unseen/in-domain and out-of-domain tasks:

```text
MATH500 test
MMLU test
RLPR test
LiveCodeBench V6
AIME25
BigCodeBench
GPQA-Diamond
```

The paper states that the base model and datasets are publicly available, but the exact paper-specific 960-question filtered training mixture was not discovered as a single packaged public dataset.

### Public dataset URLs identified

MATH / MATH500:

```text
https://huggingface.co/datasets/HuggingFaceH4/MATH-500
https://huggingface.co/datasets/hendrycks/competition_math
```

MMLU:

```text
https://huggingface.co/datasets/cais/mmlu
```

RLPR / WebInstruct:

```text
https://huggingface.co/datasets/openbmb/RLPR-Train-Dataset
https://huggingface.co/datasets/openbmb/RLPR-Evaluation
https://huggingface.co/datasets/TIGER-Lab/WebInstructSub
https://huggingface.co/datasets/TIGER-Lab/WebInstructFull
```

LiveCodeBench:

```text
https://github.com/LiveCodeBench/LiveCodeBench
https://huggingface.co/datasets/livecodebench/code_generation_lite
```

BigCodeBench:

```text
https://huggingface.co/datasets/bigcode/bigcodebench
https://huggingface.co/datasets/bigcode/bigcodebench-hard
```

GPQA:

```text
https://huggingface.co/datasets/Idavidrein/gpqa
```

Note: GPQA is gated/terms-controlled on Hugging Face.

AIME 2025:

```text
https://www.kaggle.com/benchmarks/open-benchmarks/aime-2025
```

### Base model URLs for Conductor-style coordinator

Original Qwen2.5 7B models:

```text
https://huggingface.co/Qwen/Qwen2.5-7B
https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
```

Unsloth variants:

```text
https://huggingface.co/unsloth/Qwen2.5-7B
https://huggingface.co/unsloth/Qwen2.5-7B-Instruct
```

Recommended 8B-class practical alternative:

```text
https://huggingface.co/Qwen/Qwen3-8B
https://huggingface.co/unsloth/Qwen3-8B
https://huggingface.co/unsloth/Qwen3-8B-unsloth-bnb-4bit
```

---

## Paper 2: TRINITY: An Evolved LLM Coordinator

File:

```text
SakanaWhitePaper-Trinity-LLM-Coordinator.pdf
```

Title discovered in PDF metadata:

```text
TRINITY: An Evolved LLM Coordinator
```

ArXiv ID found in metadata:

```text
https://arxiv.org/abs/2512.04695v3
```

### Core idea

TRINITY is a lightweight learned coordinator that uses a small model's hidden states and a tiny trainable head to select:

```text
1. which worker LLM to call
2. which role to assign
```

Roles:

```text
Thinker
Worker
Verifier
```

The coordinator loops for up to a fixed turn budget. The process terminates when a verifier accepts the current response, or when the turn budget is exhausted.

### Important TRINITY details

Extracted/discovered details:

```text
Coordinator backbone: Qwen3-0.6B
Trainable head: ~10K parameters
Additional training: singular value fine-tuning on selected SLM matrices
Training optimizer: sep-CMA-ES
Maximum coordination turns: 5
Worker max generated tokens: 4096
```

TRINITY uses hidden states, especially penultimate-token or early-token hidden states, as contextual representations for the coordination head.

### Why TRINITY is harder to implement directly in Unsloth

TRINITY's training loop is not the default Unsloth workflow. It requires:

- custom hidden-state extraction,
- a custom trainable coordination head,
- evolutionary optimization via sep-CMA-ES,
- trajectory-level binary reward evaluation,
- repeated worker calls.

Unsloth's public notebooks primarily support:

- SFT,
- LoRA/QLoRA,
- GRPO,
- DPO/ORPO-style workflows,
- tool-calling examples.

Therefore, the practical notebook plan should initially use a **Conductor-style generative coordinator** with **TRINITY roles** rather than implementing full TRINITY hidden-state CMA-ES immediately.

### Public TRINITY-style base model URLs

Qwen3 0.6B:

```text
https://huggingface.co/Qwen/Qwen3-0.6B
https://huggingface.co/Qwen/Qwen3-0.6B-Base
https://huggingface.co/unsloth/Qwen3-0.6B
```

---

## Open-source worker models mentioned in papers

The papers use a mix of closed and open-source worker LLMs. Public/open-source worker models identified:

```text
https://huggingface.co/Qwen/Qwen3-32B
https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
https://huggingface.co/google/gemma-3-27b-it
```

Note: `google/gemma-3-27b-it` is gated/manual license acceptance on Hugging Face.

---

## Unsloth notebook discovery

Fetched page:

```text
https://unsloth.ai/docs/get-started/unsloth-notebooks
```

The page was fetched successfully with a browser-like user agent and contained many Colab notebook links.

Relevant Unsloth notebooks discovered:

### GRPO notebooks

```text
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/DeepSeek_R1_0528_Qwen3_(8B)_GRPO.ipynb
```

### SFT / instruct notebooks

```text
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-Instruct.ipynb
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb
```

### Tool-calling notebook

```text
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_Coder_(1.5B)-Tool_Calling.ipynb
```

### Notebook code details inspected

`Qwen3_(4B)-GRPO.ipynb` includes:

- `FastLanguageModel.from_pretrained`
- `FastLanguageModel.get_peft_model`
- `SFTTrainer` warm-up on `unsloth/OpenMathReasoning-mini`
- custom format rewards
- answer-check rewards
- `GRPOConfig`
- `GRPOTrainer`
- example settings:

```text
model_name = "unsloth/Qwen3-4B-Base"
max_seq_length = 2048
lora_rank = 32
num_generations = 4
max_steps = 100
```

`Llama3.1_(8B)-GRPO.ipynb` includes:

```text
model_name = "unsloth/meta-Llama-3.1-8B-Instruct"
max_seq_length = 1024
lora_rank = 32
GRPOTrainer
GSM8K reward functions
format rewards
correctness rewards
```

`Qwen3_(4B)-Instruct.ipynb` includes:

- SFT via `SFTTrainer`,
- chat template setup,
- `mlabonne/FineTome-100k`,
- response-only training helper.

`Qwen2.5_Coder_(1.5B)-Tool_Calling.ipynb` includes:

- OpenAI-style function definitions,
- tool schema examples,
- JSON generation constraints,
- examples of multi-step tool usage.

### Practical conclusion from Unsloth discovery

The best practical path is:

```text
Conductor-style generative coordinator
+ JSON workflow schema
+ TRINITY Thinker/Worker/Verifier roles
+ SFT warm start
+ GRPO-lite with actual worker execution
```

This is much more Colab-compatible than full TRINITY CMA-ES.

---

## Colab MCP / Pi discovery

Earlier discovery found that Pi does not include built-in MCP support, but can use an extension.

Installed/configured:

```text
npm:pi-mcp-adapter
```

Configured official Google Colab MCP server:

```text
~/.pi/agent/mcp.json
```

Server:

```text
googlecolab/colab-mcp
https://github.com/googlecolab/colab-mcp
```

The configured command was validated with a JSON-RPC initialize handshake:

```text
uvx --python 3.13 git+https://github.com/googlecolab/colab-mcp
```

The server responded as:

```json
{"serverInfo": {"name": "ColabMCP"}}
```

This means a Pi-to-Colab workflow is possible, but the current plan below focuses on a self-contained Colab Pro+ notebook.

---

## Compute and cost discovery

### Lambda.ai prices provided by user

```text
1x GH200 96GB ARM64 + H100: $2.29/hr
2x H100 80GB SXM5: $8.38/hr
1x H100 80GB SXM5: $4.29/hr
1x H100 80GB PCIe: $3.29/hr
1x A10 24GB PCIe: $1.29/hr
1x A100 40GB SXM4: $1.99/hr
8x Tesla V100 16GB: $6.32/hr
```

### Approximate $50 runtime on Lambda

```text
1x A100 40GB at $1.99/hr: ~25.1 hr
1x GH200 96GB at $2.29/hr: ~21.8 hr
1x H100 PCIe at $3.29/hr: ~15.2 hr
1x H100 SXM5 at $4.29/hr: ~11.7 hr
2x H100 SXM5 at $8.38/hr: ~6.0 hr
1x A10 24GB at $1.29/hr: ~38.8 hr
8x V100 at $6.32/hr: ~7.9 hr
```

### Colab Pro+ discovery from user-provided details

User plans to use Colab Pro+:

```text
$49.99/month
600 Compute Units/month
Approx. 40 A100 hours if A100 burns 13-15 CUs/hr
Potential access: T4, V100, L4, RTX 6000 Ada, A100, H100
High-RAM runtimes up to ~52GB RAM
Temporary local disk often ~150GB-256GB
Google Drive for persistent storage
```

Practical recommendation discovered:

```text
Use Colab Pro+ first for development and first training attempts.
Fallback to Lambda only if Colab cannot provide A100/RTX6000/H100 or if runtime interruptions become a blocker.
```

---

## OpenRouter worker model discovery

The user requested these worker models:

```text
z-ai/glm-5.2
moonshotai/kimi-k2.7-code
qwen/qwen3.7-plus
minimax/minimax-m3
```

OpenRouter's model API was queried and all four IDs were found.

### z-ai/glm-5.2

```text
OpenRouter ID: z-ai/glm-5.2
Name: Z.ai: GLM 5.2
Modality: text -> text
Context length: 1,048,576
Max completion tokens: 32,768
Prompt price: $0.00000095/token = $0.95/M input tokens
Completion price: $0.000003/token = $3.00/M output tokens
Input cache read: $0.00000018/token = $0.18/M tokens
Supports structured outputs, tools, reasoning controls, temperature, top_p, seed, etc.
```

Noted description from OpenRouter summary: suited for long-horizon agent workflows and project-level software engineering.

### moonshotai/kimi-k2.7-code

```text
OpenRouter ID: moonshotai/kimi-k2.7-code
Name: MoonshotAI: Kimi K2.7 Code
Modality: text+image -> text
Context length: 262,144
Max completion tokens: 16,384
Prompt price: $0.00000074/token = $0.74/M input tokens
Completion price: $0.0000035/token = $3.50/M output tokens
Input cache read: $0.00000015/token = $0.15/M tokens
Supports structured outputs, tools, reasoning controls, temperature, top_p, seed, etc.
```

Noted description from OpenRouter summary: coding-focused model for end-to-end programming tasks over long contexts.

### qwen/qwen3.7-plus

```text
OpenRouter ID: qwen/qwen3.7-plus
Name: Qwen: Qwen3.7 Plus
Modality: text+image -> text
Context length: 1,000,000
Max completion tokens: 65,536
Prompt price: $0.00000032/token = $0.32/M input tokens
Completion price: $0.00000128/token = $1.28/M output tokens
Input cache read: $0.000000064/token = $0.064/M tokens
Input cache write: $0.0000004/token = $0.40/M tokens
Supports structured outputs, tools, reasoning controls, temperature, top_p, seed, etc.
```

This is the cheapest of the four requested workers by listed input/output token prices.

### minimax/minimax-m3

```text
OpenRouter ID: minimax/minimax-m3
Name: MiniMax: MiniMax M3
Modality: text+image+video -> text
Context length reported by model: 1,048,576
Top provider context length: 524,288
Max completion tokens: 512,000
Prompt price: $0.0000003/token = $0.30/M input tokens
Completion price: $0.0000012/token = $1.20/M output tokens
Input cache read: $0.00000006/token = $0.06/M tokens
Supports structured outputs, tools, reasoning controls, temperature, top_p, seed, etc.
```

This is also very low-cost among the requested models and has multimodal support.

### Approximate per-call cost intuition

For a worker call with roughly 1,000 input tokens and 1,000 output tokens:

```text
minimax/minimax-m3: about $0.0015
qwen/qwen3.7-plus: about $0.0016
z-ai/glm-5.2: about $0.0040
moonshotai/kimi-k2.7-code: about $0.0042
```

For 2,000 input tokens and 2,000 output tokens:

```text
minimax/minimax-m3: about $0.0030
qwen/qwen3.7-plus: about $0.0032
z-ai/glm-5.2: about $0.0079
moonshotai/kimi-k2.7-code: about $0.0085
```

Actual costs will depend on prompt length, generated length, cache support, retries, and OpenRouter provider routing.

---

## Key design discovery: worker model choice affects Conductor training

The choice of worker models has little direct effect during pure SFT because no workers are called.

During GRPO/RL, worker choice matters heavily because the Conductor's reward depends on the answer produced by executing the workflow against those workers.

Therefore, the Conductor learns:

- which worker is good at which task,
- which worker is good as thinker vs worker vs verifier,
- how to prompt each worker,
- which worker combinations succeed,
- cost/performance tradeoffs if cost penalties are included.

Risk discovered:

```text
If worker IDs are fixed, the Conductor can overfit to model_id numbers.
```

Mitigation:

```text
Use randomized worker order and randomized worker subsets during training.
Always provide worker metadata in the prompt.
Make the Conductor read the worker descriptions instead of memorizing fixed IDs.
```

---

## Recommended practical architecture discovered

The trained local model should not directly call OpenRouter. Instead:

```text
User question
  -> trained local Conductor model in Colab
  -> JSON workflow plan
  -> Python harness parses/validates plan
  -> harness calls OpenRouter worker models
  -> harness passes prior worker results according to access list
  -> final answer returned
```

Therefore, the project needs:

1. a local coordinator model trained with Unsloth,
2. a worker registry for OpenRouter models,
3. a JSON workflow schema,
4. a workflow executor/harness,
5. caching for OpenRouter calls,
6. reward functions for SFT/GRPO,
7. evaluation baselines.

---

## Recommended target approach discovered

Best practical plan:

```text
Mini-Conductor = Conductor-style generative coordinator + TRINITY roles.
```

Use:

```text
Local coordinator: unsloth/Qwen3-8B-unsloth-bnb-4bit or unsloth/Qwen3-8B
Workers: OpenRouter models specified by user
Training: SFT warm-start, then GRPO-lite
Notebook environment: Colab Pro+ first, Lambda fallback only if needed
```

Reason:

- Closest to Conductor paper while feasible in Unsloth.
- Incorporates TRINITY Thinker/Worker/Verifier structure.
- Avoids implementing custom hidden-state CMA-ES initially.
- Keeps most work inside Colab Pro+.

---

## Existing local file

There is an existing file:

```text
ORIGINAL_ANALYSIS.md
```

It contains earlier analysis and plan text. `DISCOVERY.md` and `PLAN.md` are now created separately as requested.
