import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";

const MODEL_ID = "mini-conductor-qwen3-router-sft-grpo-32k";

// Served by the local gateway (start it with --hf-model-id bebrws/mini-conductor-
// qwen3-router-sft-grpo-32k). The conductor plans over a 32k-token window; workers
// receive the full request via OpenRouter, so Pi may send larger contexts and the
// gateway clips only the planning copy.
const MODELS: ProviderModelConfig[] = [
	{
		id: MODEL_ID,
		name: "Mini Conductor SFT-GRPO 32k (local)",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128_000,
		maxTokens: 2_048,
	},
];

export default function (pi: ExtensionAPI) {
	pi.registerProvider("mini-conductor", {
		name: "Mini Conductor Local",
		baseUrl: "http://127.0.0.1:8008/v1",
		// Resolve the local gateway bearer token from Pi's auth store at request time.
		// This keeps the secret out of the repo and avoids depending on the Pi process
		// inheriting CONDUCTOR_SERVER_API_KEY from your shell.
		apiKey:
			"!python3 -c \"import json,pathlib; p=pathlib.Path.home()/'.pi/agent/auth.json'; print(json.loads(p.read_text())['mini-conductor']['key'])\"",
		api: "openai-completions",
		compat: {
			supportsStore: false,
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
			supportsUsageInStreaming: false,
			supportsStrictMode: false,
			supportsLongCacheRetention: false,
			maxTokensField: "max_tokens",
		},
		models: MODELS,
	});
}
