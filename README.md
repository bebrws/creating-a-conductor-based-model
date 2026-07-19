# Creating a Router Based Model

Similar to [OpenRouter's AutoRouter](https://openrouter.ai/openrouter/auto-beta) this will create a model that will anaylze your prompt. Pick the best model for the job and route your prompt to that model.
The repo includes instructions for use with Pi. To get started clone the repository and ask to install the model to Pi and what command is needed to run the OpenAI compatible endpoint.

## What To Expect

Prompt: `Draw a Pelican riding a bike using SVG`

Results in the OpenAI compatible server hosting the conductor/router model being handed the prompt and it generates a JSON plan — not just one model choice, but one or more steps, each assigned to a worker model (parse_and_validate_plan). execute_workflow then runs those steps by calling the chosen worker models through OpenRouter, giving them the full untruncated request, under a per-request budget guard. The final synthesized answer is wrapped back into an OpenAI-shaped response (optionally as coarse SSE chunks, since orchestration finishes before "streaming" starts) and returned to the harness as if a single model had answered.
