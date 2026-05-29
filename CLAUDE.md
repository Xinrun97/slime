# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (editable, no deps):**
```bash
pip install -e . --no-deps
```

**Lint (pre-commit hooks: ruff, black, isort, autoflake):**
```bash
pre-commit run --all-files
```
Line length is 119. Imports are sorted with `isort --profile=black`.

**CPU-only unit tests (no GPU required):**
```bash
# Run all CPU unit tests
python -m pytest tests/test_sample.py tests/test_agent_trajectory.py tests/test_dp_schedule.py \
  tests/test_chunked_gae.py tests/test_loss_cp_invariance.py tests/test_rm_gpqa.py

# Run plugin contract tests (validate customization interface signatures)
python -m pytest tests/plugin_contracts/
```

**Single CPU test file:**
```bash
python tests/test_sample.py
python tests/test_agent_trajectory.py
```

**E2E tests (require GPUs + Docker image `slimerl/slime:latest`):**
Each e2e test file has `NUM_GPUS = N` at the top and `prepare()` / `execute()` functions. In CI they run via:
```bash
python tests/ci/gpu_lock_exec.py --count <NUM_GPUS> -- python tests/<test_file>.py
```
CI jobs are label-gated: add `run-ci-short`, `run-ci-megatron`, `run-ci-changed`, etc. to a PR to trigger them.

## Architecture

slime has three cooperating layers, connected through Ray:

```
Megatron (training) ←──── Data Buffer ────→ SGLang router (rollout / inference)
        ↑                                              ↑
   weight sync                              custom generation logic
   (full or delta)                          (tool calls, sandboxes, agents)
```

**Training** (`train.py` / `train_async.py`): Megatron actors run forward/backward on batches pulled from the Data Buffer. Supports GRPO, PPO, REINFORCE++, and on-policy distillation (OPD).

**Rollout** (`slime/rollout/`): SGLang actors generate samples via `sglang_rollout.py`. The rollout loop, reward computation, and generation step are all replaceable via `--*-path` arguments without touching core code.

**Data Buffer** (`slime/ray/`): Ray object store bridges the async rollout and synchronous training steps. Handles prompt initialization, re-queuing, and partial rollout recycling.

**Core data structure** (`slime/utils/types.py`): `Sample` is the unit that flows through the entire pipeline — it carries `tokens`, `loss_mask`, `reward`, `rollout_log_probs`, `status`, and multi-turn metadata. It crosses Ray actor boundaries as a dict (`to_dict` / `from_dict`).

**Multi-turn agent infrastructure** (`slime/agent/`):
- `trajectory.py`: `TurnRecord` → `TurnSegment` → `TokenSegment` merge chain with per-token `loss_mask`
- `sandbox.py`: `E2BSandbox` for code execution
- `parsing.py`: `parse_model_output` — extracts reasoning, visible text, and tool calls from raw model output

## Customization Interfaces

All custom logic is injected via `--*-path` CLI arguments that point to Python import paths. The key ones:

| Argument | When to use |
|---|---|
| `--custom-generate-function-path` | Per-sample agent loop, tool calls, RAG, sandbox — reuses default rollout outer loop |
| `--rollout-function-path` | Replace the entire rollout orchestration |
| `--custom-rm-path` | Custom reward (verifier, test execution, remote service) |
| `--dynamic-sampling-filter-path` | Reject sample groups (e.g., DAPO: drop all-same-reward groups) |
| `--buffer-filter-path` | Filter the rollout buffer before training |
| `--rollout-data-postprocess-path` | Modify loss masks or add metadata after log_probs are computed |
| `--custom-loss-function-path` | Replace the policy gradient loss (requires `--loss-type custom_loss`) |

See `customization.md` for full signatures and examples. `examples/` contains working implementations for multi-agent, search-R1, fully-async, and coding-agent RL patterns.

**Plugin contract tests** (`tests/plugin_contracts/`) validate that custom implementations satisfy the required signatures without GPUs. Pass your own path via env var or CLI arg:
```bash
python tests/plugin_contracts/test_plugin_rollout_contracts.py \
  --rollout-function-path my_project.custom_rollout.generate_rollout
```

## Model Support

Megatron ↔ HuggingFace weight converters live in `slime/backends/megatron_utils/convert_checkpoint/`. Supported families: Qwen3.6/3.5/3Next/3MoE/3/2.5, DeepSeek-V3/V3.1/R1, GLM4/GLM4MoE, Llama 3, MIMO.

## Training on a Custom Coding Agent (Qwen3-27B + simple-coding-harness)

The goal: use `~/repos/simple-coding-harness` (a provider-agnostic coding agent using LiteLLM) + slime to train Qwen3-27B via RL so the model improves at coding tasks over time.

**Why this works:** `examples/coding_agent_rl/` already does this pattern with Claude Code CLI. The target setup replaces that CLI with `simple-coding-harness` and an open model served by SGLang. Qwen3-27B is natively supported (converters exist). `simple-coding-harness/rl_data/grpo/reward.py` already implements GRPO-compatible rewards (`r_task`, `r_quality_det`, `r_cost_penalty`).

**Integration architecture:**
```
slime training loop (Megatron Qwen3-27B)
    ↓ --rollout-function-path
Custom rollout function (new):
  1. Boot sandbox (E2B or Docker)
  2. Start simple-coding-harness → MODEL=openai/qwen3 OPENAI_API_BASE=http://localhost:30000/v1
  3. Middleware intercepts LiteLLM calls → records token-level log_probs from SGLang
  4. Collect reward via harness reward system
  5. Return TokenSegment to slime
```

**Key integration files to write:**
- `middleware.py`: bypass LiteLLM, call SGLang `/v1/chat/completions` with `logprobs=True`, return token IDs + log_probs — modeled on `examples/coding_agent_rl/middleware.py`
- `rollout_fn.py`: orchestrate sandbox + harness + trajectory collection — modeled on `examples/coding_agent_rl/generate.py`
- `reward.py` wrapper: adapt `simple-coding-harness/rl_data/grpo/reward.py` to `async (args, sample) -> float` signature for `--custom-rm-path`

**Recommended start:** validate the pipeline end-to-end with Qwen3-4B/8B before scaling to 27B. Use `r_quality_det` (continuous test-pass ratio) rather than binary `r_task` to reduce reward sparsity.
