# Lium Inference — One-Script GPU Pod Launcher for Open-Source LLMs

Spin up a rented GPU on [Lium.io](https://lium.io), deploy an open-source LLM with vLLM, and get an OpenAI-compatible endpoint ready for Open Agent — all from a single shell script.

## Quick Start

```bash
# 1. Install prerequisites (one-time)
pip install lium-cli && lium init
sudo apt install jq curl

# 2. Fund your Lium account
lium fund

# 3. Launch your inference endpoint
./lium-inference.sh                    # Interactive model picker
./lium-inference.sh --model qwen3-235b # Direct model selection

# 4. When done, tear down
./lium-inference.sh --stop
```

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│ Choose Model │────▶│ Auto-Select  │────▶│ Launch Lium Pod │────▶│ Deploy vLLM  │
│ (23 options) │     │ GPU on Lium  │     │ (lium up)       │     │ (OpenAI API) │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
                                                                         │
                                                                         ▼
                                                                  ┌──────────────┐
                                                                  │ OA Endpoint  │
                                                                  │ Configured!  │
                                                                  └──────────────┘
```

1. **Pick a model** — Interactive menu or `--model` flag
2. **GPU auto-selection** — Script maps model VRAM needs to the right GPU type (RTX 4090 → A100 → H100), with fallback if your first choice is unavailable
3. **Pod launch** — Uses `lium up` with the right GPU type and count
4. **vLLM deployment** — Installs vLLM on the pod, downloads the model, starts the server
5. **OA configuration** — Automatically writes endpoint URL + model to `~/.open-agents/config.json` and exports `OA_ENDPOINT` / `OA_MODEL` env vars

## Supported Models

| Model | Size | Min VRAM | GPU | Quant |
|-------|------|----------|-----|-------|
| Gemma 3 27B | 27B | 34 GB | RTX 4090 | fp16 |
| Qwen3 30B-A3B | 30B | 24 GB | RTX 4090 | fp16 |
| DeepSeek R1 Distill Qwen 32B | 32B | 40 GB | A100 | fp16 |
| GLM-4 32B | 32B | 40 GB | A100 | fp16 |
| Yi-34B | 34B | 40 GB | A100 | fp16 |
| Command-R 35B | 35B | 40 GB | A100 | fp16 |
| Llama 3 70B | 70B | 42 GB | A100 | 4bit |
| Qwen 2.5 72B | 72B | 42 GB | A100 | 4bit |
| DeepSeek R1 Distill Llama 70B | 70B | 42 GB | A100 | 4bit |
| Mixtral 8x22B | 141B | 92 GB | 2×A100 | 4bit |
| Command-R Plus 104B | 104B | 60 GB | H100 | 4bit |
| Llama 4 Scout 109B | 109B | 120 GB | 2×H100 | fp16 |
| Qwen3 235B (MoE) | 235B | 136 GB | 2×H100 | fp16 |
| Qwen3 235B 4-bit | 235B | 72 GB | H100 | 4bit |
| DeepSeek R1 | 671B | 670 GB | 8×H100 | fp16 |
| DeepSeek R1 4-bit | 671B | 180 GB | 3×H100 | 4bit |
| DeepSeek V3 | 685B | 670 GB | 8×H100 | fp16 |
| Llama 4 Maverick 400B | 400B | 420 GB | 6×H100 | fp16 |
| Qwen 2.5 1M | 1.5B+ | 160 GB | 2×H100 | 4bit |
| GLM-4 1M | 1.5B+ | 160 GB | 2×H100 | 4bit |

Run `./lium-inference.sh --list` for the full interactive table.

## Command Reference

| Command | Description |
|---------|-------------|
| `./lium-inference.sh` | Interactive model selection menu |
| `./lium-inference.sh --model NAME` | Skip menu, deploy specific model |
| `./lium-inference.sh --list` | Show all available models |
| `./lium-inference.sh --status` | Check running pod & endpoint health |
| `./lium-inference.sh --stop` | Tear down pod & stop billing |
| `./lium-inference.sh --help` | Show usage info |

## GPU Auto-Selection Logic

The script automatically maps each model to the minimum GPU it needs:

- **RTX 4090 (24 GB)** — Small models under 24 GB VRAM
- **A100 (80 GB)** — Mid-size models (32B-72B) in fp16 or 4-bit
- **H100 (80 GB)** — Large models (100B+) or multi-GPU setups

If your preferred GPU type is unavailable, the script falls back through the GPU hierarchy (H100 → A100 → A6000 → RTX 4090), checking that the total VRAM meets the model's minimum requirement.

## Open Agent Integration

After the endpoint is live, the script automatically:

1. Writes to `~/.open-agents/config.json`:
   ```json
   {
     "endpoint": "http://<pod-ip>:8000/v1",
     "model": "Qwen/Qwen3-235B-A22B"
   }
   ```

2. Exports environment variables:
   ```bash
   export OA_ENDPOINT="http://<pod-ip>:8000/v1"
   export OA_MODEL="Qwen/Qwen3-235B-A22B"
   ```

3. Persists to `~/.bashrc` for future sessions

When you launch Open Agent, it will pick up the endpoint and use your Lium-hosted model.

## Manual Testing

Once the endpoint is up, test it directly:

```bash
curl http://<pod-ip>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-235B-A22B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## State Management

The script saves state to `~/.lium-inference/state.json` so you can:

- Run `--status` to check the pod and endpoint
- Run `--stop` to tear everything down
- Resume awareness across terminal sessions

## Cost Management

- Lium bills per GPU-hour (check `lium ps` for current rates)
- Pareto-optimal nodes (★ in `lium ls`) offer best price/performance
- **Always run `--stop` when done** to avoid ongoing charges
- Smaller models on RTX 4090 are cheapest; 8×H100 for DeepSeek R1 is most expensive

## Prerequisites

| Tool | Install | Purpose |
|------|---------|---------|
| `lium` | `pip install lium-cli` | GPU pod management |
| `jq` | `sudo apt install jq` | JSON state parsing |
| `curl` | `sudo apt install curl` | Health checks |
| Lium account | `lium init && lium fund` | Authentication & billing |

## Adding New Models

Edit the `MODEL_CATALOG` associative array in the script:

```bash
MODEL_CATALOG[your-model]="org/model-repo|min_vram_gb|GPU_TYPE|gpu_count|quant|context_length"
```

Example:
```bash
MODEL_CATALOG[phi4-mini]="microsoft/Phi-4-mini-instruct|16|RTX4090|1|fp16|16384"
```

## Architecture

```
lium-inference.sh
├── Model Catalogue (23 models, VRAM/GPU mapping)
├── Interactive Picker (numbered menu)
├── GPU Auto-Selection (fallback hierarchy)
├── Pod Lifecycle (launch → wait → deploy → monitor)
├── vLLM Deployment (pip install + serve with tensor parallel)
├── Endpoint Discovery (IP detection, health polling)
├── OA Config Writer (config.json + env vars + bashrc)
└── State Management (JSON persistence, cleanup on stop)
```
