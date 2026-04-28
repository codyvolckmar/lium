#!/usr/bin/env bash
# =============================================================================
# lium-inference.sh — Spin up a Lium GPU pod with an open-source LLM endpoint
#
# Usage:
#   ./lium-inference.sh                    # Interactive model selection
#   ./lium-inference.sh --model qwen3-235b  # Skip menu, pick model directly
#   ./lium-inference.sh --list              # Just list available models
#   ./lium-inference.sh --stop              # Tear down the running pod
#   ./lium-inference.sh --status            # Check pod & endpoint status
#   ./lium-inference.sh -f -m llama3.1-8b   # Fire-and-forget (writes .lium-endpoint.json)
#   ./lium-inference.sh -f -m qwen3-32b -t 4h  # Fire-and-forget with 4h TTL
#
# Configuration:
#   Set APIKEY variable below or export LIUM_API_KEY environment variable
#   before running this script. This is required when running on a machine
#   without lium CLI configured.
#
# Prerequisites:
#   - lium CLI installed & configured (pip install lium-cli && lium init)
#   - curl, jq
#   - Funded Lium account (lium fund)
#
# After launch, the script prints an OpenAI-compatible endpoint URL that you
# can plug into Open Agent's config as a custom inference endpoint.
# =============================================================================

set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── API Key Configuration ───────────────────────────────────────────────────
# Set your Lium API key here or export LIUM_API_KEY environment variable
# The script will use this key to authenticate with Lium's API
APIKEY="${LIUM_API_KEY:-sk_BvT7nx59eHNgj37hbSOQIGRV0vC9xI-I9_b2LfX3f3g}"

# If APIKEY is set, export it for lium CLI to use
if [[ -n "$APIKEY" ]]; then
    export LIUM_API_KEY="$APIKEY"
fi

# ── Model Catalogue ─────────────────────────────────────────────────────────
# HIGH-PERFORMANCE TEXT-ONLY LLMs — No vision/multimodal models
# Each entry: "hf_repo|min_vram_gb|gpu_type|gpu_count|quant_note|context_length"
# VRAM estimates assume FP16/BF16 for full-precision, GPTQ/AWQ 4-bit for quantized.
# Adjusted for KV cache overhead (~20% headroom).
# 
# GPU TIER GUIDE:
#   RTX3090/RTX4090 (24GB) → Small models, efficient MoEs
#   A100 (80GB)           → Mid-size models, 70B quantized
#   H100 (80GB)           → Large models, multi-GPU setups
# ─────────────────────────────────────────────────────────────────────────────

declare -A MODEL_CATALOG

# ═════════════════════════════════════════════════════════════════════════════
# TIER 1: LIGHTWEIGHT (≤10GB VRAM) — Fast inference, single RTX3090/4090
# ═════════════════════════════════════════════════════════════════════════════

# Qwen3.5 Series — Excellent reasoning, tool use, multilingual
MODEL_CATALOG[qwen3.5-0.8b]="Qwen/Qwen3.5-0.8B|4|RTX3090|1|fp16|32768"
MODEL_CATALOG[qwen3.5-2b]="Qwen/Qwen3.5-2B|6|RTX3090|1|fp16|32768"
MODEL_CATALOG[qwen3.5-4b]="Qwen/Qwen3.5-4B|10|RTX3090|1|fp16|32768"

# ═════════════════════════════════════════════════════════════════════════════
# TIER 2: COMPACT (10-24GB VRAM) — Single RTX3090/4090
# ═════════════════════════════════════════════════════════════════════════════

# Qwen3.5 Mid-size
MODEL_CATALOG[qwen3.5-9b]="Qwen/Qwen3.5-9B|20|RTX3090|1|fp16|16384"
MODEL_CATALOG[qwen3.5-35b-a3b]="Qwen/Qwen3.5-35B-A3B|24|RTX3090|1|fp16|8192"

# GLM-4/5 Series — Strong general performance
MODEL_CATALOG[glm4-9b]="zai-org/GLM-4-9B-0414|20|RTX3090|1|fp16|8192"
MODEL_CATALOG[glm5-9b]="zai-org/GLM-5|20|RTX3090|1|fp16|8192"
# MODEL_CATALOG[glm5.1-9b] removed - model does not exist on HuggingFace

# Qwen3 MoE — Efficient sparse architecture
MODEL_CATALOG[qwen3-30b-a3b]="Qwen/Qwen3-30B-A3B|24|RTX3090|1|fp16|8192"

# ═════════════════════════════════════════════════════════════════════════════
# TIER 3: STANDARD (24-48GB VRAM) — Single A100 or RTX4090 with quantization
# ═════════════════════════════════════════════════════════════════════════════

# DeepSeek R1 Distill — Reasoning specialists
MODEL_CATALOG[deepseek-r1-distill-qwen-32b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|40|A100|1|fp16|32768"
MODEL_CATALOG[deepseek-r1-distill-llama-70b]="deepseek-ai/DeepSeek-R1-Distill-Llama-70B|42|A100|1|4bit|8192"

# GLM-4 Mid-size
MODEL_CATALOG[glm4-32b]="zai-org/GLM-4-32B-0414|40|A100|1|fp16|32768"

# Yi Series — Strong context handling
MODEL_CATALOG[yi-34b]="01-ai/Yi-34B|40|A100|1|fp16|4096"

# Command-R — Excellent RAG & tool use
MODEL_CATALOG[command-r-35b]="CohereForAI/c4ai-command-r-v01|40|A100|1|fp16|131072"

# Qwen3.5 Large MoE
MODEL_CATALOG[qwen3.5-27b]="Qwen/Qwen3.5-27B|60|A100|1|fp16|32768"

# ═════════════════════════════════════════════════════════════════════════════
# TIER 4: HEAVYWEIGHT (48-160GB VRAM) — Single H100 or multi-GPU A100
# ═════════════════════════════════════════════════════════════════════════════

# Llama 3 70B — Industry standard
MODEL_CATALOG[llama3-70b]="meta-llama/Meta-Llama-3-70B|42|A100|1|4bit|8192"
MODEL_CATALOG[llama3-70b-fp16]="meta-llama/Meta-Llama-3-70B|148|A100|2|fp16|8192"

# Qwen2.5 72B — Long context excellence
MODEL_CATALOG[qwen2.5-72b]="Qwen/Qwen2.5-72B-Instruct|42|A100|1|4bit|131072"
MODEL_CATALOG[qwen2.5-72b-fp16]="Qwen/Qwen2.5-72B-Instruct|148|A100|2|fp16|131072"

# Mixtral 8x22B — MoE efficiency at scale
MODEL_CATALOG[mixtral-8x22b]="mistralai/Mixtral-8x22B-v0.1|92|A100|2|4bit|65536"

# Command-R Plus — Enterprise-grade RAG
MODEL_CATALOG[command-r-plus-104b]="CohereForAI/c4ai-command-r-plus|60|H100|1|4bit|131072"

# Qwen3.5 Large MoE
MODEL_CATALOG[qwen3.5-122b-a10b]="Qwen/Qwen3.5-122B-A10B|80|H100|1|fp16|32768"
MODEL_CATALOG[qwen3.5-122b-a10b-4bit]="Qwen/Qwen3.5-122B-A10B|40|A100|1|4bit|32768"

# ═════════════════════════════════════════════════════════════════════════════
# TIER 5: FRONTIER (160GB+ VRAM) — Multi-GPU H100 clusters
# ═════════════════════════════════════════════════════════════════════════════

# Qwen3 Massive MoE
MODEL_CATALOG[qwen3-235b]="Qwen/Qwen3-235B-A22B|136|H100|2|fp16|32768"
MODEL_CATALOG[qwen3-235b-4bit]="Qwen/Qwen3-235B-A22B|72|H100|1|4bit|32768"

# Qwen3.5 Massive MoE
MODEL_CATALOG[qwen3.5-397b-a17b]="Qwen/Qwen3.5-397B-A17B|240|H100|3|fp16|32768"
MODEL_CATALOG[qwen3.5-397b-a17b-4bit]="Qwen/Qwen3.5-397B-A17B|80|H100|1|4bit|32768"

# DeepSeek R1/V3 — Reasoning powerhouses
MODEL_CATALOG[deepseek-r1]="deepseek-ai/DeepSeek-R1|670|H100|8|fp16|131072"
MODEL_CATALOG[deepseek-r1-4bit]="deepseek-ai/DeepSeek-R1|180|H100|3|4bit|131072"
MODEL_CATALOG[deepseek-v3]="deepseek-ai/DeepSeek-V3|670|H100|8|fp16|131072"
MODEL_CATALOG[deepseek-v3-4bit]="deepseek-ai/DeepSeek-V3|180|H100|3|4bit|131072"

# GLM-5.1 — Massive scale
MODEL_CATALOG[glm5.1]="zai-org/GLM-5.1-0414|800|H100|10|fp16|131072"
MODEL_CATALOG[glm5.1-4bit]="zai-org/GLM-5.1-0414|250|H100|4|4bit|131072"

# ═════════════════════════════════════════════════════════════════════════════
# SPECIALTY: LONG CONTEXT (1M+ tokens)
# ═════════════════════════════════════════════════════════════════════════════

MODEL_CATALOG[qwen2.5-1m]="Qwen/Qwen2.5-1M|160|H100|2|4bit|1000000"
MODEL_CATALOG[glm4-1m]="zai-org/GLM-4-1M-0414|160|H100|2|4bit|1000000"

# ── State file ───────────────────────────────────────────────────────────────
STATE_DIR="${HOME}/.lium-inference"
STATE_FILE="${STATE_DIR}/state.json"
ENDPOINT_FILE=".lium-endpoint.json"
FIRE_AND_FORGET=false
API_KEY=""
mkdir -p "${STATE_DIR}"

save_state() {
    cat > "${STATE_FILE}" <<EOF
{
  "pod_name": "${POD_NAME:-}",
  "model": "${SELECTED_MODEL:-}",
  "hf_repo": "${HF_REPO:-}",
  "gpu_type": "${GPU_TYPE:-}",
  "gpu_count": ${GPU_COUNT:-0},
  "endpoint_url": "${ENDPOINT_URL:-}",
  "port": ${ENDPOINT_PORT:-8000},
  "launched_at": "$(date -Iseconds)"
}
EOF
}

load_state() {
    if [[ -f "${STATE_FILE}" ]]; then
        POD_NAME=$(jq -r '.pod_name' "${STATE_FILE}")
        SELECTED_MODEL=$(jq -r '.model' "${STATE_FILE}")
        HF_REPO=$(jq -r '.hf_repo' "${STATE_FILE}")
        GPU_TYPE=$(jq -r '.gpu_type' "${STATE_FILE}")
        GPU_COUNT=$(jq -r '.gpu_count' "${STATE_FILE}")
        ENDPOINT_URL=$(jq -r '.endpoint_url' "${STATE_FILE}")
        ENDPOINT_PORT=$(jq -r '.port' "${STATE_FILE}")
    fi
}

# ── Write endpoint file (for fire-and-forget consumers like OpenClaw) ────────
write_endpoint_file() {
    cat > "${ENDPOINT_FILE}" <<EOF
{
  "status": "verified",
  "endpoint": "${ENDPOINT_URL}/v1",
  "api_key": "${API_KEY}",
  "model": "${HF_REPO}",
  "model_key": "${SELECTED_MODEL}",
  "pod_id": "${POD_NAME}",
  "gpu": "${GPU_TYPE} x${GPU_COUNT}",
  "port": "${ENDPOINT_PORT}",
  "ttl": "${POD_TTL:-none}",
  "created_at": "$(date -Iseconds)"
}
EOF
    ok "Endpoint file written to ${ENDPOINT_FILE}"
    info "  endpoint: ${ENDPOINT_URL}/v1"
    info "  api_key:  ${API_KEY}"
    info "  model:    ${HF_REPO}"
}

# ── Print pending endpoint info (provisioning in progress) ───────────────────
print_endpoint_info_pending() {
    cat > "${ENDPOINT_FILE}" << EOF
{
  "status": "provisioning",
  "message": "vLLM is starting in background. Poll this file or check pod status.",
  "endpoint": "pending",
  "api_key": "${API_KEY}",
  "model": "${HF_REPO}",
  "model_key": "${SELECTED_MODEL}",
  "pod_id": "${POD_NAME}",
  "gpu": "${GPU_TYPE} x${GPU_COUNT}",
  "port": "${POD_PORT}",
  "ttl": "${POD_TTL:-none}",
  "created_at": "$(date -Iseconds)",
  "check_status_cmd": "lium exec ${POD_NAME} 'cat /root/provision-status.json'",
  "tail_logs_cmd": "lium exec ${POD_NAME} 'tail -f /root/vllm.log'"
}
EOF
    ok "Pending endpoint file written to ${ENDPOINT_FILE}"
    info "  status: provisioning (vLLM starting in background)"
    info "  pod:    ${POD_NAME}"
    info "  model:  ${HF_REPO}"
    info ""
    info "Check provisioning status:"
    info "  lium exec ${POD_NAME} 'cat /root/provision-status.json'"
    info "  lium exec ${POD_NAME} 'tail -f /root/vllm.log'"
    info "  cat ${ENDPOINT_FILE}"
}

# ── Prerequisites check ─────────────────────────────────────────────────────
check_prereqs() {
    local missing=0
    for cmd in lium jq curl; do
        if ! command -v "$cmd" &>/dev/null; then
            err "'$cmd' not found. Please install it first."
            missing=1
        fi
    done
    if [[ "$missing" -eq 1 ]]; then
        echo ""
        info "Install missing tools:"
        info "  pip install lium-cli"
        info "  sudo apt install jq curl"
        exit 1
    fi
}

# ── Parse lium ls output and extract available GPUs ───────────────────────────
# Global associative array to store available GPU counts
declare -A AVAILABLE_GPUS
# Store executor info: EXECUTOR_INFO[id]="gpu_type:price"
declare -A EXECUTOR_INFO
# Store cheapest executor per GPU type: CHEAPEST_EXECUTOR[gpu_type]=id
declare -A CHEAPEST_EXECUTOR
# Store ALL executors per GPU type: ALL_EXECUTORS_BY_GPU[gpu_type]="id1 id2 id3 ..."
declare -A ALL_EXECUTORS_BY_GPU
# Selected executor (set by pick_executor)
SELECTED_EXECUTOR=""
# Track max GPUs available on a single executor (pod) per GPU type
# This is critical for multi-GPU models — they need all GPUs on the SAME pod
declare -A MAX_GPUS_PER_EXECUTOR
declare -A GPU_VRAM_MAP=(
    ["B300"]=269
    ["B200"]=192
    ["H200"]=140
    ["H100"]=80
    ["A100"]=80
    ["Edition"]=96
    ["A6000"]=48
    ["RTX6000"]=48
    ["RTX5090"]=32
    ["V100"]=32
    ["RTX4090"]=24
    ["RTX3090"]=24
    ["L40"]=48
    ["L4"]=24
    ["A10"]=24
    ["A10G"]=24
    ["T4"]=16
)

parse_lium_ls() {
    AVAILABLE_GPUS=()
    EXECUTOR_INFO=()
    CHEAPEST_EXECUTOR=()
    ALL_EXECUTORS_BY_GPU=()
    MAX_GPUS_PER_EXECUTOR=()
    local lium_output
    # Use --sort price_gpu to get cheapest GPUs first
    lium_output=$(lium ls --sort price_gpu 2>/dev/null) || return 1
    
    if [[ -z "$lium_output" ]]; then
        return 1
    fi
    
    # Parse lium ls output - format varies, try multiple patterns
    # Output is sorted by price_gpu (cheapest first)
    # Expected columns: ID, GPU, Price, Location (varies by lium version)
    
    while IFS= read -r line; do
        # Skip header lines and empty lines
        [[ -z "$line" ]] && continue
        [[ "$line" =~ ^(GPU|Name|Type|ID|Available|---|\||\+) ]] && continue
        
        # Try to extract GPU type, executor ID, and price from various formats
        local gpu_type=""
        local executor_id=""
        local price=""
        local count=0
        
        # Normalize line to uppercase for matching
        local line_upper="${line^^}"
        # Replace Unicode × (U+00D7, UTF-8 C3 97) with ASCII X for regex compatibility
        line_upper="${line_upper//×/X}"
        
        # Try to extract executor ID (usually first column, alphanumeric)
        # Format: "executor_id  GPU_TYPE  price  ..." or "ID | GPU | Price | ..."
        if [[ "$line" =~ ^[[:space:]]*([a-zA-Z0-9_-]+)[[:space:]]+ ]]; then
            executor_id="${BASH_REMATCH[1]}"
        fi
        
        # Try to extract price (looks like $X.XX or X.XX format)
        if [[ "$line" =~ \$?([0-9]+\.[0-9]+) ]]; then
            price="${BASH_REMATCH[1]}"
        fi
        
        # Format: "2 x H100" or "H100 (2 available)" - case insensitive, handles spaces
        # Also match standalone numbers like "5090", "4090", "3090", "6000" without RTX prefix
        if [[ "$line_upper" =~ ([0-9]+)[[:space:]]*X[[:space:]]*(B300|B200|H200|H100|A100|A6000|RTX.?5090|RTX.?4090|RTX.?3090|RTX.?6000|L40S?|Edition|5090|4090|3090|6000|V100|T4|L4|A10|A10G) ]]; then
            count="${BASH_REMATCH[1]}"
            # Normalize GPU type (remove spaces, standardize naming)
            gpu_type="${BASH_REMATCH[2]}"
            gpu_type="${gpu_type// /}"  # Remove any spaces
            # Add RTX prefix if missing for consumer GPUs
            [[ "$gpu_type" =~ ^(5090|4090|3090|6000)$ ]] && gpu_type="RTX$gpu_type"
            # Normalize L40S -> L40 for VRAM/rank lookups (same VRAM)
            [[ "$gpu_type" == "L40S" ]] && gpu_type="L40"
        # Format: "H100" with count in another column - case insensitive
        elif [[ "$line_upper" =~ (B300|B200|H200|H100|A100|A6000|RTX.?5090|RTX.?4090|RTX.?3090|RTX.?6000|L40S?|Edition|5090|4090|3090|6000|V100|T4|L4|A10|A10G) ]]; then
            gpu_type="${BASH_REMATCH[1]}"
            gpu_type="${gpu_type// /}"  # Remove any spaces
            # Add RTX prefix if missing for consumer GPUs
            [[ "$gpu_type" =~ ^(5090|4090|3090|6000)$ ]] && gpu_type="RTX$gpu_type"
            # Normalize L40S -> L40 for VRAM/rank lookups (same VRAM)
            [[ "$gpu_type" == "L40S" ]] && gpu_type="L40"
            # Look for a number in the line
            if [[ "$line" =~ ([0-9]+)[[:space:]]*(available|pods?|GPUs?|$) ]]; then
                count="${BASH_REMATCH[1]}"
            else
                count=1  # At least one available
            fi
        fi
        
        if [[ -n "$gpu_type" && "$count" -gt 0 ]]; then
            # Accumulate counts for same GPU type
            if [[ -n "${AVAILABLE_GPUS[$gpu_type]:-}" ]]; then
                AVAILABLE_GPUS[$gpu_type]=$((AVAILABLE_GPUS[$gpu_type] + count))
            else
                AVAILABLE_GPUS[$gpu_type]=$count
            fi
            
            # Track max GPUs on a single executor (pod) — critical for multi-GPU models
            if [[ "$count" -gt "${MAX_GPUS_PER_EXECUTOR[$gpu_type]:-0}" ]]; then
                MAX_GPUS_PER_EXECUTOR[$gpu_type]=$count
            fi
            
            # Store executor info if we have ID and price
            if [[ -n "$executor_id" && -n "$price" ]]; then
                EXECUTOR_INFO[$executor_id]="${gpu_type}:${price}:${count}"
                # Add to ALL executors list for this GPU type
                if [[ -n "${ALL_EXECUTORS_BY_GPU[$gpu_type]:-}" ]]; then
                    ALL_EXECUTORS_BY_GPU[$gpu_type]="${ALL_EXECUTORS_BY_GPU[$gpu_type]} $executor_id"
                else
                    ALL_EXECUTORS_BY_GPU[$gpu_type]="$executor_id"
                fi
                # Track cheapest executor for each GPU type
                # Prefer single-GPU pods over multi-GPU pods (cheaper overall cost)
                local current_best="${CHEAPEST_EXECUTOR[$gpu_type]:-}"
                if [[ -z "$current_best" ]]; then
                    # First executor for this GPU type
                    CHEAPEST_EXECUTOR[$gpu_type]="$executor_id"
                else
                    # Compare GPU counts - prefer fewer GPUs (single GPU is cheapest)
                    local current_info="${EXECUTOR_INFO[$current_best]:-}"
                    local current_count="${current_info##*:}"
                    if [[ "$count" -lt "$current_count" ]]; then
                        # This executor has fewer GPUs, prefer it
                        CHEAPEST_EXECUTOR[$gpu_type]="$executor_id"
                    elif [[ "$count" -eq "$current_count" ]] && [[ "$(echo "$price < ${current_info#*:}" | bc -l 2>/dev/null || echo 0)" -eq 1 ]]; then
                        # Same GPU count but cheaper price
                        CHEAPEST_EXECUTOR[$gpu_type]="$executor_id"
                    fi
                fi
            fi
        fi
    done <<< "$lium_output"
    
    return 0
}

# ── GPU Hierarchy (better GPUs can substitute for lesser ones) ───────────────
# Higher rank = more powerful. Used to determine if a GPU can substitute another.
declare -A GPU_RANK=(
    ["B300"]=115
    ["B200"]=110
    ["H200"]=105
    ["H100"]=100
    ["A100"]=90
    ["Edition"]=85
    ["A6000"]=70
    ["RTX6000"]=70
    ["L40"]=68
    ["RTX5090"]=65
    ["RTX4090"]=60
    ["RTX3090"]=55
    ["L4"]=50
    ["A10"]=50
    ["A10G"]=50
    ["V100"]=45
    ["T4"]=40
)

# Check if available GPU can substitute for required GPU type
gpu_can_substitute() {
    local avail_gpu="$1"
    local req_gpu="$2"
    
    # Exact match always works
    [[ "$avail_gpu" == "$req_gpu" ]] && return 0
    
    # Check hierarchy - available GPU must have rank >= required GPU
    local avail_rank="${GPU_RANK[$avail_gpu]:-0}"
    local req_rank="${GPU_RANK[$req_gpu]:-0}"
    
    [[ "$avail_rank" -ge "$req_rank" ]] && return 0
    return 1
}

# ── Check if a model can run on available GPUs ────────────────────────────────
can_run_model() {
    local model_key="$1"
    local entry="${MODEL_CATALOG[$model_key]}"
    
    if [[ -z "$entry" ]]; then
        return 1
    fi
    
    IFS='|' read -r hf_repo min_vram gpu_type gpu_count quant ctx <<< "$entry"
    
    # Check each available GPU type
    for avail_gpu in "${!AVAILABLE_GPUS[@]}"; do
        local avail_count="${AVAILABLE_GPUS[$avail_gpu]}"
        local max_per_pod="${MAX_GPUS_PER_EXECUTOR[$avail_gpu]:-$avail_count}"
        local gpu_vram="${GPU_VRAM_MAP[$avail_gpu]:-0}"
        
        # First check: GPU type compatibility (hierarchy-based substitution)
        if gpu_can_substitute "$avail_gpu" "$gpu_type"; then
            # Check if a single pod has enough GPUs AND total VRAM
            local total_vram=$((gpu_vram * max_per_pod))
            if [[ "$max_per_pod" -ge "$gpu_count" && "$total_vram" -ge "$min_vram" ]]; then
                return 0
            fi
            
            # Check if a single GPU of this type has enough VRAM (single-GPU models)
            if [[ "$gpu_vram" -ge "$min_vram" && "$max_per_pod" -ge 1 ]]; then
                return 0
            fi
        fi
        
        # VRAM-based fallback: calculate how many GPUs of this type we'd need
        # to satisfy the model's VRAM requirement, then check if a single pod has that many
        if [[ "$gpu_vram" -gt 0 ]]; then
            local needed_count=$(( (min_vram + gpu_vram - 1) / gpu_vram ))  # ceil division
            local total_vram=$((gpu_vram * needed_count))
            if [[ "$max_per_pod" -ge "$needed_count" && "$total_vram" -ge "$min_vram" ]]; then
                return 0
            fi
        fi
    done
    
    return 1
}

# ── Show available GPUs (parsed) ─────────────────────────────────────────────
show_available_gpus() {
    echo ""
    info "Checking available GPUs on Lium..."
    echo ""
    
    if parse_lium_ls; then
        if [[ ${#AVAILABLE_GPUS[@]} -eq 0 ]]; then
            warn "No GPUs detected in lium ls output."
            warn "Raw output:"
            lium ls 2>/dev/null || true
        else
            echo -e "${BOLD}Available GPUs:${NC}"
            for gpu in "${!AVAILABLE_GPUS[@]}"; do
                local vram="${GPU_VRAM_MAP[$gpu]:-?}"
                local count="${AVAILABLE_GPUS[$gpu]}"
                local max_pod="${MAX_GPUS_PER_EXECUTOR[$gpu]:-$count}"
                if [[ "$max_pod" -ge 2 ]]; then
                    printf "  ${GREEN}✓${NC} %s x%d (%s GB each, %s GB total) [up to %d per pod = %s GB]\n" "$gpu" "$count" "$vram" "$((vram * count))" "$max_pod" "$((vram * max_pod))"
                else
                    printf "  ${GREEN}✓${NC} %s x%d (%s GB each, %s GB total)\n" "$gpu" "$count" "$vram" "$((vram * count))"
                fi
            done | sort
            echo ""
        fi
    else
        warn "Could not fetch GPU list. Make sure 'lium' is configured (run 'lium init')."
    fi
}

# ── List models ──────────────────────────────────────────────────────────────
list_models() {
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║                    Open-Source LLM Catalogue for Lium                       ║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════════════════════════════════════╣${NC}"
    printf "${BOLD}║ %-28s │ %-8s │ %-8s │ %-4s │ %-6s ║${NC}\n" "Model" "Min VRAM" "GPU" "Cnt" "Quant"
    echo -e "${BOLD}╠══════════════════════════════════════════════════════════════════════════════╣${NC}"

    # Sort by VRAM (field 2)
    for key in $(echo "${!MODEL_CATALOG[@]}" | tr ' ' '\n' | sort); do
        IFS='|' read -r hf_repo min_vram gpu_type gpu_count quant ctx <<< "${MODEL_CATALOG[$key]}"
        printf "║ %-28s │ %5s GB │ %-8s │ %-4s │ %-6s ║\n" "$key" "$min_vram" "$gpu_type" "$gpu_count" "$quant"
    done

    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
}

# ── Interactive model picker ─────────────────────────────────────────────────
pick_model() {
    # Parse available GPUs first
    parse_lium_ls 2>/dev/null || true
    
    local keys=($(echo "${!MODEL_CATALOG[@]}" | tr ' ' '\n' | sort))
    local viable_keys=()
    local i=1

    echo ""
    
    # Check if we have GPU availability data
    if [[ ${#AVAILABLE_GPUS[@]} -gt 0 ]]; then
        echo -e "${BOLD}Select an LLM to deploy (filtered by available GPUs):${NC}"
        echo ""
        
        # Separate viable and non-viable models
        local viable_models=()
        local blocked_models=()
        
        for key in "${keys[@]}"; do
            if can_run_model "$key"; then
                viable_models+=("$key")
            else
                blocked_models+=("$key")
            fi
        done
        
        # Show viable models first
        if [[ ${#viable_models[@]} -gt 0 ]]; then
            echo -e "${GREEN}Available models (can run on current GPUs):${NC}"
            for key in "${viable_models[@]}"; do
                IFS='|' read -r hf_repo min_vram gpu_type gpu_count quant ctx <<< "${MODEL_CATALOG[$key]}"
                printf "  ${CYAN}%2d${NC}) %-30s ${YELLOW}[%s GB | %sx%s | %s]${NC}\n" "$i" "$key" "$min_vram" "$gpu_type" "$gpu_count" "$quant"
                viable_keys+=("$key")
                ((i++))
            done
            echo ""
        fi
        
        # Show blocked models (dimmed)
        if [[ ${#blocked_models[@]} -gt 0 ]]; then
            echo -e "${RED}Unavailable models (insufficient GPU resources):${NC}"
            for key in "${blocked_models[@]}"; do
                IFS='|' read -r hf_repo min_vram gpu_type gpu_count quant ctx <<< "${MODEL_CATALOG[$key]}"
                printf "  ${RED} --${NC}) %-30s ${RED}[%s GB | %sx%s | %s]${NC}\n" "$key" "$min_vram" "$gpu_type" "$gpu_count" "$quant"
            done
            echo ""
        fi
        
        if [[ ${#viable_keys[@]} -eq 0 ]]; then
            err "No models can run on currently available GPUs."
            err "Try again later or check 'lium ls' for available hardware."
            exit 1
        fi
    else
        # Fallback: show all models if we couldn't parse GPU availability
        echo -e "${BOLD}Select an LLM to deploy:${NC}"
        echo ""
        for key in "${keys[@]}"; do
            IFS='|' read -r hf_repo min_vram gpu_type gpu_count quant ctx <<< "${MODEL_CATALOG[$key]}"
            printf "  ${CYAN}%2d${NC}) %-30s ${YELLOW}[%s GB | %sx%s | %s]${NC}\n" "$i" "$key" "$min_vram" "$gpu_type" "$gpu_count" "$quant"
            viable_keys+=("$key")
            ((i++))
        done
        echo ""
    fi

    read -rp "Enter number [1-$((i-1))]: " choice

    if [[ "$choice" -lt 1 || "$choice" -gt $((i-1)) ]] 2>/dev/null; then
        err "Invalid selection."
        exit 1
    fi

    SELECTED_MODEL="${viable_keys[$((choice-1))]}"
}

# ── Parse model entry ───────────────────────────────────────────────────────
parse_model() {
    local entry="${MODEL_CATALOG[$SELECTED_MODEL]}"
    if [[ -z "$entry" ]]; then
        err "Unknown model: $SELECTED_MODEL"
        err "Use --list to see available models."
        exit 1
    fi
    IFS='|' read -r HF_REPO MIN_VRAM GPU_TYPE GPU_COUNT QUANT CTX_LEN <<< "$entry"
    info "Model:       ${BOLD}${SELECTED_MODEL}${NC}"
    info "HF Repo:     ${HF_REPO}"
    info "Min VRAM:    ${MIN_VRAM} GB"
    info "GPU:         ${GPU_TYPE} x${GPU_COUNT}"
    info "Quant:       ${QUANT}"
    info "Context:     ${CTX_LEN} tokens"
}

# ── Find best GPU on Lium ───────────────────────────────────────────────────
find_gpu() {
    info "Searching Lium for available ${GPU_TYPE} GPUs (x${GPU_COUNT})..."

    # Check if the required GPU type is available (using pre-parsed data)
    local gpu_available="${AVAILABLE_GPUS[$GPU_TYPE]:-0}"
    local gpu_list=""

    if [[ "$gpu_available" -ge "$GPU_COUNT" ]]; then
        # We have enough of the required GPU type — but check per-pod too
        local max_per_pod="${MAX_GPUS_PER_EXECUTOR[$GPU_TYPE]:-$gpu_available}"
        if [[ "$max_per_pod" -ge "$GPU_COUNT" ]]; then
            gpu_list=$(lium ls 2>/dev/null | grep -i "${GPU_TYPE}" || true)
        else
            warn "Have ${gpu_available}x ${GPU_TYPE} total but max ${max_per_pod} per pod (need ${GPU_COUNT}). Trying alternatives..."
            gpu_available=0  # Force fallback path
        fi
    fi
    
    if [[ "$gpu_available" -lt "$GPU_COUNT" ]]; then
        warn "No ${GPU_TYPE} GPUs currently available (need ${GPU_COUNT}, have ${gpu_available}). Trying alternatives..."

        # Fallback GPU hierarchy: B200 > H100 > A100 80GB > A6000 > L40 > RTX6000 > RTX5090 > RTX4090 > RTX3090
        local -a FALLBACKS=()
        if [[ "${TEST_MODE:-}" == true ]]; then
            # Test mode: only use consumer GPUs
            FALLBACKS=("RTX4090" "RTX3090")
        else
            case "$GPU_TYPE" in
                B200)    FALLBACKS=("H100" "A100" "A6000" "L40" "RTX6000" "RTX5090" "RTX4090" "RTX3090") ;;
                H100)    FALLBACKS=("B200" "A100" "A6000" "L40" "RTX6000" "RTX5090" "RTX4090" "RTX3090") ;;
                A100)    FALLBACKS=("B200" "H100" "A6000" "L40" "RTX6000" "RTX5090" "RTX4090" "RTX3090") ;;
                A6000)   FALLBACKS=("B200" "H100" "A100" "L40" "RTX6000" "RTX5090" "RTX4090" "RTX3090") ;;
                L40)     FALLBACKS=("B200" "H100" "A100" "A6000" "RTX6000" "RTX5090" "RTX4090" "RTX3090") ;;
                RTX6000) FALLBACKS=("B200" "H100" "A100" "A6000" "L40" "RTX5090" "RTX4090" "RTX3090") ;;
                RTX5090) FALLBACKS=("B200" "H100" "A100" "A6000" "L40" "RTX6000" "RTX4090" "RTX3090") ;;
                RTX4090) FALLBACKS=("B200" "H100" "A100" "A6000" "L40" "RTX6000" "RTX5090" "RTX3090") ;;
                RTX3090) FALLBACKS=("B200" "H100" "A100" "A6000" "L40" "RTX6000" "RTX5090" "RTX4090") ;;
                *)       FALLBACKS=("B200" "H100" "A100" "A6000" "L40" "RTX6000" "RTX5090" "RTX4090" "RTX3090") ;;
            esac
        fi

        for fb in "${FALLBACKS[@]}"; do
            local fb_count="${AVAILABLE_GPUS[$fb]:-0}"
            local fb_max_per_pod="${MAX_GPUS_PER_EXECUTOR[$fb]:-$fb_count}"
            info "  Trying ${fb}... (${fb_count} total, ${fb_max_per_pod} per pod)"
            
            # Calculate how many of this fallback GPU type we need
            local fb_vram="${GPU_VRAM_MAP[$fb]:-0}"
            if [[ "$fb_vram" -eq 0 ]]; then
                continue
            fi
            local needed_count=$(( (MIN_VRAM + fb_vram - 1) / fb_vram ))  # ceil division
            if [[ "$needed_count" -lt 1 ]]; then
                needed_count=1
            fi
            
            if [[ "$fb_max_per_pod" -ge "$needed_count" ]]; then
                local total_vram=$((fb_vram * needed_count))
                ok "Found ${fb} GPUs with sufficient VRAM (${needed_count}x${fb_vram}GB = ${total_vram} GB >= ${MIN_VRAM} GB)"
                GPU_TYPE="$fb"
                GPU_COUNT="$needed_count"
                gpu_list=$(lium ls 2>/dev/null | grep -i "${fb}" || true)
                break
            else
                local total_vram=$((fb_vram * fb_max_per_pod))
                warn "  ${fb} max ${fb_max_per_pod}/pod, need ${needed_count} (${total_vram} GB < ${MIN_VRAM} GB needed)"
            fi
        done

        if [[ -z "$gpu_list" ]]; then
            err "No suitable GPUs available on Lium right now."
            err "Try again later or choose a smaller model."
            exit 1
        fi
    fi

    echo ""
    echo "$gpu_list"
    echo ""
    info "Available ${GPU_TYPE} GPUs shown above."
}

# ── Pick executor from available pods ───────────────────────────────────────
pick_executor() {
    # Get all executors that can run this model
    local -a compatible_executors=()
    local -a executor_ids=()
    
    # Find executors with compatible GPU types (using hierarchy)
    for avail_gpu in "${!ALL_EXECUTORS_BY_GPU[@]}"; do
        if gpu_can_substitute "$avail_gpu" "$GPU_TYPE"; then
            for exec_id in ${ALL_EXECUTORS_BY_GPU[$avail_gpu]}; do
                local exec_info="${EXECUTOR_INFO[$exec_id]}"
                local exec_gpu="${exec_info%%:*}"
                local exec_price="${exec_info#*:}"
                exec_price="${exec_price%:*}"
                local exec_count="${exec_info##*:}"
                
                # Check if this executor has enough GPUs
                if [[ "$exec_count" -ge "$GPU_COUNT" ]]; then
                    # In test mode, only allow RTX3090/RTX4090 with single GPU
                    if [[ "${TEST_MODE:-}" == true ]]; then
                        if [[ "$exec_gpu" == "RTX3090" || "$exec_gpu" == "RTX4090" ]] && [[ "$exec_count" -eq 1 ]]; then
                            compatible_executors+=("$exec_id|$exec_gpu|$exec_price|$exec_count")
                            executor_ids+=("$exec_id")
                        fi
                    else
                        compatible_executors+=("$exec_id|$exec_gpu|$exec_price|$exec_count")
                        executor_ids+=("$exec_id")
                    fi
                fi
            done
        fi
    done
    
    if [[ ${#compatible_executors[@]} -eq 0 ]]; then
        err "No compatible executors found for ${GPU_TYPE} x${GPU_COUNT}"
        exit 1
    fi
    
    # Sort by price (cheapest first)
    IFS=$'\n' sorted=($(sort -t'|' -k3 -n <<<"${compatible_executors[*]}")); unset IFS
    compatible_executors=("${sorted[@]}")
    
    echo ""
    echo -e "${BOLD}Select an executor (pod) to run on:${NC}"
    echo ""
    
    local i=1
    for entry in "${compatible_executors[@]}"; do
        IFS='|' read -r exec_id exec_gpu exec_price exec_count <<< "$entry"
        printf "  ${CYAN}%2d${NC}) %-25s ${YELLOW}[%s x%d | $%s/hr]${NC}\n" "$i" "$exec_id" "$exec_gpu" "$exec_count" "$exec_price"
        ((i++))
    done
    echo ""
    
    read -rp "Enter number [1-$((i-1)), default=1 (cheapest)]: " exec_choice
    exec_choice="${exec_choice:-1}"
    
    if [[ "$exec_choice" -lt 1 || "$exec_choice" -gt $((i-1)) ]] 2>/dev/null; then
        err "Invalid selection."
        exit 1
    fi
    
    # Get selected executor
    local selected_entry="${compatible_executors[$((exec_choice-1))]}"
    SELECTED_EXECUTOR="${selected_entry%%|*}"
    
    # Update GPU_TYPE and GPU_COUNT from selected executor
    IFS='|' read -r _ exec_gpu _ exec_count <<< "$selected_entry"
    GPU_TYPE="$exec_gpu"
    GPU_COUNT="$exec_count"
    
    ok "Selected executor: ${SELECTED_EXECUTOR} (${GPU_TYPE} x${GPU_COUNT})"
}

# ── Ask for duration ────────────────────────────────────────────────────────
ask_duration() {
    echo ""
    echo -e "${BOLD}${YELLOW}⏱  COST CONTROL: How long should the GPU run?${NC}"
    echo -e "${YELLOW}This auto-terminates the pod to avoid surprise invoices!${NC}"
    echo ""
    echo "  1) 1 hour   (quick test)"
    echo "  2) 2 hours  (short session)"
    echo "  3) 4 hours  (half day)"
    echo "  4) 8 hours  (full day)"
    echo "  5) 24 hours (long run)"
    echo "  6) Custom duration"
    echo "  7) No limit (⚠ manual stop required)"
    echo ""
    read -rp "Select duration [1-7, default=2]: " dur_choice
    dur_choice="${dur_choice:-2}"

    case "$dur_choice" in
        1) POD_TTL="1h" ;;
        2) POD_TTL="2h" ;;
        3) POD_TTL="4h" ;;
        4) POD_TTL="8h" ;;
        5) POD_TTL="24h" ;;
        6)
            read -rp "Enter duration (e.g., 30m, 6h, 2d): " custom_dur
            POD_TTL="${custom_dur:-2h}"
            ;;
        7) POD_TTL="" ;;
        *) POD_TTL="2h" ;;
    esac

    if [[ -n "$POD_TTL" ]]; then
        ok "Pod will auto-terminate after ${POD_TTL}"
    else
        warn "No auto-terminate set. Remember to run './lium-inference.sh --stop' when done!"
    fi
}

# ── Launch pod ───────────────────────────────────────────────────────────────
launch_pod() {
    POD_NAME="oa-${SELECTED_MODEL//[^a-z0-9]/-}"
    # Trim to reasonable length
    POD_NAME="${POD_NAME:0:40}"

    # Use selected executor (from pick_executor) or fall back to cheapest
    local use_executor="${SELECTED_EXECUTOR:-${CHEAPEST_EXECUTOR[$GPU_TYPE]:-}}"
    local executor_info=""
    if [[ -n "$use_executor" ]]; then
        executor_info="${EXECUTOR_INFO[$use_executor]:-}"
        info "Launching pod '${POD_NAME}' on ${GPU_TYPE} x${GPU_COUNT} (executor: ${use_executor})..."
    else
        info "Launching pod '${POD_NAME}' on ${GPU_TYPE} x${GPU_COUNT}..."
    fi

    # Build TTL flag if set
    local ttl_flag=""
    if [[ -n "${POD_TTL:-}" ]]; then
        ttl_flag="--ttl ${POD_TTL}"
    fi

    local launch_output
    local launch_rc
    # Don't specify port - let Lium assign any available port
    # Use -y flag to skip confirmation prompt (no 'yes' pipe needed)
    # Run in background with output to temp file - lium up attaches to console otherwise
    # NOTE: --jupyter bypasses broken default templates on some executors
    local launch_tmpfile
    launch_tmpfile=$(mktemp)
    
    if [[ -n "$use_executor" ]]; then
        # Use specific executor ID
        lium up "${use_executor}" --name "${POD_NAME}" --ports 1 --jupyter -y ${ttl_flag} > "$launch_tmpfile" 2>&1 &
    elif [[ "$GPU_COUNT" -gt 1 ]]; then
        lium up --gpu "${GPU_TYPE}" -c "${GPU_COUNT}" --name "${POD_NAME}" --ports 1 --jupyter -y ${ttl_flag} > "$launch_tmpfile" 2>&1 &
    else
        lium up --gpu "${GPU_TYPE}" --name "${POD_NAME}" --ports 1 --jupyter -y ${ttl_flag} > "$launch_tmpfile" 2>&1 &
    fi
    local launch_pid=$!
    
    info "Pod launch initiated (PID: ${launch_pid}). Waiting for pod to appear..."
    
    # Wait for pod to appear in lium ps (up to 60s)
    local wait_count=0
    local max_launch_wait=60
    while [[ $wait_count -lt $max_launch_wait ]]; do
        sleep 2
        wait_count=$((wait_count + 2))
        
        # Check if pod exists
        if lium ps 2>/dev/null | grep -q "${POD_NAME}"; then
            ok "Pod '${POD_NAME}' appeared in lium ps!"
            break
        fi
        
        # Check if launch process exited with error
        if ! kill -0 $launch_pid 2>/dev/null; then
            launch_output=$(cat "$launch_tmpfile" 2>/dev/null || echo "No output captured")
            if echo "$launch_output" | grep -qi "No executors available\|error\|failed"; then
                err "Failed to launch pod!"
                err "  ${launch_output}"
                rm -f "$launch_tmpfile"
                return 1
            fi
        fi
        
        printf "  [%ds / %ds] Waiting for pod creation...\n" "$wait_count" "$max_launch_wait"
    done
    
    # Kill the background lium up process if still running (it attaches to console)
    if kill -0 $launch_pid 2>/dev/null; then
        info "Detaching from pod console (killing attach process)..."
        kill $launch_pid 2>/dev/null || true
        sleep 1
    fi
    
    rm -f "$launch_tmpfile"

    # Verify pod actually exists and discover the assigned port
    sleep 2
    local pod_info
    pod_info=$(lium ps 2>/dev/null | grep "${POD_NAME}" || true)
    if [[ -z "$pod_info" ]]; then
        err "Pod '${POD_NAME}' was not created. Lium may have no available executors."
        err "  ${launch_output}"
        return 1
    fi

    # Extract the assigned port from lium ps output
    # Lium format: "40031:40031, 40032:40032, ..." (external:internal mapping)
    # We need to pick one of the exposed ports (not SSH port 22)
    # 
    # Strategy: Find all port mappings, exclude SSH (22:xxxxx), pick first exposed port
    POD_PORT=$(echo "$pod_info" | grep -oP '\b\d{5}:\d{5}\b' | head -1 | cut -d: -f1 || true)
    if [[ -z "$POD_PORT" ]]; then
        # Fallback: try to find any 5-digit port (Lium uses 40xxx range)
        POD_PORT=$(echo "$pod_info" | grep -oP '\b4\d{4}\b' | head -1 || true)
    fi
    if [[ -z "$POD_PORT" ]]; then
        # Last resort: default to 40031 (first exposed port typically)
        warn "Could not auto-detect port from lium ps. Using default 40031."
        POD_PORT=40031
    fi

    ok "Pod '${POD_NAME}' created! Assigned port: ${POD_PORT}"
}

# ── Wait for pod to be ready ────────────────────────────────────────────────
wait_for_pod() {
    info "Waiting for pod to reach 'running' state..."
    local max_wait=120
    local elapsed=0
    while [[ "$elapsed" -lt "$max_wait" ]]; do
        local pod_line
        pod_line=$(lium ps 2>/dev/null | grep "${POD_NAME}" || true)
        # Check if pod appears in output (it exists) and look for running status
        # lium ps format varies, so check the whole line for status indicators
        if [[ -n "$pod_line" ]]; then
            # Pod exists - check for running status in any column
            if echo "$pod_line" | grep -qi "running"; then
                ok "Pod is running!"
                return 0
            fi
            # Also check for 'active' or 'ready' as alternative status words
            if echo "$pod_line" | grep -qi "active\|ready"; then
                ok "Pod is active!"
                return 0
            fi
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        printf "  [%ds / %ds] Waiting for pod...\n" "$elapsed" "$max_wait"
    done
    # Even if we timeout, check if pod exists - it might be usable anyway
    local pod_check
    pod_check=$(lium ps 2>/dev/null | grep "${POD_NAME}" || true)
    if [[ -n "$pod_check" ]]; then
        warn "Pod exists but status unclear. Proceeding anyway..."
        return 0
    fi
    err "Pod '${POD_NAME}' not found after ${max_wait}s!"
    return 1
}

# ── Deploy inference server ─────────────────────────────────────────────────
deploy_inference() {
    info "Deploying vLLM inference server for ${HF_REPO}..."

    # Generate a one-off API key for this pod session
    API_KEY=$(openssl rand -hex 16 2>/dev/null || head -c 32 /dev/urandom | xxd -p | head -c 32)
    info "Generated API key: ${API_KEY}"

    # Build the vLLM launch command
    # Use --break-system-packages for Ubuntu 24.04+ (PEP 668)
    local VLLM_CMD="pip install --break-system-packages vllm && "

    # For multi-GPU, use tensor parallel
    if [[ "$GPU_COUNT" -gt 1 ]]; then
        VLLM_CMD+="vllm serve ${HF_REPO} --tensor-parallel-size ${GPU_COUNT} --port ${POD_PORT} --max-model-len ${CTX_LEN} --trust-remote-code --api-key ${API_KEY}"
    else
        VLLM_CMD+="vllm serve ${HF_REPO} --port ${POD_PORT} --max-model-len ${CTX_LEN} --trust-remote-code --api-key ${API_KEY}"
    fi

    # Add quantization flag if needed
    if [[ "$QUANT" == "4bit" ]]; then
        VLLM_CMD+=" --quantization awq"
    fi

    info "Running inference server setup on pod..."
    info "  Command: ${VLLM_CMD}"

    # Create a comprehensive provisioning script and send it to the pod
    info "Creating provisioning script on pod..."
    
    # Write the provisioning script to a local temp file first
    local PROVISION_TMPFILE
    PROVISION_TMPFILE=$(mktemp /tmp/provision-vllm.XXXXXX.sh)
    
    # Use unquoted heredoc to allow variable expansion
    cat << REMOTE_SCRIPT > "$PROVISION_TMPFILE"
#!/bin/bash
# Auto-generated vLLM provisioning script
# Runs in background, writes status to /root/provision-status.json

set -e

LOG="/root/vllm.log"
STATUS_FILE="/root/provision-status.json"
PORT="${POD_PORT}"
MODEL="${HF_REPO}"
GPU_COUNT="${GPU_COUNT}"
API_KEY="${API_KEY}"
VLLM_CMD="${VLLM_CMD}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] \$*" >> "\$LOG"; }
update_status() {
    echo '{"status":"'\$1'","message":"'\$2'","timestamp":"'$(date -Iseconds)'","port":"'\$PORT'","model":"'\$MODEL'"}' > "\$STATUS_FILE"
}

log "=== vLLM Provisioning Started ==="
log "Model: \$MODEL"
log "Port: \$PORT"
log "GPU Count: \$GPU_COUNT"
update_status "starting" "Initializing provisioning"

# GPU check
log "=== GPU Check ==="
nvidia-smi >> "\$LOG" 2>&1 || { log "WARNING: nvidia-smi failed"; }
update_status "checking" "GPU verification"

# Install vLLM
log "=== Installing vLLM ==="
update_status "installing" "Installing vLLM package"
if ! pip install --break-system-packages vllm >> "\$LOG" 2>&1; then
    log "ERROR: pip install vllm failed!"
    update_status "failed" "pip install vllm failed - check /root/vllm.log"
    # Try to capture and show the error
    tail -50 "\$LOG" >> "\$LOG"
    exit 1
fi
log "vLLM installed successfully"
update_status "installed" "vLLM installed, starting server"

# Start vLLM server
log "=== Starting vLLM Server ==="
update_status "loading" "Loading model (this takes several minutes)"

log "Executing: \$VLLM_CMD"
eval "\$VLLM_CMD" >> "\$LOG" 2>&1 &
VLLM_PID=\$!

log "vLLM started with PID: \$VLLM_PID"

# Wait for vLLM to be ready (poll health endpoint)
log "Waiting for vLLM to become healthy..."
MAX_WAIT=600
ELAPSED=0
while [ \$ELAPSED -lt \$MAX_WAIT ]; do
    # vLLM returns HTTP 200 when healthy (body may be empty or vary)
    if curl -s -o /dev/null -w '%{http_code}' "http://localhost:\$PORT/health" 2>/dev/null | grep -q "200"; then
        log "=== vLLM is HEALTHY ==="
        update_status "ready" "Endpoint is live and ready"
        echo "PROVISIONING_COMPLETE" >> "\$LOG"
        exit 0
    fi
    # Also check /v1/models as fallback
    if curl -s -o /dev/null -w '%{http_code}' "http://localhost:\$PORT/v1/models" 2>/dev/null | grep -q "200"; then
        log "=== vLLM is HEALTHY (via /v1/models) ==="
        update_status "ready" "Endpoint is live and ready"
        echo "PROVISIONING_COMPLETE" >> "\$LOG"
        exit 0
    fi
    sleep 5
    ELAPSED=\$((ELAPSED + 5))
    update_status "loading" "Loading model (\${ELAPSED}s elapsed)"
done

log "ERROR: vLLM did not become healthy within \${MAX_WAIT}s"
update_status "failed" "Timeout waiting for vLLM"
exit 1
REMOTE_SCRIPT

    # Make the local script executable
    chmod +x "$PROVISION_TMPFILE"
    
    # Send the script to the pod using lium exec --script
    info "Uploading provisioning script to pod..."
    lium exec "${POD_NAME}" --script "$PROVISION_TMPFILE" 2>&1 || {
        err "Failed to upload provisioning script to pod!"
        rm -f "$PROVISION_TMPFILE"
        return 1
    }
    
    # Clean up local temp file
    rm -f "$PROVISION_TMPFILE"
    
    # Run the provisioning script in background with nohup
    info "Starting vLLM provisioning in background..."
    lium exec "${POD_NAME}" "nohup /root/provision-vllm.sh >> /root/vllm.log 2>&1 &"

    # Give it a moment to start
    sleep 2

    # Show initial status
    info "Provisioning started. Checking initial status..."
    lium exec "${POD_NAME}" "cat /root/provision-status.json 2>/dev/null || echo '{\"status\":\"starting\"}'" 2>/dev/null || true

    # Show initial log output
    info "Initial log output:"
    lium exec "${POD_NAME}" "head -20 /root/vllm.log" 2>/dev/null || echo "  (log not yet available)"

    ok "Inference server provisioning in background on pod."
    info "Model download + loading may take 5-15 minutes depending on model size."
    info ""
    info "Check status with:"
    info "  lium exec ${POD_NAME} 'cat /root/provision-status.json'"
    info "  lium exec ${POD_NAME} 'tail -f /root/vllm.log'"
}

# ── Wait for endpoint ───────────────────────────────────────────────────────
wait_for_endpoint() {
    info "Waiting for inference endpoint to come online..."

    # Get pod IP from lium ps
    local POD_IP=""
    local max_wait=600  # 10 minutes for large models
    local elapsed=0

    # First, get the pod's connection info
    while [[ "$elapsed" -lt 30 ]]; do
        POD_IP=$(lium ps 2>/dev/null | grep "${POD_NAME}" | grep -oP '\d+\.\d+\.\d+\.\d+' | head -1 || true)
        if [[ -n "$POD_IP" ]]; then
            break
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done

    if [[ -z "$POD_IP" ]]; then
        # Try SSH-based check instead
        warn "Could not auto-detect pod IP. Checking via SSH..."
        POD_IP=$(lium ssh "${POD_NAME}" "hostname -I" 2>/dev/null | awk '{print $1}' || true)
    fi

    if [[ -z "$POD_IP" ]]; then
        # Try lium ps --json for structured output
        warn "Trying JSON output from lium ps..."
        POD_IP=$(lium ps --json 2>/dev/null | jq -r ".[] | select(.name == \"${POD_NAME}\") | .ip // .host_ip // empty" | head -1 || true)
    fi

    if [[ -z "$POD_IP" ]]; then
        # Try extracting from lium port-forward output
        warn "Trying port-forward to detect IP..."
        local pf_info
        pf_info=$(lium port-forward "${POD_NAME}" --dry-run 2>&1 || true)
        POD_IP=$(echo "$pf_info" | grep -oP '\d+\.\d+\.\d+\.\d+' | head -1 || true)
    fi

    ENDPOINT_PORT=${POD_PORT}

    if [[ -n "$POD_IP" ]]; then
        ENDPOINT_URL="http://${POD_IP}:${ENDPOINT_PORT}"
    else
        # Final fallback: use localhost with port-forward
        warn "Could not determine pod IP. Setting up port-forward on localhost..."
        ENDPOINT_URL="http://localhost:${ENDPOINT_PORT}"
        # Start port-forward in background
        lium port-forward "${POD_NAME}" "${ENDPOINT_PORT}:${ENDPOINT_PORT}" &
        sleep 2
        ok "Using localhost port-forward. Endpoint: ${ENDPOINT_URL}"
    fi

    # Print endpoint info immediately so user can use it even if health check hangs
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ENDPOINT READY TO USE (vLLM is loading model)${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "  ${BOLD}Endpoint URL:${NC}    ${ENDPOINT_URL}"
    echo -e "  ${BOLD}API Key:${NC}        ${API_KEY}"
    echo -e "  ${BOLD}Model:${NC}          ${HF_REPO}"
    echo -e "  ${BOLD}Pod Name:${NC}       ${POD_NAME}"
    echo ""
    echo -e "  ${CYAN}OpenAI-compatible API:${NC}"
    echo -e "    curl ${ENDPOINT_URL}/v1/chat/completions \\"
    echo -e "      -H 'Authorization: Bearer ${API_KEY}' \\"
    echo -e "      -H 'Content-Type: application/json' \\"
    echo -e "      -d '{\"model\": \"${HF_REPO}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Wait for vLLM to be ready
    info "Waiting for vLLM to finish loading model (this can take a while for large models)..."
    elapsed=0
    while [[ "$elapsed" -lt "$max_wait" ]]; do
        # Check if vLLM is responding - try multiple endpoints and response formats
        local health
        health=$(lium exec "${POD_NAME}" "curl -s -o /dev/null -w '%{http_code}' http://localhost:${POD_PORT}/health" 2>/dev/null || true)
        # vLLM returns 200 when healthy (even if body is empty or different)
        if [[ "$health" == "200" ]]; then
            ok "Inference endpoint is LIVE!"
            return 0
        fi
        # Also try the models endpoint as a fallback
        local models_check
        models_check=$(lium exec "${POD_NAME}" "curl -s -o /dev/null -w '%{http_code}' http://localhost:${POD_PORT}/v1/models" 2>/dev/null || true)
        if [[ "$models_check" == "200" ]]; then
            ok "Inference endpoint is LIVE! (verified via /v1/models)"
            return 0
        fi

        # Show progress from log (use newline for visibility)
        local last_log
        last_log=$(lium exec "${POD_NAME}" "tail -1 /root/vllm.log" 2>/dev/null || true)
        if [[ -n "$last_log" ]]; then
            printf "  [%3dm] %s\n" "$((elapsed/60))" "${last_log:0:100}"
        else
            printf "  [%3dm] Waiting for vLLM to start...\n" "$((elapsed/60))"
        fi

        sleep 10
        elapsed=$((elapsed + 10))
    done

    err "Endpoint not ready after ${max_wait}s. Check manually:"
    err "  lium ssh ${POD_NAME}"
    err "  tail -f /root/vllm.log"
    return 1
}

# ── Verify endpoint with actual API call ─────────────────────────────────────
verify_endpoint() {
    info "Verifying endpoint with test API call..."
    
    local test_response
    test_response=$(lium exec "${POD_NAME}" "curl -s -X POST http://localhost:${POD_PORT}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{\"model\": \"${HF_REPO}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say OK\"}], \"max_tokens\": 5}'" 2>/dev/null || true)
    
    if echo "$test_response" | grep -q '"choices"'; then
        ok "✓ Endpoint verified! API is responding correctly."
        return 0
    else
        warn "Endpoint health check passed but API test failed."
        warn "Response: ${test_response:0:200}"
        return 1
    fi
}

# ── Print endpoint info ─────────────────────────────────────────────────────
print_endpoint_info() {
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║                    🚀 Inference Endpoint Ready!                             ║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════════════════════════════════════╣${NC}"
    printf "║ %-14s %-61s ║\n" "Model:" "${SELECTED_MODEL}"
    printf "║ %-14s %-61s ║\n" "HF Repo:" "${HF_REPO}"
    printf "║ %-14s %-61s ║\n" "Pod Name:" "${POD_NAME}"
    printf "║ %-14s %-61s ║\n" "GPU:" "${GPU_TYPE} x${GPU_COUNT}"
    printf "║ %-14s %-61s ║\n" "Endpoint:" "${ENDPOINT_URL}"
    printf "║ %-14s %-61s ║\n" "Port:" "${ENDPOINT_PORT}"
    if [[ -n "${POD_TTL:-}" ]]; then
        printf "║ %-14s %-61s ║\n" "Auto-stop:" "${POD_TTL} (then billing stops)"
    else
        printf "║ %-14s %-61s ║\n" "Auto-stop:" "⚠ Manual stop required!"
    fi
    echo -e "${BOLD}╠══════════════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║  OpenAI-Compatible API:                                                     ║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════════════════════════════════════╣${NC}"
    printf "║  POST %-69s ║\n" "${ENDPOINT_URL}/v1/chat/completions"
    printf "║  POST %-69s ║\n" "${ENDPOINT_URL}/v1/completions"
    echo -e "${BOLD}╠══════════════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║  Open Agent Configuration:                                                  ║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "║  Set in your OA config or environment:                                      ║"
    echo -e "║  ${CYAN}export OA_ENDPOINT=${ENDPOINT_URL}/v1${NC}"
    echo -e "║  ${CYAN}export OA_MODEL=${SELECTED_MODEL}${NC}"
    echo -e "║  Or in .oa/config.json:                                                     ║"
    echo -e "║  ${CYAN}{\"endpoint\": \"${ENDPOINT_URL}/v1\", \"model\": \"${SELECTED_MODEL}\"}${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║  Quick Test:                                                                ║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "║  ${CYAN}curl ${ENDPOINT_URL}/v1/chat/completions \\${NC}"
    echo -e "║  ${CYAN}  -H 'Content-Type: application/json' \\${NC}"
    echo -e "║  ${CYAN}  -d '{\"model\": \"${HF_REPO}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║  Management:                                                                ║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "║  ${CYAN}./lium-inference.sh --status${NC}    Check pod & endpoint status"
    echo -e "║  ${CYAN}./lium-inference.sh --stop${NC}      Tear down pod & stop billing"
    echo -e "║  ${CYAN}lium ssh ${POD_NAME}${NC}            SSH into the pod"
    echo -e "║  ${CYAN}lium exec ${POD_NAME} 'tail -f /root/vllm.log'${NC}   Watch logs"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"

    # Save state
    save_state

    # Write endpoint file for fire-and-forget consumers (always write it)
    write_endpoint_file
}

# ── Configure OA endpoint ───────────────────────────────────────────────────
configure_oa() {
    local oa_config_dir="${HOME}/.open-agents"
    local oa_config="${oa_config_dir}/config.json"

    mkdir -p "${oa_config_dir}"

    # Merge endpoint into existing config or create new
    if [[ -f "${oa_config}" ]]; then
        info "Updating existing OA config at ${oa_config}..."
        local tmp
        tmp=$(mktemp)
        jq --arg url "${ENDPOINT_URL}/v1" --arg model "${HF_REPO}" \
            '.endpoint = $url | .model = $model' "${oa_config}" > "$tmp" && mv "$tmp" "${oa_config}"
    else
        info "Creating OA config at ${oa_config}..."
        cat > "${oa_config}" <<EOF
{
  "endpoint": "${ENDPOINT_URL}/v1",
  "model": "${HF_REPO}"
}
EOF
    fi

    ok "OA config updated with endpoint: ${ENDPOINT_URL}/v1"
    ok "Model set to: ${HF_REPO}"

    # Also set environment variables for current session
    export OA_ENDPOINT="${ENDPOINT_URL}/v1"
    export OA_MODEL="${HF_REPO}"
    export OA_API_KEY="${API_KEY}"

    # Persist to shell profile
    local profile="${HOME}/.bashrc"
    if ! grep -q "OA_ENDPOINT" "${profile}" 2>/dev/null; then
        echo "" >> "${profile}"
        echo "# Open Agent inference endpoint (set by lium-inference.sh)" >> "${profile}"
        echo "export OA_ENDPOINT=\"${ENDPOINT_URL}/v1\"" >> "${profile}"
        echo "export OA_MODEL=\"${HF_REPO}\"" >> "${profile}"
        echo "export OA_API_KEY=\"${API_KEY}\"" >> "${profile}"
        info "Added OA_ENDPOINT, OA_MODEL, and OA_API_KEY to ${profile}"
    fi
}

# ── Status check ─────────────────────────────────────────────────────────────
show_status() {
    load_state

    if [[ -z "${POD_NAME:-}" ]]; then
        info "No active inference pod found."
        return
    fi

    echo -e "${BOLD}Current Inference Pod:${NC}"
    echo "  Model:    ${SELECTED_MODEL}"
    echo "  Pod:      ${POD_NAME}"
    echo "  GPU:      ${GPU_TYPE} x${GPU_COUNT}"
    echo "  Endpoint: ${ENDPOINT_URL}"
    echo ""

    info "Pod status:"
    lium ps 2>/dev/null | grep -E "NAME|${POD_NAME}" || warn "Pod not found in 'lium ps'"

    echo ""
    info "Endpoint health check..."
    local health
    health=$(lium exec "${POD_NAME}" "curl -s http://localhost:${POD_PORT}/health" 2>/dev/null || true)
    if [[ "$health" == *"OK"* || "$health" == *"ready"* ]]; then
        ok "Endpoint is healthy!"
    else
        warn "Endpoint not responding. Model may still be loading."
        info "Check logs: lium exec ${POD_NAME} 'tail -20 /root/vllm.log'"
    fi
}

# ── Stop pod ─────────────────────────────────────────────────────────────────
stop_pod() {
    load_state

    if [[ -z "${POD_NAME:-}" ]]; then
        info "No active inference pod to stop."
        return
    fi

    warn "Stopping pod '${POD_NAME}'..."
    lium rm "${POD_NAME}"

    # Clean up state
    rm -f "${STATE_FILE}"

    # Remove env vars from bashrc
    sed -i '/# Open Agent inference endpoint/d' "${HOME}/.bashrc" 2>/dev/null || true
    sed -i '/export OA_ENDPOINT=/d' "${HOME}/.bashrc" 2>/dev/null || true
    sed -i '/export OA_MODEL=/d' "${HOME}/.bashrc" 2>/dev/null || true
    sed -i '/export OA_API_KEY=/d' "${HOME}/.bashrc" 2>/dev/null || true

    # Remove endpoint file
    rm -f "${ENDPOINT_FILE}"

    ok "Pod removed and state cleaned up."
}

# ── Main ─────────────────────────────────────────────────────────────────────
main() {
    echo -e "${BOLD}"
    echo "  ╔═══════════════════════════════════════════════════════════╗"
    echo "  ║          Lium Inference — GPU Pod Launcher              ║"
    echo "  ║     Spin up open-source LLMs on rented GPUs             ║"
    echo "  ╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Parse arguments first (help/list don't need prereqs)
    case "${1:-}" in
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --list, -l              List available models"
            echo "  --model NAME, -m        Select model by name"
            echo "  --fire-and-forget, -f   Non-interactive: launch, write .lium-endpoint.json, exit"
            echo "  --ttl DURATION, -t      Pod TTL (e.g. 1h, 2h, 4h, 8h, 24h). Default: 2h"
            echo "  --status, -s            Show current pod status"
            echo "  --stop                  Stop and remove the pod"
            echo "  --test                  Test mode: restrict to single 3090/4090 (cheaper testing)"
            echo "  --help, -h              Show this help"
            echo ""
            echo "Fire-and-forget mode:"
            echo "  ./lium-inference.sh -f -m llama3.1-8b"
            echo "  ./lium-inference.sh -f -m qwen3-32b -t 4h"
            echo "  Writes .lium-endpoint.json with endpoint URL, API key, and model info."
            echo "  OpenClaw or any client can read this file to connect."
            echo ""
            echo "Run without arguments for interactive model selection."
            exit 0
            ;;
        --list|-l)
            list_models
            exit 0
            ;;
    esac

    # All other commands need full prereqs
    check_prereqs
    show_available_gpus

    # Parse all arguments
    local CUSTOM_TTL=""
    SELECTED_MODEL=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model|-m)
                SELECTED_MODEL="${2:-}"
                if [[ -z "$SELECTED_MODEL" ]]; then
                    err "--model requires a model name. Use --list to see options."
                    exit 1
                fi
                shift 2
                ;;
            --fire-and-forget|-f)
                FIRE_AND_FORGET=true
                shift
                ;;
            --test)
                TEST_MODE=true
                shift
                ;;
            --ttl|-t)
                CUSTOM_TTL="${2:-}"
                if [[ -z "$CUSTOM_TTL" ]]; then
                    err "--ttl requires a duration (e.g. 1h, 2h, 4h)"
                    exit 1
                fi
                shift 2
                ;;
            --status|-s)
                show_status
                exit 0
                ;;
            --stop)
                stop_pod
                exit 0
                ;;
            "")
                shift
                ;;
            *)
                # Treat as model name shorthand (only if no --model given)
                if [[ -z "$SELECTED_MODEL" ]]; then
                    SELECTED_MODEL="$1"
                else
                    err "Unknown option: $1"
                    exit 1
                fi
                shift
                ;;
        esac
    done

    # If no model selected, go interactive
    if [[ -z "$SELECTED_MODEL" ]]; then
        list_models
        pick_model
    fi

    # Parse the selected model's specs
    parse_model

    # Refresh GPU availability — data from earlier parse_lium_ls() may be stale
    # after the user spent time in the interactive picker
    info "Refreshing GPU availability data..."
    parse_lium_ls 2>/dev/null || true

    # Apply test mode restrictions
    if [[ "${TEST_MODE:-}" == true ]]; then
        info "Test mode: restricting to single RTX3090 or RTX4090 GPUs..."
        # Override GPU selection to consumer GPUs only
        if [[ "$GPU_TYPE" != "RTX3090" && "$GPU_TYPE" != "RTX4090" ]]; then
            info "  Original GPU type ${GPU_TYPE} overridden for test mode"
            GPU_TYPE="RTX4090"
        fi
        GPU_COUNT=1
        info "  Using ${GPU_TYPE} x${GPU_COUNT} for testing"
    fi

    # Find available GPUs
    find_gpu

    # Let user pick which executor (pod) to use
    if [[ "$FIRE_AND_FORGET" != true ]]; then
        pick_executor
    fi

    # Duration / TTL
    if [[ -n "$CUSTOM_TTL" ]]; then
        POD_TTL="$CUSTOM_TTL"
        ok "Pod TTL set to ${POD_TTL}"
    elif [[ "$FIRE_AND_FORGET" == true ]]; then
        POD_TTL="2h"
        info "Fire-and-forget mode: defaulting to 2h TTL"
    else
        ask_duration
    fi

    # Confirm before launching (skip in fire-and-forget)
    if [[ "$FIRE_AND_FORGET" != true ]]; then
        echo ""
        read -rp "$(echo -e ${BOLD}Launch pod with ${GPU_TYPE} x${GPU_COUNT} for ${SELECTED_MODEL}? [Y/n]: ${NC})" confirm
        confirm="${confirm:-Y}"
        if [[ ! "$confirm" =~ ^[Yy] ]]; then
            info "Cancelled."
            exit 0
        fi
    else
        info "Fire-and-forget: launching ${GPU_TYPE} x${GPU_COUNT} for ${SELECTED_MODEL} (TTL: ${POD_TTL:-none})"
    fi

    # Launch!
    if ! launch_pod; then
        err "Pod launch failed. Cannot proceed."
        exit 1
    fi
    if ! wait_for_pod; then
        err "Pod did not become ready. Check manually with: lium ps"
        exit 1
    fi
    deploy_inference

    # Write pending endpoint file immediately so user has something to poll
    ENDPOINT_URL="pending"  # Will be updated when ready
    print_endpoint_info_pending
    
    if [[ "$FIRE_AND_FORGET" == true ]]; then
        # Fork to background: wait for endpoint, verify, update file, then exit parent
        info "Pod launched. Forking to background to wait for endpoint..."
        info "Endpoint file will be written to: ${ENDPOINT_FILE}"
        (
            if ! wait_for_endpoint; then
                err "Endpoint failed to come online. Check logs: lium ssh ${POD_NAME} 'tail -f /root/vllm.log'"
                exit 1
            fi
            # Verify with actual API call before declaring ready
            if ! verify_endpoint; then
                err "Endpoint verification failed. Check logs: lium ssh ${POD_NAME} 'tail -f /root/vllm.log'"
                exit 1
            fi
            print_endpoint_info
            configure_oa
            echo ""
            ok "Fire-and-forget complete! Endpoint file: ${ENDPOINT_FILE}"
            ok "Point your agent at: ${ENDPOINT_URL}/v1"
        ) &
        info "Background PID: $!"
        info "You can monitor progress: tail -f /proc/$!/fd/1"
    else
        # Interactive mode: return immediately, let user poll for endpoint
        info ""
        info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        info "  Pod is provisioning in the background."
        info "  The endpoint file will be updated when ready."
        info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        info ""
        ok "Provisioning started! Check status with:"
        info "  lium exec ${POD_NAME} 'cat /root/provision-status.json'"
        info "  lium exec ${POD_NAME} 'tail -f /root/vllm.log'"
        info "  cat ${ENDPOINT_FILE}"
        info ""
        info "When status shows 'verified', the endpoint URL will be in ${ENDPOINT_FILE}"
        
        # Fork background process to update endpoint file when ready
        (
            if wait_for_endpoint; then
                # Verify with actual API call before declaring ready
                if verify_endpoint; then
                    print_endpoint_info
                    configure_oa
                    ok "Endpoint ready! Updated: ${ENDPOINT_FILE}"
                fi
            fi
        ) &
        info "Background watcher PID: $!"
    fi
}

main "$@"
