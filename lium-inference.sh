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
APIKEY="${LIUM_API_KEY:-}"

# If APIKEY is set, export it for lium CLI to use
if [[ -n "$APIKEY" ]]; then
    export LIUM_API_KEY="$APIKEY"
fi

# ── Model Catalogue ─────────────────────────────────────────────────────────
# Each entry: MODEL_ID | HuggingFace repo | Min VRAM (GB) | Recommended GPU | GPU count | Quant
# VRAM estimates assume FP16/BF16 for full-precision, GPTQ/AWQ 4-bit for quantized.
# Adjusted for KV cache overhead (~20% headroom).

declare -A MODEL_CATALOG

# --- Format: "hf_repo|min_vram_gb|gpu_type|gpu_count|quant_note|context_length" ---

# Small/Mid (27B-32B) — single GPU
MODEL_CATALOG[deepseek-r1-distill-qwen-32b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|40|A100|1|fp16|32768"
MODEL_CATALOG[qwen3-30b-a3b]="Qwen/Qwen3-30B-A3B|24|RTX4090|1|fp16|32768"
MODEL_CATALOG[glm4-32b]="THUDM/GLM-4-32B-0414|40|A100|1|fp16|32768"
MODEL_CATALOG[yi-34b]="01-ai/Yi-34B|40|A100|1|fp16|4096"
MODEL_CATALOG[command-r-35b]="CohereForAI/c4ai-command-r-v01|40|A100|1|fp16|131072"

# Qwen3.5 Small (0.8B-9B) — single GPU
MODEL_CATALOG[qwen3.5-0.8b]="Qwen/Qwen3.5-0.8B|4|RTX4090|1|fp16|32768"
MODEL_CATALOG[qwen3.5-2b]="Qwen/Qwen3.5-2B|6|RTX4090|1|fp16|32768"
MODEL_CATALOG[qwen3.5-4b]="Qwen/Qwen3.5-4B|10|RTX4090|1|fp16|32768"
MODEL_CATALOG[qwen3.5-9b]="Qwen/Qwen3.5-9B|20|RTX4090|1|fp16|32768"

# GLM Small/Mid
MODEL_CATALOG[glm4-9b]="THUDM/GLM-4-9B-0414|20|RTX4090|1|fp16|8192"

# Mid (70B-72B) — single 80GB GPU with 4-bit, or 2x80GB fp16
MODEL_CATALOG[llama3-70b]="meta-llama/Meta-Llama-3-70B|42|A100|1|4bit|8192"
MODEL_CATALOG[llama3-70b-fp16]="meta-llama/Meta-Llama-3-70B|148|A100|2|fp16|8192"
MODEL_CATALOG[qwen2.5-72b]="Qwen/Qwen2.5-72B-Instruct|42|A100|1|4bit|131072"
MODEL_CATALOG[qwen2.5-72b-fp16]="Qwen/Qwen2.5-72B-Instruct|148|A100|2|fp16|131072"
MODEL_CATALOG[deepseek-r1-distill-llama-70b]="deepseek-ai/DeepSeek-R1-Distill-Llama-70B|42|A100|1|4bit|8192"
MODEL_CATALOG[gemma3-27b]="google/gemma-3-27b-it|34|RTX4090|1|fp16|131072"
MODEL_CATALOG[mixtral-8x22b]="mistralai/Mixtral-8x22B-v0.1|92|A100|2|4bit|65536"
MODEL_CATALOG[command-r-plus-104b]="CohereForAI/c4ai-command-r-plus|60|H100|1|4bit|131072"

# Large (110B-235B) — multi-GPU
MODEL_CATALOG[qwen3-235b]="Qwen/Qwen3-235B-A22B|136|H100|2|fp16|32768"
MODEL_CATALOG[qwen3-235b-4bit]="Qwen/Qwen3-235B-A22B|72|H100|1|4bit|32768"
MODEL_CATALOG[deepseek-r1]="deepseek-ai/DeepSeek-R1|670|H100|8|fp16|131072"
MODEL_CATALOG[deepseek-r1-4bit]="deepseek-ai/DeepSeek-R1|180|H100|3|4bit|131072"
MODEL_CATALOG[deepseek-v3]="deepseek-ai/DeepSeek-V3|670|H100|8|fp16|131072"
MODEL_CATALOG[deepseek-v3-4bit]="deepseek-ai/DeepSeek-V3|180|H100|3|4bit|131072"
MODEL_CATALOG[llama4-maverick-400b]="meta-llama/Llama-4-Maverick-17B-128E|420|H100|6|fp16|131072"
MODEL_CATALOG[llama4-scout-109b]="meta-llama/Llama-4-Scout-17B-16E|120|H100|2|fp16|131072"
MODEL_CATALOG[qwen2.5-1m]="Qwen/Qwen2.5-1M|160|H100|2|4bit|1000000"
MODEL_CATALOG[glm4-1m]="THUDM/GLM-4-1M-0414|160|H100|2|4bit|1000000"

# GLM-5.x (Large MoE models)
MODEL_CATALOG[glm5-9b]="THUDM/GLM-5-9B-0414|20|RTX4090|1|fp16|8192"
MODEL_CATALOG[glm5.1-9b]="THUDM/GLM-5.1-9B-0414|20|RTX4090|1|fp16|8192"
MODEL_CATALOG[glm5.1]="THUDM/GLM-5.1-0414|800|H100|10|fp16|131072"
MODEL_CATALOG[glm5.1-4bit]="THUDM/GLM-5.1-0414|250|H100|4|4bit|131072"

# Qwen3.5 Large (27B-397B MoE)
MODEL_CATALOG[qwen3.5-27b]="Qwen/Qwen3.5-27B|60|H100|1|fp16|32768"
MODEL_CATALOG[qwen3.5-35b-a3b]="Qwen/Qwen3.5-35B-A3B|24|RTX4090|1|fp16|32768"
MODEL_CATALOG[qwen3.5-122b-a10b]="Qwen/Qwen3.5-122B-A10B|80|H100|1|fp16|32768"
MODEL_CATALOG[qwen3.5-122b-a10b-4bit]="Qwen/Qwen3.5-122B-A10B|40|H100|1|4bit|32768"
MODEL_CATALOG[qwen3.5-397b-a17b]="Qwen/Qwen3.5-397B-A17B|240|H100|3|fp16|32768"
MODEL_CATALOG[qwen3.5-397b-a17b-4bit]="Qwen/Qwen3.5-397B-A17B|80|H100|1|4bit|32768"

# ── State file ───────────────────────────────────────────────────────────────
STATE_DIR="${HOME}/.lium-inference"
STATE_FILE="${STATE_DIR}/state.json"
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
declare -A GPU_VRAM_MAP=(
    ["H100"]=80
    ["A100"]=80
    ["A6000"]=48
    ["RTX5090"]=32
    ["RTX4090"]=24
    ["RTX3090"]=24
    ["RTX6000"]=48
    ["V100"]=32
    ["T4"]=16
    ["L4"]=24
    ["A10"]=24
    ["A10G"]=24
)

parse_lium_ls() {
    AVAILABLE_GPUS=()
    local lium_output
    # No port filter - we'll discover the assigned port after pod creation
    lium_output=$(lium ls 2>/dev/null) || return 1
    
    if [[ -z "$lium_output" ]]; then
        return 1
    fi
    
    # Parse lium ls output - format varies, try multiple patterns
    # Pattern 1: "GPU_TYPE  COUNT" (space-separated)
    # Pattern 2: Table with headers
    # Pattern 3: "N x GPU_TYPE" format
    
    while IFS= read -r line; do
        # Skip header lines and empty lines
        [[ -z "$line" ]] && continue
        [[ "$line" =~ ^(GPU|Name|Type|ID|Available|---|\||\+) ]] && continue
        
        # Try to extract GPU type and count from various formats
        local gpu_type=""
        local count=0
        
        # Normalize line to uppercase for matching
        local line_upper="${line^^}"
        
        # Format: "2 x H100" or "H100 (2 available)" - case insensitive, handles spaces
        # Also match standalone numbers like "5090", "4090", "3090", "6000" without RTX prefix
        if [[ "$line_upper" =~ ([0-9]+)[[:space:]]*X[[:space:]]*(H100|A100|A6000|RTX.?5090|RTX.?4090|RTX.?3090|RTX.?6000|5090|4090|3090|6000|V100|T4|L4|A10|A10G) ]]; then
            count="${BASH_REMATCH[1]}"
            # Normalize GPU type (remove spaces, standardize naming)
            gpu_type="${BASH_REMATCH[2]}"
            gpu_type="${gpu_type// /}"  # Remove any spaces
            # Add RTX prefix if missing for consumer GPUs
            [[ "$gpu_type" =~ ^(5090|4090|3090|6000)$ ]] && gpu_type="RTX$gpu_type"
        # Format: "H100" with count in another column - case insensitive
        elif [[ "$line_upper" =~ (H100|A100|A6000|RTX.?5090|RTX.?4090|RTX.?3090|RTX.?6000|5090|4090|3090|6000|V100|T4|L4|A10|A10G) ]]; then
            gpu_type="${BASH_REMATCH[1]}"
            gpu_type="${gpu_type// /}"  # Remove any spaces
            # Add RTX prefix if missing for consumer GPUs
            [[ "$gpu_type" =~ ^(5090|4090|3090|6000)$ ]] && gpu_type="RTX$gpu_type"
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
        fi
    done <<< "$lium_output"
    
    return 0
}

# ── GPU Hierarchy (better GPUs can substitute for lesser ones) ───────────────
# Higher rank = more powerful. Used to determine if a GPU can substitute another.
declare -A GPU_RANK=(
    ["H100"]=100
    ["A100"]=90
    ["A6000"]=70
    ["RTX6000"]=70
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
        local gpu_vram="${GPU_VRAM_MAP[$avail_gpu]:-0}"
        local total_vram=$((gpu_vram * avail_count))
        
        # First check: GPU type compatibility (hierarchy-based substitution)
        if ! gpu_can_substitute "$avail_gpu" "$gpu_type"; then
            continue  # This GPU type is not suitable, try next
        fi
        
        # Check if we have enough GPUs of this type
        if [[ "$avail_count" -ge "$gpu_count" && "$total_vram" -ge "$min_vram" ]]; then
            return 0
        fi
        
        # Check if a single GPU of this type has enough VRAM
        if [[ "$gpu_vram" -ge "$min_vram" && "$avail_count" -ge 1 ]]; then
            return 0
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
                printf "  ${GREEN}✓${NC} %s x%d (%s GB each, %s GB total)\n" "$gpu" "$count" "$vram" "$((vram * count))"
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
        # We have enough of the required GPU type
        gpu_list=$(lium ls 2>/dev/null | grep -i "${GPU_TYPE}" || true)
    else
        warn "No ${GPU_TYPE} GPUs currently available (need ${GPU_COUNT}, have ${gpu_available}). Trying alternatives..."

        # Fallback GPU hierarchy: H100 > A100 80GB > A6000 > RTX4090 > A100 40GB
        local -a FALLBACKS=()
        case "$GPU_TYPE" in
            H100)  FALLBACKS=("A100" "A6000" "RTX4090") ;;
            A100)  FALLBACKS=("H100" "A6000" "RTX4090") ;;
            A6000) FALLBACKS=("A100" "H100" "RTX4090") ;;
            RTX4090) FALLBACKS=("A6000" "A100" "H100") ;;
            *)     FALLBACKS=("H100" "A100" "A6000" "RTX4090") ;;
        esac

        for fb in "${FALLBACKS[@]}"; do
            local fb_count="${AVAILABLE_GPUS[$fb]:-0}"
            info "  Trying ${fb}... (${fb_count} available)"
            if [[ "$fb_count" -ge "$GPU_COUNT" ]]; then
                # Check if fallback GPU has enough VRAM for our model
                local fb_vram
                case "$fb" in
                    H100)    fb_vram=80 ;;
                    A100)    fb_vram=80 ;;
                    A6000)   fb_vram=48 ;;
                    RTX4090) fb_vram=24 ;;
                    *)       fb_vram=0 ;;
                esac
                local total_vram=$((fb_vram * GPU_COUNT))
                if [[ "$total_vram" -ge "$MIN_VRAM" ]]; then
                    ok "Found ${fb} GPUs with sufficient VRAM (${total_vram} GB >= ${MIN_VRAM} GB)"
                    GPU_TYPE="$fb"
                    gpu_list=$(lium ls 2>/dev/null | grep -i "${fb}" || true)
                    break
                else
                    warn "  ${fb} x${GPU_COUNT} = ${total_vram} GB < ${MIN_VRAM} GB needed"
                fi
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

    info "Launching pod '${POD_NAME}' on ${GPU_TYPE} x${GPU_COUNT}..."

    # Build TTL flag if set
    local ttl_flag=""
    if [[ -n "${POD_TTL:-}" ]]; then
        ttl_flag="--ttl ${POD_TTL}"
    fi

    local launch_output
    local launch_rc
    # Don't specify port - let Lium assign any available port
    # Pipe 'yes' to auto-confirm the interactive prompt
    # NOTE: Disable pipefail because 'yes' gets SIGPIPE when lium exits
    set +o pipefail
    if [[ "$GPU_COUNT" -gt 1 ]]; then
        launch_output=$(yes | lium up --gpu "${GPU_TYPE}" -c "${GPU_COUNT}" --name "${POD_NAME}" ${ttl_flag} 2>&1) || true
    else
        launch_output=$(yes | lium up --gpu "${GPU_TYPE}" --name "${POD_NAME}" ${ttl_flag} 2>&1) || true
    fi
    launch_rc=$?
    set -o pipefail

    # Check for failure indicators in output
    if [[ "$launch_rc" -ne 0 ]] || echo "$launch_output" | grep -qi "No executors available\|error\|failed"; then
        err "Failed to launch pod!"
        err "  ${launch_output}"
        return 1
    fi

    # Verify pod actually exists and discover the assigned port
    sleep 2
    local pod_info
    pod_info=$(lium ps 2>/dev/null | grep "${POD_NAME}" || true)
    if [[ -z "$pod_info" ]]; then
        err "Pod '${POD_NAME}' was not created. Lium may have no available executors."
        err "  ${launch_output}"
        return 1
    fi

    # Extract the assigned port from lium ps output (format varies)
    # Try to find a port number in the output
    POD_PORT=$(echo "$pod_info" | grep -oP ':\K\d+' | head -1 || true)
    if [[ -z "$POD_PORT" ]]; then
        # Fallback: try to extract from ports column
        POD_PORT=$(echo "$pod_info" | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/) print $i}' | head -1 || true)
    fi
    if [[ -z "$POD_PORT" ]]; then
        # Last resort: default to 8000 and hope for the best
        warn "Could not auto-detect port from lium ps. Using default 8000."
        POD_PORT=8000
    fi

    ok "Pod '${POD_NAME}' created! Assigned port: ${POD_PORT}"
}

# ── Wait for pod to be ready ────────────────────────────────────────────────
wait_for_pod() {
    info "Waiting for pod to reach 'running' state..."
    local max_wait=120
    local elapsed=0
    while [[ "$elapsed" -lt "$max_wait" ]]; do
        local status
        status=$(lium ps 2>/dev/null | grep "${POD_NAME}" | awk '{print $3}' || true)
        if [[ "$status" == *"running"* || "$status" == *"Running"* ]]; then
            ok "Pod is running!"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        printf "  [%ds / %ds]\r" "$elapsed" "$max_wait"
    done
    warn "Pod not yet running after ${max_wait}s. Check 'lium ps' manually."
}

# ── Deploy inference server ─────────────────────────────────────────────────
deploy_inference() {
    info "Deploying vLLM inference server for ${HF_REPO}..."

    # Build the vLLM launch command
    local VLLM_CMD="pip install vllm && "

    # For multi-GPU, use tensor parallel
    if [[ "$GPU_COUNT" -gt 1 ]]; then
        VLLM_CMD+="vllm serve ${HF_REPO} --tensor-parallel-size ${GPU_COUNT} --port ${POD_PORT} --max-model-len ${CTX_LEN} --trust-remote-code"
    else
        VLLM_CMD+="vllm serve ${HF_REPO} --port ${POD_PORT} --max-model-len ${CTX_LEN} --trust-remote-code"
    fi

    # Add quantization flag if needed
    if [[ "$QUANT" == "4bit" ]]; then
        VLLM_CMD+=" --quantization awq"
    fi

    info "Running inference server setup on pod..."
    info "  Command: ${VLLM_CMD}"

    # Execute on pod via lium exec (non-blocking: nohup in background)
    lium exec "${POD_NAME}" "nohup bash -c '${VLLM_CMD}' > /root/vllm.log 2>&1 &"

    ok "Inference server starting in background on pod."
    info "Model download + loading may take 5-15 minutes depending on model size."
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

    ENDPOINT_PORT=${POD_PORT}

    if [[ -n "$POD_IP" ]]; then
        ENDPOINT_URL="http://${POD_IP}:${ENDPOINT_PORT}"
    else
        # Fallback: use lium ssh port forwarding info
        warn "Using SSH tunnel for endpoint access."
        ENDPOINT_URL="ssh-tunnel"
    fi

    # Wait for vLLM to be ready
    info "Waiting for vLLM to finish loading model (this can take a while for large models)..."
    elapsed=0
    while [[ "$elapsed" -lt "$max_wait" ]]; do
        # Check if vLLM is responding
        local health
        health=$(lium exec "${POD_NAME}" "curl -s http://localhost:${POD_PORT}/health" 2>/dev/null || true)
        if [[ "$health" == *"OK"* || "$health" == *"ready"* ]]; then
            ok "Inference endpoint is LIVE!"
            return 0
        fi

        # Show progress from log
        local last_log
        last_log=$(lium exec "${POD_NAME}" "tail -1 /root/vllm.log" 2>/dev/null || true)
        if [[ -n "$last_log" ]]; then
            printf "  [%3dm] %s\r" "$((elapsed/60))" "${last_log:0:80}"
        fi

        sleep 10
        elapsed=$((elapsed + 10))
    done

    warn "Endpoint not ready after ${max_wait}s. Check manually:"
    warn "  lium ssh ${POD_NAME}"
    warn "  tail -f /root/vllm.log"
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

    # Persist to shell profile
    local profile="${HOME}/.bashrc"
    if ! grep -q "OA_ENDPOINT" "${profile}" 2>/dev/null; then
        echo "" >> "${profile}"
        echo "# Open Agent inference endpoint (set by lium-inference.sh)" >> "${profile}"
        echo "export OA_ENDPOINT=\"${ENDPOINT_URL}/v1\"" >> "${profile}"
        echo "export OA_MODEL=\"${HF_REPO}\"" >> "${profile}"
        info "Added OA_ENDPOINT and OA_MODEL to ${profile}"
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
            echo "  --list, -l          List available models"
            echo "  --model NAME, -m    Select model by name"
            echo "  --status, -s        Show current pod status"
            echo "  --stop              Stop and remove the pod"
            echo "  --help, -h          Show this help"
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

    case "${1:-}" in
        --model|-m)
            SELECTED_MODEL="${2:-}"
            if [[ -z "$SELECTED_MODEL" ]]; then
                err "--model requires a model name. Use --list to see options."
                exit 1
            fi
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
            # Interactive mode
            list_models
            pick_model
            ;;
        *)
            # Treat first arg as model name shorthand
            SELECTED_MODEL="$1"
            ;;
    esac

    # Parse the selected model's specs
    parse_model

    # Find available GPUs
    find_gpu

    # Ask for duration (cost control)
    ask_duration

    # Confirm before launching
    echo ""
    read -rp "$(echo -e ${BOLD}Launch pod with ${GPU_TYPE} x${GPU_COUNT} for ${SELECTED_MODEL}? [Y/n]: ${NC})" confirm
    confirm="${confirm:-Y}"
    if [[ ! "$confirm" =~ ^[Yy] ]]; then
        info "Cancelled."
        exit 0
    fi

    # Launch!
    launch_pod
    wait_for_pod
    deploy_inference
    wait_for_endpoint
    print_endpoint_info
    configure_oa

    echo ""
    ok "All done! Your inference endpoint is ready for Open Agent."
}

main "$@"
