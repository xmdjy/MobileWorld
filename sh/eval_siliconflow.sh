#!/usr/bin/env bash
# Run MobileWorld evaluation against a VL model on SiliconFlow (https://siliconflow.cn).
#
# Prereqs:
#   1. Fill in API_KEY in .env with your SiliconFlow key (https://cloud.siliconflow.cn/account/ak).
#   2. Docker + KVM ready (sudo uv run mw env check).
#
# Override anything via env vars, e.g.:
#   MODEL=Qwen/Qwen2.5-VL-32B-Instruct TASK=SetAlarmTask bash sh/eval_siliconflow.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# Load .env so API_KEY is exported into the eval process.
set -a
# shellcheck disable=SC1091
source .env
set +a

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
BASE_URL="${BASE_URL:-https://api.siliconflow.cn/v1}"
AGENT_TYPE="${AGENT_TYPE:-qwen3vl}"
TASK="${TASK:-ALL}"
# Strip slashes from model name for output dir (Qwen/Qwen2.5-VL-72B -> Qwen-Qwen2.5-VL-72B)
MODEL_TAG="${MODEL//\//-}"
OUTPUT="${OUTPUT:-./output/${MODEL_TAG}-baseline-hist1}"
ENV_COUNT="${ENV_COUNT:-5}"
MAX_ROUND="${MAX_ROUND:-50}"
STEP_WAIT="${STEP_WAIT:-3}"
HISTORY_N_IMAGES="${HISTORY_N_IMAGES:-1}"

# Make sure enough docker envs are healthy; spin up if not.
healthy=$(docker ps \
    --filter "name=mobile_world_env_" \
    --filter "health=healthy" \
    --format "{{.Names}}" | wc -l)
if [ "$healthy" -lt "$ENV_COUNT" ]; then
    sudo uv run mw env run --count "$ENV_COUNT" --launch-interval 20
else
    echo "Found $healthy healthy mobile_world_env containers; skip mw env run."
fi

# Quick sanity check on the API key.
if [ -z "${API_KEY:-}" ] || [[ "$API_KEY" == sk-your-siliconflow-key* ]]; then
    echo "ERROR: API_KEY in .env is still the placeholder. Fill in your real SiliconFlow key from https://cloud.siliconflow.cn/account/ak." >&2
    exit 1
fi

mkdir -p "$OUTPUT"

sudo env \
    HISTORY_N_IMAGES="$HISTORY_N_IMAGES" \
    API_KEY="$API_KEY" \
    uv run mw eval \
        --agent_type "$AGENT_TYPE" \
        --task "$TASK" \
        --model_name "$MODEL" \
        --llm_base_url "$BASE_URL" \
        --api_key "$API_KEY" \
        --output "$OUTPUT" \
        --max_round "$MAX_ROUND" \
        --step_wait_time "$STEP_WAIT"
