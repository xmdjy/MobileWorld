#!/usr/bin/env bash
# checker_MAS v1.2 experiment
#   - checker triggers only on FINISHED/ANSWER (dangerous actions)
#   - MAS decoupled from checker; fires directly on stuck_loop (no_change_count >= 3)
#   - mas_skip_recent_k = 8 (CLI / code defaults aligned)
#   - 3 candidates (warm/cold/skip_recent), same model as main agent, random + filter
#   - Same 6-task subset as v1.0 / v1.1 for comparison
#
# Override via env vars: TASK=... OUTPUT=... ENV_COUNT=...

set -euo pipefail

cd "$(dirname "$0")/.."

set -a
# shellcheck disable=SC1091
source .env
set +a

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
BASE_URL="${BASE_URL:-https://api.siliconflow.cn/v1}"
AGENT_TYPE="${AGENT_TYPE:-qwen3vl}"
OUTPUT="${OUTPUT:-./output/checker_MAS_v1.2}"
MAX_ROUND="${MAX_ROUND:-50}"
STEP_WAIT="${STEP_WAIT:-3}"
HISTORY_N_IMAGES="${HISTORY_N_IMAGES:-1}"
ENV_COUNT="${ENV_COUNT:-1}"

TASK="${TASK:-AcceptMeetingTask,AdjustFontIconMaximumTask,CheckConferenceAndSendSmsTask2,CheckInvoiceTask3,SetAlarmTask}"

healthy=$(docker ps \
    --filter "name=mobile_world_env_" \
    --filter "health=healthy" \
    --format "{{.Names}}" | wc -l)
if [ "$healthy" -lt "$ENV_COUNT" ]; then
    sudo uv run mw env run --count "$ENV_COUNT" --launch-interval 20
else
    echo "Found $healthy healthy mobile_world_env containers; skip mw env run."
fi

if [ -z "${API_KEY:-}" ] || [[ "$API_KEY" == sk-your-siliconflow-key* ]]; then
    echo "ERROR: API_KEY in .env is still the placeholder." >&2
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
        --step_wait_time "$STEP_WAIT" \
        --auto-retry 0 \
        --max-concurrency 1 \
        --checker-enabled \
        --mas-enabled
