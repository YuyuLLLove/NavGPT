#!/usr/bin/env bash
set -euo pipefail

# NavGPT + OpenAI 兼容 HTTP（百炼 / OpenAI / OpenRouter 等）
#
# 推荐：在 nav_src/parser.py 里填写 OPENAI_COMPAT_API_KEY / OPENAI_COMPAT_API_BASE，
# 此处可不设环境变量。
#
# 可选覆盖（会覆盖代码里的常量）:
#   export OPENAI_API_KEY=...  或 DASHSCOPE_API_KEY
#   export OPENAI_API_BASE=https://...
#   不设 MODEL_NAME 时用 parser.py 里 --llm_model_name 的默认值（如应用 ID）。
#   MODEL_NAME=qwen-plus bash run_qwen_api.sh smoke   # 仅直连模型网关时需要
#
# preds/submit_${VAL_ENV_NAME}.json 已存在时 NavGPT 会跳过评测；
# smoke 会先删该文件；全量重跑可加 CLEAN_PREDS=1
#
# smoke 默认跑 2 条；跑 10 条示例：
#   SMOKE_ITERS=10 bash run_qwen_api.sh smoke
#   bash run_qwen_api.sh smoke10

ACTION="${1:-run}"

_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
NAV_SRC_DIR="${NAV_SRC_DIR:-$(cd "${_SCRIPT_DIR}/.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-${_REPO_ROOT}/datasets/R2R/exprs/qwen-api}"
VAL_ENV_NAME="${VAL_ENV_NAME:-R2R_val_unseen_instr}"

# 留空则不把 --llm_model_name 传给 Python，使用 parser.py 中的 default
MODEL_NAME="${MODEL_NAME:-}"
KEY="${OPENAI_API_KEY:-${DASHSCOPE_API_KEY:-${API_KEY:-}}}"
API_BASE="${API_BASE:-${OPENAI_API_BASE:-}}"

TEMPERATURE="${TEMPERATURE:-0.0}"
ITERS="${ITERS:-10}"

[[ -n "${KEY}" ]] && export OPENAI_API_KEY="${KEY}"
[[ -n "${API_BASE}" ]] && export OPENAI_API_BASE="${API_BASE}"

maybe_clean_submit_json() {
  local f="${OUTPUT_DIR}/preds/submit_${VAL_ENV_NAME}.json"
  if [[ -f "${f}" ]]; then
    echo "Removing ${f} so evaluation runs (NavGPT skips if this file exists)."
    rm -f "${f}"
  fi
}

run_navgpt() {
  if [[ "${CLEAN_PREDS:-}" == "1" ]]; then
    maybe_clean_submit_json
  fi
  cd "${NAV_SRC_DIR}"
  CMD=(
    python NavGPT.py
    --temperature "${TEMPERATURE}"
    --output_dir "${OUTPUT_DIR}"
    --val_env_name "${VAL_ENV_NAME}"
    --iters "${ITERS}"
  )
  [[ -n "${MODEL_NAME}" ]] && CMD+=(--llm_model_name "${MODEL_NAME}")
  [[ -n "${KEY}" ]] && CMD+=(--api_key "${KEY}")
  [[ -n "${API_BASE}" ]] && CMD+=(--api_base "${API_BASE}")
  "${CMD[@]}"
}

smoke_navgpt() {
  maybe_clean_submit_json
  ITERS="${SMOKE_ITERS:-2}" run_navgpt
}

case "${ACTION}" in
  run)
    run_navgpt
    ;;
  smoke)
    smoke_navgpt
    ;;
  smoke10)
    SMOKE_ITERS=10 smoke_navgpt
    ;;
  *)
    echo "Usage: bash run_qwen_api.sh [run|smoke|smoke10]"
    exit 1
    ;;
esac
