#!/bin/zsh
# Start Jarvis Gemma GGUF via llama-server (background-friendly)

set -euo pipefail

JARVIS_DIR="${JARVIS_DIR:-$HOME/Applications/Jarvis}"
START_SCRIPT="$JARVIS_DIR/scripts/start-primary-llm.sh"
MODEL_FILE="$JARVIS_DIR/models/gemma4-v2-Q4_K_M.gguf"
LOG_FILE="${JARVIS_LLM_LOG:-/tmp/juan-llm.log}"

if [[ ! -f "$MODEL_FILE" ]]; then
  echo "Model not found: $MODEL_FILE"
  exit 1
fi

if [[ ! -x "$START_SCRIPT" ]]; then
  echo "Jarvis start script not found: $START_SCRIPT"
  exit 1
fi

echo "Starting llama-server with gemma4-v2-Q4_K_M.gguf..."
echo "Log: $LOG_FILE"
nohup "$START_SCRIPT" >>"$LOG_FILE" 2>&1 &
echo $! > /tmp/juan-llm.pid