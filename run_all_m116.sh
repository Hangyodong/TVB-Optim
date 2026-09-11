#!/bin/bash
# Mouse_allen_116 34마리 simple optim 순차 실행
# 사용: bash run_all_m116.sh [START] [END] [TAG] [EXTRA_ARGS...]
#   예: bash run_all_m116.sh 0 33 _ns01 --noise-level 0.01 --chunk 4
set -u
START=${1:-1}
END=${2:-33}
TAG=${3:-}
shift $(( $# < 3 ? $# : 3 ))
LOG_DIR=output_mouse116/_fopt/logs
mkdir -p "$LOG_DIR"
for i in $(seq "$START" "$END"); do
    if [ -f "output_mouse116/_fopt/idx${i}${TAG}.json" ]; then
        echo "[skip] idx${i}${TAG} 결과 있음"
        continue
    fi
    echo "[run] idx${i}${TAG}  $(date +%H:%M:%S)"
    python3 fopt_mouse.py --idx "$i" --tag "$TAG" "$@" > "$LOG_DIR/idx${i}${TAG}.log" 2>&1
    tail -1 "$LOG_DIR/idx${i}${TAG}.log"
done
echo "ALL DONE $(date +%H:%M:%S)"
