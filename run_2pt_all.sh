#!/usr/bin/env bash
# twopoint_fic.py 를 idx 4~12 에 대해 2개씩 순차 실행.
# 이미 돌고 있는 첫 쌍(4,5)이 끝나기를 기다린 뒤 다음 쌍으로 넘어간다.
set -u
cd /scratch/home/wog3597/optim
LOG=output_ppmi_pd/_logs

done_pair() {           # 인자 idx 들이 전부 끝났나 (RESULT 또는 Traceback)
    for i in "$@"; do
        grep -qE "^RESULT|Traceback" "$LOG/2pt_idx$i.log" 2>/dev/null || return 1
    done
    return 0
}

echo "=== 2점 FIC 체인 시작 $(date '+%F %T') ==="
until done_pair 4 5; do sleep 60; done
echo "[pair 4,5] 완료 $(date '+%F %T')"
grep -h "^RESULT" $LOG/2pt_idx4.log $LOG/2pt_idx5.log

for pair in "6 7" "8 9" "10 11" "12"; do
    echo "[pair $pair] 시작 $(date '+%F %T')"
    for i in $pair; do
        nohup python3 twopoint_fic.py --idx "$i" > "$LOG/2pt_idx$i.log" 2>&1 &
    done
    wait
    echo "[pair $pair] 완료 $(date '+%F %T')"
    for i in $pair; do grep -hE "^RESULT|Traceback" "$LOG/2pt_idx$i.log" | head -3; done
done
echo "=== 전체 완료 $(date '+%F %T') ==="
