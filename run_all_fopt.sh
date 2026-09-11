#!/bin/bash
# run_all_fopt.sh — formula_optim 병렬 배치. usage: ./run_all_fopt.sh "5 6 7" 4 [maxfev]
# 멀티프로세스 GPU 공유 필수 env (bench_par.py 와 동일)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
IDXS=${1:-"4 5 6 7 8 9 10 11 12"}
PAR=${2:-4}
MAXFEV=${3:-64}
mkdir -p output_ppmi_pd/_fopt
echo "$IDXS" | tr ' ' '\n' | xargs -P "$PAR" -I{} sh -c "
  python3 formula_optim.py --idx {} --maxfev $MAXFEV > output_ppmi_pd/_fopt/run_idx{}.log 2>&1
  grep -h RESULT output_ppmi_pd/_fopt/run_idx{}.log || echo 'idx {} FAILED'"
