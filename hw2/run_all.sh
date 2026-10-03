#!/usr/bin/env bash
set -euo pipefail
MODELS=("${@:-Qwen/Qwen3-8B}")

python test_seedlm.py
python design_space.py --device cuda:0 --n 4096

for MODEL in "${MODELS[@]}"; do
  NAME=$(basename "$MODEL" | tr 'A-Z' 'a-z')
  EV="python evaluate.py --model $MODEL --zeroshot"
  $EV --method fp16
  for B in 4 3; do
    [ -d runs/$NAME-w$B/layers ] && [ -f runs/$NAME-w$B/compress_log.json ] || \
      python compress.py --model "$MODEL" --bits $B --out runs/$NAME-w$B --compile
    $EV --method seedlm --compressed runs/$NAME-w$B
    $EV --method rtn --bits $B --group 0     
    $EV --method awq --bits $B --group 0       
    $EV --method rtn --bits $B --group 128  
  done
done
python collect_results.py
