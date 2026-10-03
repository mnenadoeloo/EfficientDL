# HW2 — воспроизведение SeedLM (arXiv:2410.10714)

## Окружение

```bash
pip install torch transformers datasets accelerate lm_eval pandas
```

## Запуск

```bash
python test_seedlm.py                  

python design_space.py --device cuda:0 --n 4096

# сжатие 
python compress.py --model Qwen/Qwen3-8B --bits 4 --out runs/qwen3-8b-w4 --kernel triton

# оценка
python evaluate.py --model Qwen/Qwen3-8B --method fp16 --zeroshot
python evaluate.py --model Qwen/Qwen3-8B --method seedlm --compressed runs/qwen3-8b-w4 --zeroshot
python evaluate.py --model Qwen/Qwen3-8B --method rtn --bits 4 --group 0 --zeroshot

# AWQ-baseline
python test_awq.py                                 
python evaluate.py --model Qwen/Qwen3-8B --method awq --bits 4 --group 0 --zeroshot
python evaluate.py --model Qwen/Qwen3-8B --method awq --bits 3 --group 0 --zeroshot

python collect_results.py # results/results.md
```

Всё сразу для нескольких моделей: `./run_all.sh Qwen/Qwen3-8B Qwen/Qwen3-14B`.