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
## Дополнительные baseline'ы и замер скорости (написаны без GPU, ни разу не запускались)

```bash
python test_baselines.py # smoke-тесты на крошечных моделях: OmniQuant, QuIP#, ядра matvec

# OmniQuant (LWC, weight-only, калибровка WikiText-2, 20 эпох)
python evaluate.py --model Qwen/Qwen3-8B --method omniquant --bits 4 --group 0 --zeroshot
python evaluate.py --model Qwen/Qwen3-8B --method omniquant --bits 3 --group 0 --zeroshot --omni-epochs 20

# QuIP# (без fine-tuning)
git clone https://github.com/Cornell-RelaxML/quip-sharp third_party/quip-sharp
pip install primefac scipy # primefac нужен только для размеров без матрицы Адамара (MLP у Qwen3-14B)
python evaluate.py --model Qwen/Qwen3-8B --method quip --bits 4 --zeroshot

# GPU-аналог Таблицы 5: матрично-векторное умножение BF16 / int4 / SeedLM (декодирование seed в ядре)
python bench_matvec.py # -> results/matvec.csv
```
OmniQuant реализован как порт LWC из официального репозитория (раздел 3.2 отчёта). QuIP# использует только
поматричный квантователь из репозитория (GPL-3.0, внешняя зависимость), Гессианы собираются в `quip_baseline.py`.
