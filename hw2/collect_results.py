"""Collect results/*.json into results/results.md."""
import glob
import json
from collections import defaultdict

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "boolq"]
ORDER = {"fp16": 0, "seedlm": 1, "rtn": 2, "awq": 3, "omniquant": 4, "quip": 5}
NAMES = {"awq": "AWQ", "omniquant": "OmniQuant", "quip": "QuIP#", "rtn": "RTN"}

by_model = defaultdict(list)
for f in sorted(glob.glob("results/*.json")):
    r = json.load(open(f))
    by_model[r["model"]].append(r)


def label(r):
    if r["method"] == "fp16":
        return "BF16"
    if r["method"] == "seedlm":
        return f"SeedLM W{round(r['bits_per_weight'])}"
    if r["method"] == "quip":
        return f"QuIP# W{r['rtn_bits']}"
    return f"{NAMES[r['method']]} W{r['rtn_bits']} " + (f"g{r['rtn_group']}" if r["rtn_group"] else "на канал")


def size_key(m): # 0.6B < 8B < 14B
    return float(m.split("-")[-1].rstrip("B"))


out = []
for model in sorted(by_model, key=size_key):
    rows = by_model[model]
    base = next((r for r in rows if r["method"] == "fp16"), None)
    rows.sort(key=lambda r: (ORDER[r["method"]], -round(r["bits_per_weight"]), r.get("rtn_group") or 0))
    out += [f"### {model.split('/')[-1]}", "",
            "| Метод | Бит/вес | WikiText-2 ppl | Δppl, % | " + " | ".join(TASKS) + " | Среднее | Retained, % |",
            "|---|---|---|---|" + "---|" * (len(TASKS) + 2)]
    for r in rows:
        zs = r.get("zero_shot")
        cells = [f"{100 * zs[t]:.2f}" for t in TASKS] + [f"{100 * r['zero_shot_mean']:.2f}"] if zs else ["–"] * (len(TASKS) + 1)
        dppl = f"{100 * (r['wikitext2_ppl'] / base['wikitext2_ppl'] - 1):+.1f}" if base and r is not base else "–"
        ret = f"{100 * r['zero_shot_mean'] / base['zero_shot_mean']:.1f}" if base and zs and base.get("zero_shot") else "–"
        out.append(f"| {label(r)} | {r['bits_per_weight']:.2f} | {r['wikitext2_ppl']:.2f} | {dppl} | "
                   + " | ".join(cells) + f" | {ret} |")
    out.append("")

open("results/results.md", "w").write("\n".join(out))
print("\n".join(out))
