"""Score the 319-item sample answered by the proprietary models (results/claude_sample/)
and write paper/generated/tab_large.tex. The six open-weight models are re-scored on the same items."""
import collections, json
from pathlib import Path
import score as S

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "results" / "claude_sample"
rows = {}
for l in open(ROOT / "data" / "gintieval.jsonl", encoding="utf-8"):
    r = json.loads(l); rows[r["id"]] = r
ids = json.load(open(SAMPLE / "ids.json"))
TASKS = ["w2d", "d2w", "arith", "frac", "conv"]
LARGE = [("haiku", "Claude Haiku 5.5"), ("sonnet", "Claude Sonnet 5.5"), ("opus", "Claude Opus 5.5"), ("fable", "Claude Fable 5.2")]
SMALL = [("llama-3.2-3b", "Llama 3.2 3B"), ("gemma-3-4b", "Gemma 3 4B"), ("qwen2.5-7b", "Qwen2.5 7B"),
         ("llama-3.1-8b", "Llama 3.1 8B"), ("aya-expanse-8b", "Aya Expanse 8B"), ("gemma-3-12b", "Gemma 3 12B")]


def acc(xs):
    return 100 * sum(xs) / len(xs)


def load(path, key):
    out = {}
    for l in open(path, encoding="utf-8"):
        if l.strip():
            a = json.loads(l); out[a["id"]] = str(a.get(key, ""))
    return out


def evaluate(ans):
    per, hand, other = collections.defaultdict(list), [], []
    for i in ids:
        r = rows[i]; ok = bool(S.score(r, ans.get(i, ""))["correct"])
        per[r["task"]].append(ok)
        (hand if (r["lang"] == "hi-Latn" or r["task"] == "frac") else other).append(ok)
    d = {t: acc(per[t]) for t in TASKS}
    d["avg"] = sum(d[t] for t in TASKS) / len(TASKS)
    d["hand"], d["other"], d["n_hand"], d["n_other"] = acc(hand), acc(other), len(hand), len(other)
    d["n"] = {t: len(per[t]) for t in TASKS}
    return d


res = {}
for key, name in LARGE:
    ans = load(SAMPLE / f"answers_{key}.jsonl", "output")
    assert set(ids) <= set(ans), key
    res[name] = evaluate(ans)
for key, name in SMALL:
    first = json.loads(open(ROOT / "results" / f"{key}.jsonl", encoding="utf-8").readline())
    field = "output" if "output" in first else "out"
    res[name] = evaluate(load(ROOT / "results" / f"{key}.jsonl", field))

lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
         "\\textbf{Model} & \\textbf{W2D} & \\textbf{D2W} & \\textbf{Arith} & \\textbf{Frac} & \\textbf{Conv} & \\textbf{Avg} \\\\", "\\midrule"]
for group in (LARGE, SMALL):
    for _, name in group:
        d = res[name]
        lines.append(name + " & " + " & ".join(f"{d[t]:.1f}" for t in TASKS) + f" & {d['avg']:.1f} \\\\")
    if group is LARGE:
        lines.append("\\midrule")
lines += ["\\bottomrule", "\\end{tabular}"]
(ROOT / "paper" / "generated").mkdir(parents=True, exist_ok=True)
(ROOT / "paper" / "generated" / "tab_large.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
json.dump(res, open(SAMPLE / "scores.json", "w"), indent=1)
for name, d in res.items():
    print(f"{name:18s}", " ".join(f"{d[t]:5.1f}" for t in TASKS), f"avg {d['avg']:.1f}  hand {d['hand']:.1f} (n={d['n_hand']})  other {d['other']:.1f} (n={d['n_other']})")
print(res["Claude Haiku 5.5"]["n"])
