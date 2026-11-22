"""Check that the D2W output budget never turned a correct answer into an error.
Flags wrong D2W outputs that (i) used >= budget-1 tokens and (ii) are a prefix of an accepted answer."""
import json
from transformers import AutoTokenizer
import lexicon as L, score as S

REPO = {"llama-3.2-3b": "mlx-community/Llama-3.2-3B-Instruct-4bit", "gemma-3-4b": "mlx-community/gemma-3-4b-it-qat-4bit",
        "qwen2.5-7b": "mlx-community/Qwen2.5-7B-Instruct-4bit", "llama-3.1-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "aya-expanse-8b": "mlx-community/aya-expanse-8b-4bit", "gemma-3-12b": "mlx-community/gemma-3-12b-it-qat-4bit"}
OLD = {"llama-3.2-3b", "gemma-3-4b", "llama-3.1-8b", "aya-expanse-8b"}   # run with the earlier, looser budget (see paper, App. C)
items = {json.loads(l)["id"]: json.loads(l) for l in open("../data/gintieval.jsonl", encoding="utf-8")}
res = {}
for name, repo in REPO.items():
    try:
        outs = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(f"../results/{name}.jsonl", encoding="utf-8")}
    except FileNotFoundError:
        continue
    tok = AutoTokenizer.from_pretrained(repo)
    hit = cut = 0
    for i, it in items.items():
        if it["task"] != "d2w" or i not in outs:
            continue
        o = outs[i]
        ref_t = len(tok.encode(it["answer"], add_special_tokens=False))
        new = min(384, int(1.5 * ref_t) + 8)
        old = min(384, 3 * len(it["answer"]) + 16)
        budget = min(old, new) if False else (old if name in OLD and False else new)
        # conservative: test against the tighter (new) budget for every model
        n_out = len(tok.encode(o, add_special_tokens=False))
        if n_out >= budget - 1:
            hit += 1
            if not S.score(it, o)["correct"]:
                sk = S.skeleton(S.strict_norm((o.strip().splitlines() or [""])[0]))
                refs = {S.skeleton(L.words(it["meta"]["n"], it["lang"]))} | {S.skeleton(a) for a in L.alt_words(it["meta"]["n"], it["lang"])}
                if "\n" not in o.strip() and len(sk) >= 3 and any(r.startswith(sk) for r in refs):
                    cut += 1
    res[name] = {"outputs_reaching_budget": hit, "wrong_outputs_that_are_cut_off_correct_prefixes": cut}
    print(name, res[name], flush=True)
json.dump(res, open("../results/budget_check.json", "w"), indent=1)
