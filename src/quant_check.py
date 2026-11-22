"""Sanity check: full-precision (bf16) vs 4-bit Llama 3.2 3B on a fixed subset.
Output of the run reported in the paper is stored in results/quant_check_llama32_3b.txt."""
import json
from collections import defaultdict
from mlx_lm import batch_generate, load
import score as S

items = [json.loads(l) for l in open('../data/gintieval.jsonl')]
outs = {json.loads(l)['id']: json.loads(l)['output'] for l in open('../results/llama-3.2-3b.jsonl')}
sel = [it for it in items if it['task'] == 'arith' and it['lang'] == 'digits'] + \
      [it for it in items if it['task'] == 'w2d' and it['lang'] in ('en', 'hi') and it['stratum'] in ('1-99', 'lakh')][::3]
m, t = load('mlx-community/Llama-3.2-3B-Instruct-bf16')
enc = [t.apply_chat_template([{"role": "user", "content": it['prompt']}], add_generation_prompt=True) for it in sel]
res = []
for i in range(0, len(sel), 16):
    res += batch_generate(m, t, enc[i:i + 16], max_tokens=48).texts
a = defaultdict(list)
for it, o in zip(sel, res):
    a[(it['task'], it['lang'], it['stratum'])].append((S.score(it, o)['correct'], S.score(it, outs[it['id']])['correct']))
for k, v in a.items():
    print(k, 'bf16', round(sum(x for x, _ in v) / len(v), 2), '4bit', round(sum(y for _, y in v) / len(v), 2), len(v))
