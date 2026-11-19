"""Run a model on GintiEval.

Local (Apple silicon):  python run.py --backend mlx --model mlx-community/gemma-3-4b-it-qat-4bit --name gemma-3-4b
Any OpenAI-compatible API (OpenRouter, Together, OpenAI, ...):
    export GINTI_API_KEY=...   # never hard-code keys
    python run.py --backend api --base-url https://openrouter.ai/api/v1 --model google/gemini-2.5-pro --name gemini-2.5-pro
Runs are resumable: finished item ids are skipped. Decoding is greedy (temperature 0).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def max_tokens(item, tok=None):
    """Output budget. Only the first output line of D2W is scored, so the budget only has to
    fit a correct answer: 1.5x the reference length in the model's own tokens, plus 8.
    (Without a tokenizer we fall back to 3 tokens per reference character plus 16.)"""
    if item["task"] != "d2w":
        return 48
    if tok is None:
        return min(384, 3 * len(item["answer"]) + 16)
    return min(384, int(1.5 * len(tok.encode(item["answer"], add_special_tokens=False))) + 8)


def run_mlx(model_id, items, out, bs):
    from mlx_lm import batch_generate, load
    model, tok = load(model_id)
    enc = []
    for it in items:
        ids = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], add_generation_prompt=True)
        enc.append((len(ids), ids, it, max_tokens(it, tok)))
    enc.sort(key=lambda x: (x[3], x[0]))
    t0, done = time.time(), 0
    for i in range(0, len(enc), bs):
        chunk = enc[i:i + bs]
        res = batch_generate(model, tok, [c[1] for c in chunk], max_tokens=[c[3] for c in chunk])
        for c, text in zip(chunk, res.texts):
            out.write(json.dumps({"id": c[2]["id"], "output": text}, ensure_ascii=False) + "\n")
        out.flush()
        done += len(chunk)
        if (i // bs) % 20 == 0:
            el = time.time() - t0
            print(f"{done}/{len(enc)}  {el:.0f}s  eta {el / done * (len(enc) - done):.0f}s", flush=True)


def run_api(model_id, items, out, base_url, workers):
    import concurrent.futures as cf
    import urllib.request
    key = os.environ["GINTI_API_KEY"]

    def call(it):
        body = json.dumps({"model": model_id, "temperature": 0, "max_tokens": max_tokens(it),
                           "messages": [{"role": "user", "content": it["prompt"]}]}).encode()
        for attempt in range(6):
            try:
                req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                             headers={"Authorization": f"Bearer {key}",
                                                      "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return it["id"], json.load(r)["choices"][0]["message"]["content"] or ""
            except Exception as e:  # retry with backoff
                err = repr(e)
                time.sleep(2 ** attempt)
        return it["id"], None if err else ""

    with cf.ThreadPoolExecutor(workers) as ex:
        for n, (iid, text) in enumerate(ex.map(call, items), 1):
            if text is not None:
                out.write(json.dumps({"id": iid, "output": text}, ensure_ascii=False) + "\n")
                out.flush()
            if n % 200 == 0:
                print(f"{n}/{len(items)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mlx", "api"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True, help="short name used for the results file")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data", default=str(ROOT / "data" / "gintieval.jsonl"))
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    if a.limit:
        import random
        random.Random(0).shuffle(items)
        items = items[:a.limit]
    path = ROOT / "results" / f"{a.name}.jsonl"
    path.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if path.exists() else set()
    items = [it for it in items if it["id"] not in done]
    print(f"{a.name}: {len(items)} items to run ({len(done)} already done)", flush=True)
    meta = ROOT / "results" / f"{a.name}.meta.json"
    meta.write_text(json.dumps({"name": a.name, "model": a.model, "backend": a.backend}))
    with open(path, "a", encoding="utf-8") as out:
        if a.backend == "mlx":
            run_mlx(a.model, items, out, a.batch_size)
        else:
            run_api(a.model, items, out, a.base_url, a.workers)


if __name__ == "__main__":
    main()
