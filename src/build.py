"""Build the GintiEval dataset (deterministic; seed fixed)."""
from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

import lexicon as L

SEED = 20261115
OUT = Path(__file__).resolve().parent.parent / "data" / "gintieval.jsonl"

NAME = {**{k: v[0] for k, v in L.LANGS.items()},
        "hi-Latn": "Hindi written in the Latin script (romanized Hindi)",
        "en-IN": "Indian English (using lakh and crore)",
        "en": "English"}
SCRIPT = {k: v[1] for k, v in L.LANGS.items()}

DIGITS_ONLY = "Output only the number in digits (0-9), with no commas, words or explanation."
DEC_ONLY = "Output only the number in digits (0-9), using a decimal point if needed, with no commas, words or explanation."

STRATA = [("1-99", 1, 99), ("100-999", 100, 999), ("1e3-1e5", 1000, 99_999),
          ("lakh", 100_000, 9_999_999), ("crore", 10_000_000, 999_999_999)]


def sample_numbers(rng):
    nums = [("1-99", n) for n in range(1, 100)]
    for name, lo, hi in STRATA[1:]:
        seen = set()
        while len(seen) < 50:
            # log-uniform so that every magnitude inside the stratum is covered
            import math
            n = int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))
            if lo <= n <= hi:
                seen.add(n)
        nums += [(name, n) for n in sorted(seen)]
    return nums


def frac_str(v):
    if isinstance(v, str):
        return v
    v = Fraction(v)
    if v.denominator == 1:
        return str(v.numerator)
    return str(float(v)).rstrip("0").rstrip(".")


def main():
    rng = random.Random(SEED)
    items = []

    def add(task, lang, stratum, prompt, answer, atype, **meta):
        items.append(dict(id=f"{task}-{lang}-{len(items):05d}", task=task, lang=lang, stratum=stratum,
                          prompt=prompt, answer=answer, answer_type=atype, meta=meta))

    nums = sample_numbers(rng)

    # ---- T1: words -> digits (reading) ------------------------------------
    for lang in [*L.LANGS, "hi-Latn", "en-IN", "en"]:
        for stratum, n in nums:
            w = L.words(n, lang)
            add("w2d", lang, stratum,
                f"The following is a number written in words in {NAME[lang]}:\n{w}\n"
                f"Write this number in digits. {DIGITS_ONLY}", str(n), "int", n=n, words=w)

    # ---- T2: digits -> words (verbalization) ------------------------------
    for lang in L.LANGS:
        for stratum, n in nums:
            add("d2w", lang, stratum,
                f"Write the number {n} in words in {NAME[lang]}, using the {SCRIPT[lang]} script. "
                "Output only the number words, with no digits and no explanation.",
                L.words(n, lang), "words", n=n)

    # ---- T3: arithmetic over number words ---------------------------------
    probs = []
    for i in range(100):
        small = i < 50
        lo, hi = (11, 99) if small else (1000, 999_999)
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        op = "+" if i % 2 == 0 else "-"
        if op == "-" and a < b:
            a, b = b, a
        probs.append(("2-digit" if small else "4-6-digit", a, b, op))
    for lang in ["digits", *L.LANGS, "hi-Latn", "en"]:
        for stratum, a, b, op in probs:
            if lang == "digits":
                head, wa, wb = "The two numbers below are written in digits.", str(a), str(b)
            else:
                head = f"The two numbers below are written in words in {NAME[lang]}."
                wa, wb = L.words(a, lang), L.words(b, lang)
            add("arith", lang, stratum, f"{head}\nA: {wa}\nB: {wb}\nCompute A {op} B. {DIGITS_ONLY}",
                str(a + b if op == "+" else a - b), "int", a=a, b=b, op=op)

    # ---- T4: vernacular fraction words -------------------------------------
    for lang in L.FRACTION_LEX:
        for phrase, val, kind, meta in L.fraction_items(lang):
            if kind == "time":
                p = (f'In {L.FRACTION_LANG_NAME[lang]}, what clock time does the phrase "{phrase}" refer to? '
                     "Output only the time in H:MM format (12-hour clock, no AM/PM), with no explanation.")
                add("frac", lang, kind, p, val, "time", phrase=phrase, **meta)
            else:
                p = (f'In {L.FRACTION_LANG_NAME[lang]}, what number does the phrase "{phrase}" denote? {DEC_ONLY}')
                add("frac", lang, kind, p, frac_str(val), "decimal", phrase=phrase, **meta)

    # ---- T5: Indian <-> international system conversion --------------------
    units = {"thousand": 10**3, "lakh": 10**5, "million": 10**6, "crore": 10**7, "billion": 10**9}
    pairs = [("lakh", "million"), ("million", "lakh"), ("crore", "million"), ("million", "crore"),
             ("crore", "billion"), ("billion", "crore"), ("lakh", "thousand"), ("billion", "lakh")]
    vals = [1, 2, 5, 7.5, 12, 25, 40, 75, 150, 300, 450, 2.5, 0.5, 90, 600]
    for src, dst in pairs:
        for v in vals:
            ans = Fraction(str(v)) * units[src] / units[dst]
            add("conv", "en", f"{src}->{dst}",
                f"How many {dst} is {frac_str(Fraction(str(v)))} {src}? {DEC_ONLY}",
                frac_str(ans), "decimal", v=frac_str(Fraction(str(v))), src=src, dst=dst)
    gnums = [n for s, n in nums if s in ("lakh", "crore")]
    rng.shuffle(gnums)
    for n in gnums[:60]:
        add("conv", "en", "to-indian-grouping",
            f"Rewrite the number {n} using the Indian digit-grouping system (commas at lakh and crore). "
            "Output only the formatted number, with no explanation.", L.indian_group(n), "grouped", n=n)
    for n in gnums[60:]:
        add("conv", "en", "from-indian-grouping",
            f"The number {L.indian_group(n)} is written with Indian digit grouping. Rewrite it using "
            "international digit grouping (a comma every three digits). Output only the formatted number, "
            "with no explanation.", f"{n:,}", "grouped", n=n)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter((it["task"], it["lang"]) for it in items)
    for k in sorted(c):
        print(k, c[k])
    print("TOTAL", len(items))


if __name__ == "__main__":
    main()
