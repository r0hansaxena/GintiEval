"""Score all result files and emit summary.json + LaTeX tables/figures.

Every number that appears in the paper is produced here; nothing is typed by hand.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import lexicon as L
import score as S

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "paper" / "generated"
GEN.mkdir(parents=True, exist_ok=True)

MODELS = [  # (results name, display name, params, family note)
    ("llama-3.2-3b", "Llama 3.2 3B", "3B"),
    ("gemma-3-4b", "Gemma 3 4B", "4B"),
    ("qwen2.5-7b", "Qwen2.5 7B", "7B"),
    ("llama-3.1-8b", "Llama 3.1 8B", "8B"),
    ("aya-expanse-8b", "Aya Expanse 8B", "8B"),
    ("gemma-3-12b", "Gemma 3 12B", "12B"),
]
INDIC = ["hi", "bn", "mr", "gu", "ta", "te"]
FUSED, COMP = ["hi", "bn", "mr", "gu"], ["ta", "te"]
W2D_LANGS = INDIC + ["hi-Latn", "en-IN", "en"]
LABEL = {"hi": "Hindi", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "ta": "Tamil", "te": "Telugu",
         "hi-Latn": "Hindi (rom.)", "en-IN": "English (IN)", "en": "English", "digits": "Digits"}
SHORT = {"hi": "hi", "bn": "bn", "mr": "mr", "gu": "gu", "ta": "ta", "te": "te", "hi-Latn": "hi-Latn",
         "en-IN": "en-IN", "en": "en", "digits": "digits"}
STRATA = ["1-99", "100-999", "1e3-1e5", "lakh", "crore"]
STRATA_LABEL = {"1-99": "1--99", "100-999": "100--999", "1e3-1e5": "$10^3$--$10^5$", "lakh": "lakh", "crore": "crore"}


def pct(x):
    return "--" if x is None else f"{100 * x:.1f}"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def boot_ci(vals, n=2000, seed=0):
    rng = random.Random(seed)
    vals = list(vals)
    if not vals:
        return (None, None)
    ms = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def mcnemar(a, b):
    """Exact two-sided McNemar p-value for paired boolean lists."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def bold_best(rows, cols):
    """rows: list of dict col->value; returns formatted strings with column maxima in bold."""
    best = {c: max((r[c] for r in rows if r[c] is not None), default=None) for c in cols}
    out = []
    for r in rows:
        out.append({c: (f"\\textbf{{{pct(r[c])}}}" if r[c] is not None and best[c] is not None
                        and abs(r[c] - best[c]) < 1e-12 else pct(r[c])) for c in cols})
    return out


def write(name, text):
    (GEN / name).write_text(text, encoding="utf-8")


def main():
    items = {}
    for l in open(ROOT / "data" / "gintieval.jsonl", encoding="utf-8"):
        it = json.loads(l)
        items[it["id"]] = it
    n_items = len(items)

    scored = {}  # model -> id -> score dict
    avail = []
    for name, disp, size in MODELS:
        p = ROOT / "results" / f"{name}.jsonl"
        if not p.exists():
            continue
        outs = {}
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            outs[r["id"]] = r["output"]
        if len(outs) < n_items:
            print(f"skip {name}: {len(outs)}/{n_items} items")
            continue
        scored[name] = {i: {**S.score(items[i], outs[i]), "raw": outs[i]} for i in items}
        avail.append((name, disp, size))
    print("models:", [a[0] for a in avail])

    def acc(model, pred, key="correct"):
        vals = [float(scored[model][i][key]) for i, it in items.items() if pred(it)]
        return mean(vals)

    summ = {"n_items": n_items, "models": [a[1] for a in avail], "per_model": {}}

    # ------------------------------------------------------------------ main table
    main_rows = []
    for name, disp, size in avail:
        r = {}
        r["w2d"] = mean(acc(name, lambda it, l=l: it["task"] == "w2d" and it["lang"] == l) for l in INDIC)
        r["w2d_rom"] = acc(name, lambda it: it["task"] == "w2d" and it["lang"] == "hi-Latn")
        r["w2d_en"] = acc(name, lambda it: it["task"] == "w2d" and it["lang"] == "en")
        r["d2w"] = mean(acc(name, lambda it, l=l: it["task"] == "d2w" and it["lang"] == l) for l in INDIC)
        r["arith"] = mean(acc(name, lambda it, l=l: it["task"] == "arith" and it["lang"] == l) for l in INDIC)
        r["arith_dig"] = acc(name, lambda it: it["task"] == "arith" and it["lang"] == "digits")
        r["frac"] = mean(acc(name, lambda it, l=l: it["task"] == "frac" and it["lang"] == l)
                         for l in L.FRACTION_LEX)
        r["conv"] = acc(name, lambda it: it["task"] == "conv")
        r["overall"] = mean([r["w2d"], r["d2w"], r["arith"], r["frac"], r["conv"]])
        main_rows.append(r)
        summ["per_model"][disp] = {k: round(100 * v, 1) for k, v in r.items()}
    cols = ["w2d", "d2w", "arith", "frac", "conv", "overall"]
    fm = bold_best(main_rows, cols)
    tex = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
           "\\textbf{Model} & \\textbf{W2D} & \\textbf{D2W} & \\textbf{Arith} & \\textbf{Frac} & \\textbf{Conv} & \\textbf{Avg} \\\\",
           "\\midrule"]
    for (name, disp, size), f in zip(avail, fm):
        tex.append(f"{disp} & " + " & ".join(f[c] for c in cols) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write("tab_main.tex", "\n".join(tex))

    # ------------------------------------------------------------------ W2D per language
    rows = [{l: acc(name, lambda it, l=l: it["task"] == "w2d" and it["lang"] == l) for l in W2D_LANGS}
            for name, _, _ in avail]
    fm = bold_best(rows, W2D_LANGS)
    tex = ["\\begin{tabular}{l" + "r" * len(W2D_LANGS) + "}", "\\toprule",
           "\\textbf{Model} & " + " & ".join(f"\\textbf{{{SHORT[l]}}}" for l in W2D_LANGS) + " \\\\", "\\midrule"]
    for (name, disp, size), f in zip(avail, fm):
        tex.append(f"{disp} & " + " & ".join(f[l] for l in W2D_LANGS) + " \\\\")
    avg = {l: mean(r[l] for r in rows) for l in W2D_LANGS}
    tex += ["\\midrule", "\\textit{Mean} & " + " & ".join(pct(avg[l]) for l in W2D_LANGS) + " \\\\",
            "\\bottomrule", "\\end{tabular}"]
    write("tab_w2d_lang.tex", "\n".join(tex))
    summ["w2d_lang_mean"] = {l: round(100 * avg[l], 1) for l in W2D_LANGS}
    summ["w2d_lang"] = {disp: {l: round(100 * r[l], 1) for l in W2D_LANGS} for (_, disp, _), r in zip(avail, rows)}

    # ------------------------------------------------------------------ W2D by magnitude
    def strat_acc(name, langs, s):
        return mean(acc(name, lambda it, l=l: it["task"] == "w2d" and it["lang"] == l and it["stratum"] == s)
                    for l in langs)
    tex = ["\\begin{tabular}{l" + "r" * len(STRATA) + "}", "\\toprule",
           "\\textbf{Model} & " + " & ".join(f"\\textbf{{{STRATA_LABEL[s]}}}" for s in STRATA) + " \\\\", "\\midrule"]
    rows = [{s: strat_acc(name, INDIC, s) for s in STRATA} for name, _, _ in avail]
    for (name, disp, size), f in zip(avail, bold_best(rows, STRATA)):
        tex.append(f"{disp} & " + " & ".join(f[s] for s in STRATA) + " \\\\")
    avg = {s: mean(r[s] for r in rows) for s in STRATA}
    tex += ["\\midrule", "\\textit{Mean} & " + " & ".join(pct(avg[s]) for s in STRATA) + " \\\\"]
    en = {s: mean(strat_acc(name, ["en"], s) for name, _, _ in avail) for s in STRATA}
    tex += ["\\textit{Mean (English)} & " + " & ".join(pct(en[s]) for s in STRATA) + " \\\\",
            "\\bottomrule", "\\end{tabular}"]
    write("tab_w2d_mag.tex", "\n".join(tex))
    summ["w2d_mag_mean"] = {s: round(100 * avg[s], 1) for s in STRATA}
    summ["w2d_mag_mean_en"] = {s: round(100 * en[s], 1) for s in STRATA}

    # ------------------------------------------------------------------ fused vs compositional (1-99)
    fc = {}
    tex = ["\\begin{tabular}{lrrrr}", "\\toprule",
           " & \\multicolumn{2}{c}{\\textbf{W2D (1--99)}} & \\multicolumn{2}{c}{\\textbf{D2W (1--99)}} \\\\",
           "\\textbf{Model} & \\textbf{Fused} & \\textbf{Comp.} & \\textbf{Fused} & \\textbf{Comp.} \\\\", "\\midrule"]
    allr = []
    for name, disp, size in avail:
        r = {}
        for task in ("w2d", "d2w"):
            for lab, langs in (("fused", FUSED), ("comp", COMP)):
                r[f"{task}_{lab}"] = mean(acc(name, lambda it, l=l, task=task: it["task"] == task and it["lang"] == l
                                              and it["stratum"] == "1-99") for l in langs)
        allr.append(r)
        fc[disp] = {k: round(100 * v, 1) for k, v in r.items()}
        tex.append(f"{disp} & " + " & ".join(pct(r[k]) for k in ("w2d_fused", "w2d_comp", "d2w_fused", "d2w_comp")) + " \\\\")
    m = {k: mean(r[k] for r in allr) for k in ("w2d_fused", "w2d_comp", "d2w_fused", "d2w_comp")}
    tex += ["\\midrule", "\\textit{Mean} & " + " & ".join(pct(m[k]) for k in ("w2d_fused", "w2d_comp", "d2w_fused", "d2w_comp")) + " \\\\",
            "\\bottomrule", "\\end{tabular}"]
    write("tab_fused.tex", "\n".join(tex))
    fc["mean"] = {k: round(100 * v, 1) for k, v in m.items()}
    summ["fused_vs_comp"] = fc

    # per-number accuracy (pooled over models), Hindi and fused langs, + "un-" numbers
    pern = {}
    for l in INDIC + ["hi-Latn", "en"]:
        d = defaultdict(list)
        for i, it in items.items():
            if it["task"] == "w2d" and it["lang"] == l and it["stratum"] == "1-99":
                for name, _, _ in avail:
                    d[it["meta"]["n"]].append(float(scored[name][i]["correct"]))
        pern[l] = {n: mean(v) for n, v in sorted(d.items())}
    summ["w2d_per_number_hi"] = {n: round(100 * v, 1) for n, v in pern["hi"].items()}
    # subtractive ("one less than the next decade") forms: x9 for 19..79 in the fused languages
    un = [19, 29, 39, 49, 59, 69, 79]
    summ["un_forms"] = {}
    for l in FUSED:
        a = mean(pern[l][n] for n in un)
        b = mean(v for n, v in pern[l].items() if n not in un and n > 10)
        summ["un_forms"][l] = {"un": round(100 * a, 1), "other_11_99": round(100 * b, 1)}
    coords = lambda l: " ".join(f"({n},{100 * v:.1f})" for n, v in pern[l].items())
    write("fig_pernumber.tex", "\n".join([
        "\\begin{tikzpicture}", "\\begin{axis}[width=\\columnwidth,height=4.6cm,xmin=0,xmax=100,ymin=0,ymax=100,",
        "xlabel={Number (1--99)},ylabel={W2D accuracy (\\%)},xtick={10,20,...,90},",
        "tick label style={font=\\scriptsize},label style={font=\\small},legend style={font=\\scriptsize,at={(0.5,1.03)},anchor=south,legend columns=3},",
        "ymajorgrids,grid style={gray!25}]",
        f"\\addplot[mark=none,thick,blue] coordinates {{{coords('hi')}}};", "\\addlegendentry{Hindi}",
        f"\\addplot[mark=none,thick,red,densely dashed] coordinates {{{coords('ta')}}};", "\\addlegendentry{Tamil}",
        f"\\addplot[mark=none,thick,black!60,dotted] coordinates {{{coords('en')}}};", "\\addlegendentry{English}",
        "\\end{axis}", "\\end{tikzpicture}"]))

    # ------------------------------------------------------------------ D2W per language
    tex = ["\\begin{tabular}{l" + "r" * len(INDIC) + "r}", "\\toprule",
           "\\textbf{Model} & " + " & ".join(f"\\textbf{{{l}}}" for l in INDIC) + " & \\textbf{Avg} \\\\", "\\midrule"]
    rows, srows = [], []
    for name, disp, size in avail:
        r = {l: acc(name, lambda it, l=l: it["task"] == "d2w" and it["lang"] == l) for l in INDIC}
        r["avg"] = mean(r.values())
        s = {l: acc(name, lambda it, l=l: it["task"] == "d2w" and it["lang"] == l, "strict") for l in INDIC}
        s["avg"] = mean(s.values())
        rows.append(r)
        srows.append(s)
    cl = INDIC + ["avg"]
    for (name, disp, size), f, s in zip(avail, bold_best(rows, cl), srows):
        tex.append(f"{disp} & " + " & ".join(f"{f[l]}" for l in cl) + " \\\\")
        tex.append("\\quad\\textit{\\small strict} & " + " & ".join(f"{{\\small\\color{{gray}}{pct(s[l])}}}" for l in cl) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write("tab_d2w.tex", "\n".join(tex))
    summ["d2w_lang"] = {disp: {l: round(100 * r[l], 1) for l in cl} for (_, disp, _), r in zip(avail, rows)}
    summ["d2w_lang_strict"] = {disp: {l: round(100 * r[l], 1) for l in cl} for (_, disp, _), r in zip(avail, srows)}
    summ["d2w_charsim"] = {disp: round(100 * acc(name, lambda it: it["task"] == "d2w", "sim"), 1) for name, disp, _ in avail}

    # ------------------------------------------------------------------ arithmetic
    AL = ["digits", "en"] + INDIC + ["hi-Latn"]
    tex = ["\\begin{tabular}{l" + "r" * len(AL) + "}", "\\toprule",
           "\\textbf{Model} & " + " & ".join(f"\\textbf{{{SHORT[l]}}}" for l in AL) + " \\\\", "\\midrule"]
    rows = [{l: acc(name, lambda it, l=l: it["task"] == "arith" and it["lang"] == l) for l in AL} for name, _, _ in avail]
    for (name, disp, size), f in zip(avail, bold_best(rows, AL)):
        tex.append(f"{disp} & " + " & ".join(f[l] for l in AL) + " \\\\")
    avg = {l: mean(r[l] for r in rows) for l in AL}
    tex += ["\\midrule", "\\textit{Mean} & " + " & ".join(pct(avg[l]) for l in AL) + " \\\\", "\\bottomrule", "\\end{tabular}"]
    write("tab_arith.tex", "\n".join(tex))
    summ["arith_mean"] = {l: round(100 * avg[l], 1) for l in AL}
    summ["arith"] = {disp: {l: round(100 * r[l], 1) for l in AL} for (_, disp, _), r in zip(avail, rows)}
    # paired test: digits vs each Indic language, pooled over models (same 100 problems)
    key = lambda it: (it["meta"]["a"], it["meta"]["b"], it["meta"]["op"])
    by = defaultdict(dict)
    for i, it in items.items():
        if it["task"] == "arith":
            by[it["lang"]][key(it)] = i
    pv = {}
    for l in INDIC + ["hi-Latn", "en"]:
        a, b = [], []
        for name, _, _ in avail:
            for k, i in by["digits"].items():
                a.append(scored[name][i]["correct"])
                b.append(scored[name][by[l][k]]["correct"])
        pv[l] = mcnemar(a, b)
    summ["arith_mcnemar_vs_digits"] = {l: float(f"{p:.2e}") for l, p in pv.items()}
    # is the arithmetic failure explained by reading? P(correct arith | both operands readable) needs w2d of
    # the operands, which are not in the w2d sample; instead report by operand size.
    summ["arith_by_size"] = {}
    for s in ("2-digit", "4-6-digit"):
        summ["arith_by_size"][s] = {
            "digits": round(100 * mean(acc(n, lambda it: it["task"] == "arith" and it["lang"] == "digits" and it["stratum"] == s) for n, _, _ in avail), 1),
            "indic": round(100 * mean(mean(acc(n, lambda it, l=l: it["task"] == "arith" and it["lang"] == l and it["stratum"] == s) for l in INDIC) for n, _, _ in avail), 1)}

    # ------------------------------------------------------------------ fractions
    FL = list(L.FRACTION_LEX)
    tex = ["\\begin{tabular}{l" + "r" * len(FL) + "rrr}", "\\toprule",
           " & \\multicolumn{" + str(len(FL)) + "}{c}{\\textbf{By language}} & \\multicolumn{3}{c}{\\textbf{By item type}} \\\\",
           "\\textbf{Model} & " + " & ".join(f"\\textbf{{{SHORT[l]}}}" for l in FL) + " & \\textbf{bare} & \\textbf{scaled} & \\textbf{time} \\\\", "\\midrule"]
    rows = []
    for name, disp, size in avail:
        r = {l: acc(name, lambda it, l=l: it["task"] == "frac" and it["lang"] == l) for l in FL}
        for k in ("bare", "scaled", "time"):
            r[k] = acc(name, lambda it, k=k: it["task"] == "frac" and it["stratum"] == k)
        rows.append(r)
    cl = FL + ["bare", "scaled", "time"]
    for (name, disp, size), f in zip(avail, bold_best(rows, cl)):
        tex.append(f"{disp} & " + " & ".join(f[c] for c in cl) + " \\\\")
    avg = {c: mean(r[c] for r in rows) for c in cl}
    tex += ["\\midrule", "\\textit{Mean} & " + " & ".join(pct(avg[c]) for c in cl) + " \\\\", "\\bottomrule", "\\end{tabular}"]
    write("tab_frac.tex", "\n".join(tex))
    summ["frac_mean"] = {c: round(100 * avg[c], 1) for c in cl}
    WORDS = [("one_q", "1\\textonequarter"), ("one_h", "1\\textonehalf"), ("two_h", "2\\textonehalf"),
             ("plus_q", "$N{+}\\frac14$"), ("minus_q", "$N{-}\\frac14$"), ("plus_h", "$N{+}\\frac12$")]
    fw = {}
    for w, _ in WORDS:
        fw[w] = round(100 * mean(acc(name, lambda it, w=w: it["task"] == "frac" and it["meta"]["word"] == w
                                     and it["stratum"] != "time") for name, _, _ in avail), 1)
    summ["frac_by_word_mean"] = fw
    best = max(avail, key=lambda a: summ["per_model"][a[1]]["frac"])
    summ["frac_by_word_best"] = {"model": best[1], **{w: round(100 * acc(best[0], lambda it, w=w: it["task"] == "frac"
                                 and it["meta"]["word"] == w and it["stratum"] != "time"), 1) for w, _ in WORDS}}
    tex = ["\\begin{tabular}{lrr}", "\\toprule", "\\textbf{Word class} & \\textbf{Mean} & \\textbf{Best model} \\\\", "\\midrule"]
    HI = {"one_q": "sawa", "one_h": "dedh", "two_h": "dhai", "plus_q": "sawa $N$", "minus_q": "paune $N$", "plus_h": "saadhe $N$"}
    for w, sym in WORDS:
        tex.append(f"\\textit{{{HI[w]}}} ({sym}) & {fw[w]:.1f} & {summ['frac_by_word_best'][w]:.1f} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write("tab_frac_word.tex", "\n".join(tex))

    # ------------------------------------------------------------------ conversion
    CS = ["lakh->million", "million->lakh", "crore->million", "million->crore", "crore->billion", "billion->crore",
          "lakh->thousand", "billion->lakh", "to-indian-grouping", "from-indian-grouping"]
    CSL = {s: s.replace("->", "$\\to$") for s in CS}
    CSL["to-indian-grouping"], CSL["from-indian-grouping"] = "to Indian grouping", "from Indian grouping"
    tex = ["\\begin{tabular}{l" + "r" * len(avail) + "}", "\\toprule",
           "\\textbf{Conversion} & " + " & ".join(f"\\textbf{{{d}}}" for _, d, _ in avail) + " \\\\", "\\midrule"]
    cm = {}
    for s in CS:
        vals = [acc(name, lambda it, s=s: it["task"] == "conv" and it["stratum"] == s) for name, _, _ in avail]
        cm[s] = round(100 * mean(vals), 1)
        tex.append(f"{CSL[s]} & " + " & ".join(pct(v) for v in vals) + " \\\\")
        if s == "billion->lakh":
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write("tab_conv.tex", "\n".join(tex))
    summ["conv_mean"] = cm

    # ------------------------------------------------------------------ error taxonomy (pooled)
    def taxonomy(pred_filter, fn):
        c = Counter()
        for name, _, _ in avail:
            for i, it in items.items():
                sc = scored[name][i]
                if pred_filter(it) and not sc["correct"]:
                    p = S.extract_number(sc["raw"])
                    c[fn(it, p)] += 1
        return c
    w2d_err = taxonomy(lambda it: it["task"] == "w2d" and it["lang"] in INDIC,
                       lambda it, p: S.w2d_error(it["meta"]["n"], p))
    frac_err = taxonomy(lambda it: it["task"] == "frac" and it["answer_type"] == "decimal", S.frac_error)
    conv_err = taxonomy(lambda it: it["task"] == "conv" and it["answer_type"] == "decimal", S.conv_error)
    def share(c):
        t = sum(c.values())
        return {k: round(100 * v / t, 1) for k, v in c.most_common()} | {"_n": t}
    summ["errors"] = {"w2d": share(w2d_err), "frac": share(frac_err), "conv": share(conv_err)}
    ELAB = {"multi-group": "several groups wrong", "one-group": "one group wrong", "length": "wrong digit count",
            "decade-shift": "decade shift ($\\pm$10 in a 2-digit group)", "unit-digit": "unit digit of a 2-digit group",
            "dropped-zero-padding": "groups concatenated without zero padding",
            "scale-substitution": "lakh/crore re-scaled (e.g.\\ lakh as million)", "magnitude": "power-of-ten slip",
            "no-number": "no number produced", "other": "other",
            "fraction-ignored": "fraction word ignored (rounded to $N$ or $N{+}1$)",
            "wrong-direction": "wrong direction (e.g.\\ $N{+}\\frac14$ for $N{-}\\frac14$)",
            "fraction-on-unit": "fraction applied to the unit, not the multiplier",
            "wrong-fraction": "different fraction used", "identity (units treated as equal)": "units treated as equal",
            "inverse factor": "inverse factor", "power-of-ten slip": "power-of-ten slip"}
    def err_table(c, fname):
        t = sum(c.values())
        tex = ["\\begin{tabular}{lr}", "\\toprule", "\\textbf{Error type} & \\textbf{\\%} \\\\", "\\midrule"]
        for k, v in c.most_common():
            tex.append(f"{ELAB.get(k, k)} & {100 * v / t:.1f} \\\\")
        tex += ["\\midrule", f"\\textit{{errors analysed}} & {t} \\\\", "\\bottomrule", "\\end{tabular}"]
        write(fname, "\n".join(tex))
    err_table(w2d_err, "tab_err_w2d.tex")
    err_table(frac_err, "tab_err_frac.tex")
    err_table(conv_err, "tab_err_conv.tex")

    # decade-shift share among 1-99 errors of fused vs compositional languages
    def e99(langs):
        return taxonomy(lambda it: it["task"] == "w2d" and it["lang"] in langs and it["stratum"] == "1-99",
                        lambda it, p: S.w2d_error(it["meta"]["n"], p))
    summ["errors"]["w2d_1_99_fused"] = share(e99(FUSED))
    summ["errors"]["w2d_1_99_comp"] = share(e99(COMP))

    # ------------------------------------------------------------------ native vs romanized Hindi (paired)
    def paired(task, l1, l2, keyf):
        m1 = {keyf(it): i for i, it in items.items() if it["task"] == task and it["lang"] == l1}
        m2 = {keyf(it): i for i, it in items.items() if it["task"] == task and it["lang"] == l2}
        a, b = [], []
        for name, _, _ in avail:
            for k in m1:
                if k in m2:
                    a.append(scored[name][m1[k]]["correct"])
                    b.append(scored[name][m2[k]]["correct"])
        return round(100 * mean(map(float, a)), 1), round(100 * mean(map(float, b)), 1), float(f"{mcnemar(a, b):.2e}")
    summ["hi_native_vs_roman"] = {
        "w2d": paired("w2d", "hi", "hi-Latn", lambda it: it["meta"]["n"]),
        "frac": paired("frac", "hi", "hi-Latn", lambda it: (it["meta"]["word"], it["meta"]["n"], it["meta"]["mult"], it["stratum"])),
        "arith": paired("arith", "hi", "hi-Latn", key)}
    summ["w2d_en_vs_enIN"] = paired("w2d", "en", "en-IN", lambda it: it["meta"]["n"])
    summ["w2d_fused_vs_comp_1_99_mcnemar"] = None

    # ------------------------------------------------------------------ CIs for the overall W2D number
    summ["ci"] = {}
    for name, disp, _ in avail:
        vals = [float(scored[name][i]["correct"]) for i, it in items.items() if it["task"] == "w2d" and it["lang"] in INDIC]
        lo, hi = boot_ci(vals)
        summ["ci"][disp] = {"w2d_indic": [round(100 * lo, 1), round(100 * hi, 1)]}

    # ------------------------------------------------------------------ dataset statistics table
    c = Counter((it["task"], it["lang"]) for it in items.values())
    TL = [("w2d", "Words$\\to$Digits (W2D)", "Reading a number written in words", "Lexical knowledge, place value", W2D_LANGS),
          ("d2w", "Digits$\\to$Words (D2W)", "Writing a number in words", "Lexical recall, orthography", INDIC),
          ("arith", "Arithmetic (Arith)", "Adding or subtracting two numbers given in words", "Reading, calculation", ["digits", "en"] + INDIC + ["hi-Latn"]),
          ("frac", "Fraction Words (Frac)", "Resolving quarter/half words, scaled values and clock times", "Vernacular numeracy", FL),
          ("conv", "System Conversion (Conv)", "Lakh/crore vs.\\ million/billion; digit grouping", "Scale knowledge", ["en"])]
    tex = ["\\begin{tabular}{p{3.6cm}p{5.6cm}p{3.6cm}r}", "\\toprule",
           "\\textbf{Task (\\% of Total)} & \\textbf{Description} & \\textbf{Primary Skills Tested} & \\textbf{Items} \\\\", "\\midrule"]
    for t, lab, desc, skill, ls in TL:
        tot = sum(c[(t, l)] for l in ls)
        tex.append(f"\\textbf{{{lab}}} ({100 * tot / n_items:.0f}\\%) & {desc} & {skill} & {tot:,} \\\\")
    tex += ["\\midrule", f"\\textbf{{Total}} & & & \\textbf{{{n_items:,}}} \\\\", "\\bottomrule", "\\end{tabular}"]
    write("tab_stats.tex", "\n".join(tex))

    # cross-reference agreement of the verbalizers (validity statistic)
    agree = {}
    for l in INDIC:
        tot = ok = sk = 0
        for n in range(1, 100):
            for a in L.alt_words(n, l)[:1]:
                tot += 1
                ok += S.strict_norm(a) == S.strict_norm(L.words(n, l))
                sk += S.skeleton(a) == S.skeleton(L.words(n, l))
        agree[l] = {"exact": round(100 * ok / tot, 1), "skeleton": round(100 * sk / tot, 1)}
    summ["verbalizer_agreement_1_99"] = agree

    # qualitative examples: a few wrong outputs of the best model
    bestm = max(avail, key=lambda a: summ["per_model"][a[1]]["overall"])
    ex = defaultdict(list)
    rng = random.Random(1)
    ids = list(items)
    rng.shuffle(ids)
    for i in ids:
        it, sc = items[i], scored[bestm[0]][i]
        if not sc["correct"] and len(ex[(it["task"], it["lang"])]) < 4:
            ex[(it["task"], it["lang"])].append({"id": i, "prompt": it["prompt"], "gold": it["answer"], "out": sc["raw"][:200]})
    summ["best_model"] = bestm[1]
    summ["examples_best_model"] = {f"{k[0]}|{k[1]}": v for k, v in ex.items()}

    write("macros.tex", "\n".join([
        f"\\newcommand{{\\nitems}}{{{n_items:,}}}",
        f"\\newcommand{{\\agreehi}}{{{agree['hi']['exact']:.0f}}}",
        f"\\newcommand{{\\nmodels}}{{{len(avail)}}}"]))
    # ---------------- budget check: was any wrong D2W output merely a cut-off correct answer?
    cut = {}
    for name, disp, _ in avail:
        k = 0
        for i, it in items.items():
            if it["task"] == "d2w" and not scored[name][i]["correct"]:
                o = S.skeleton(S.strict_norm((scored[name][i]["raw"].strip().splitlines() or [""])[0]))
                refs = {S.skeleton(L.words(it["meta"]["n"], it["lang"]))} | {S.skeleton(a) for a in L.alt_words(it["meta"]["n"], it["lang"])}
                if "\n" not in scored[name][i]["raw"].strip() and len(o) >= 4 and any(r.startswith(o) and len(o) >= 0.6 * len(r) for r in refs):
                    k += 1
        cut[disp] = k
    summ["d2w_possible_truncations"] = cut

    # ---------------- qualitative error examples from the best model (appendix)
    import re
    BLOCKS = [("\u0900", "\u097f", "\\devafont"), ("\u0980", "\u09ff", "\\bengfont"), ("\u0a80", "\u0aff", "\\gujrfont"),
              ("\u0b80", "\u0bff", "\\tamlfont"), ("\u0c00", "\u0c7f", "\\telufont")]
    def wrap(x):
        x = x.replace("\\", "").replace("%", "\\%").replace("&", "\\&").replace("_", "\\_").replace("#", "\\#").replace("$", "\\$").replace("{", "").replace("}", "").replace("^", "").replace("~", "")
        for lo, hi, f in BLOCKS:
            x = re.sub(f"([{lo}-{hi}](?:[{lo}-{hi}\\s]*[{lo}-{hi}])?)", lambda m: "{" + f + " " + m.group(1) + "}", x)
        return x
    want = [("w2d", "hi", "1-99"), ("w2d", "hi", "lakh"), ("w2d", "ta", "1e3-1e5"), ("w2d", "hi-Latn", "1-99"), ("d2w", "hi", "1-99"),
            ("arith", "hi", "2-digit"), ("frac", "hi", "scaled"), ("frac", "hi", "time"), ("frac", "bn", "scaled"), ("conv", "en", "lakh->million")]
    rows_ex = []
    for t, l, st in want:
        for i in sorted(items):
            it, sc = items[i], scored[bestm[0]][i]
            if (it["task"], it["lang"], it["stratum"]) == (t, l, st) and not sc["correct"] and sc["raw"].strip() and len(sc["raw"]) < 60:
                if t in ("w2d", "arith"):
                    inp = it["meta"].get("words") or f'{L.words(it["meta"]["a"], l)} {it["meta"]["op"]} {L.words(it["meta"]["b"], l)}'
                elif t == "d2w":
                    inp = str(it["meta"]["n"])
                elif t == "frac":
                    inp = it["meta"]["phrase"]
                else:
                    inp = it["prompt"].split("?")[0] + "?"
                rows_ex.append(f"{t.upper() if t in ('w2d', 'd2w') else t.capitalize()} ({SHORT[l]}) & {wrap(inp)} & {wrap(it['answer'])} & {wrap(sc['raw'].strip().splitlines()[0])} \\\\")
                break
    write("tab_examples.tex", "\n".join(["\\begin{tabular}{p{2.6cm}p{6.2cm}p{3.0cm}p{3.0cm}}", "\\toprule",
          "\\textbf{Task} & \\textbf{Input} & \\textbf{Gold} & \\textbf{Model output} \\\\", "\\midrule"] + rows_ex + ["\\bottomrule", "\\end{tabular}"]))

    (ROOT / "results" / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("per_model", "w2d_lang_mean", "w2d_mag_mean", "fused_vs_comp", "un_forms",
                                           "arith_mean", "frac_mean", "frac_by_word_mean", "conv_mean",
                                           "hi_native_vs_roman", "w2d_en_vs_enIN", "verbalizer_agreement_1_99")},
                     ensure_ascii=False, indent=1))
    print(json.dumps(summ["errors"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
