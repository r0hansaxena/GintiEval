"""Static appendix tables (lexicons, prompts) generated from the same code that builds the data."""
import json
from pathlib import Path
import lexicon as L

GEN = Path(__file__).resolve().parent.parent / "paper" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
FONT = {"hi": "\\devafont", "mr": "\\devafont", "bn": "\\bengfont", "gu": "\\gujrfont", "ta": "\\tamlfont", "te": "\\telufont"}
s = lambda lang, t: f"{{{FONT[lang]} {t}}}"

# Hindi 1-99 with the romanization used for hi-Latn
rows = []
for i in range(1, 100, 3):
    cells = []
    for n in (i, i + 1, i + 2):
        cells.append(f"{n} & {s('hi', L.words(n, 'hi'))} & \\textit{{{L.roman_hindi(n)}}}")
    rows.append(" & ".join(cells) + " \\\\")
(GEN / "tab_roman.tex").write_text("\n".join(
    ["\\begin{tabular}{rll@{\\hskip 14pt}rll@{\\hskip 14pt}rll}", "\\toprule"] + rows + ["\\bottomrule", "\\end{tabular}"]), encoding="utf-8")

# the same numbers across languages (illustration of fused vs compositional systems)
ex = [9, 19, 20, 29, 30, 39, 40]
t = ["\\begin{tabular}{r" + "l" * 3 + "}", "\\toprule",
     " & \\textbf{Hindi} & \\textbf{Tamil} & \\textbf{Telugu} \\\\", "\\midrule"]
for n in ex:
    t.append(f"{n} & {s('hi', L.words(n, 'hi'))} & {s('ta', L.words(n, 'ta'))} & {s('te', L.words(n, 'te'))} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
(GEN / "tab_systems.tex").write_text("\n".join(t), encoding="utf-8")

# fraction lexicon
FL = L.FRACTION_LEX
font = {"hi": "\\devafont", "hi-Latn": "\\itshape", "bn": "\\bengfont", "mr": "\\devafont"}
t = ["\\begin{tabular}{lllll}", "\\toprule",
     "\\textbf{Meaning} & \\textbf{hi} & \\textbf{hi-Latn} & \\textbf{bn} & \\textbf{mr} \\\\", "\\midrule"]
for key, lab in [("one_q", "1.25"), ("one_h", "1.5"), ("two_h", "2.5"), ("plus_q", "$N+0.25$"),
                 ("minus_q", "$N-0.25$"), ("plus_h", "$N+0.5$")]:
    t.append(lab + " & " + " & ".join(f"{{{font[l]} {FL[l][key]}}}" + (" $N$" if key.startswith(("plus", "minus")) else "")
                                      for l in ("hi", "hi-Latn", "bn", "mr")) + " \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
(GEN / "tab_fraclex.tex").write_text("\n".join(t), encoding="utf-8")

# one example prompt per task, verbatim from the data file
items = [json.loads(l) for l in open(GEN.parent.parent / "data" / "gintieval.jsonl", encoding="utf-8")]
pick = {"w2d": ("hi", "lakh"), "d2w": ("hi", "lakh"), "arith": ("hi", "2-digit"), "frac": ("hi", "scaled"), "conv": ("en", "lakh->million")}
out = []
esc = lambda x: x.replace("\\", "\\textbackslash ").replace("%", "\\%").replace("&", "\\&").replace("_", "\\_").replace("#", "\\#")
import re
def deva(x):
    return re.sub(r"([\u0900-\u097f][\u0900-\u097f\s]*[\u0900-\u097f]|[\u0900-\u097f])", lambda m: "{\\devafont " + m.group(1) + "}", x)
NAMES = {"w2d": "T1 Words$\\to$Digits", "d2w": "T2 Digits$\\to$Words", "arith": "T3 Arithmetic", "frac": "T4 Fraction words", "conv": "T5 System conversion"}
for task, (lang, st) in pick.items():
    it = next(i for i in items if i["task"] == task and i["lang"] == lang and i["stratum"] == st)
    p = deva(esc(it["prompt"])).replace("\n", "\\newline ")
    out.append(f"\\textbf{{{NAMES[task]}}} & {p} & {deva(esc(it['answer']))} \\\\")
(GEN / "tab_prompts.tex").write_text("\n".join(
    ["\\begin{tabular}{p{2.3cm}p{9.6cm}p{2.6cm}}", "\\toprule", "\\textbf{Task} & \\textbf{Prompt (verbatim)} & \\textbf{Gold answer} \\\\", "\\midrule"]
    + ["\\midrule\n".join(out)] + ["\\bottomrule", "\\end{tabular}"]), encoding="utf-8")

# Figure 1: three items taken verbatim from the data file
def first(task, lang, stratum, cond=lambda i: True):
    return next(i for i in items if i["task"] == task and i["lang"] == lang and i["stratum"] == stratum and cond(i))
a = first("w2d", "hi", "lakh")
b = first("frac", "hi", "scaled", lambda i: i["meta"]["word"] == "minus_q" and i["meta"]["mult"] == 10**5)
c = first("conv", "en", "lakh->million")
box = ["\\begin{tcolorbox}[colback=gray!6,colframe=black!55,boxrule=0.5pt,left=4pt,right=4pt,top=3pt,bottom=3pt,fontupper=\\small]",
       "\\textbf{Words$\\to$Digits (Hindi)}\\\\ Write this number in digits: " + deva(a["meta"]["words"]) + "\\\\ \\textbf{Answer:} " + a["answer"],
       "\\tcblower\\small",
       "\\textbf{Fraction Words (Hindi)}\\\\ What number does the phrase ``" + deva(b["meta"]["phrase"]) + "'' denote?\\\\ \\textbf{Answer:} " + b["answer"],
       "\\medskip\\hrule\\medskip",
       "\\textbf{System Conversion}\\\\ " + c["prompt"].split("?")[0] + "?\\\\ \\textbf{Answer:} " + c["answer"],
       "\\end{tcolorbox}"]
(GEN / "fig_example.tex").write_text("\n".join(box), encoding="utf-8")
print("ok")
