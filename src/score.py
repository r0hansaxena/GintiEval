"""Scoring for GintiEval: answer extraction, D2W matching and error taxonomy."""
from __future__ import annotations

import re
import unicodedata as U
from fractions import Fraction
from functools import lru_cache

import lexicon as L

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def ascii_digits(s: str) -> str:
    """Map any Unicode decimal digit (e.g. Devanagari, Bengali) to ASCII."""
    return "".join(str(U.digit(ch)) if ch.isdigit() and not ch.isascii() and U.category(ch) == "Nd" else ch
                   for ch in s)


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def extract_number(out: str):
    """Last number in the output, after removing digit-grouping commas."""
    s = ascii_digits(out)
    s = re.sub(r"(?<=\d)[,   ](?=\d)", "", s) if re.search(r"\d,\d", s) else s
    m = _NUM.findall(s)
    if not m:
        return None
    try:
        return Fraction(m[-1])
    except Exception:
        return None


def extract_time(out: str):
    m = re.findall(r"(\d{1,2})\s*[:.]\s*(\d{2})", ascii_digits(out))
    return f"{int(m[-1][0])}:{m[-1][1]}" if m else None


def extract_grouped(out: str):
    m = re.findall(r"\d[\d,]*\d|\d", ascii_digits(out))
    return m[-1] if m else None


# ---------------------------------------------------------------------------
# Digits -> words matching
# ---------------------------------------------------------------------------
_DROP = {
    "़", "়", "઼",                 # nukta
    "ँ", "ं", "ঁ", "ং", "ઁ", "ં", "ఁ", "ం",  # candrabindu / anusvara
    "ः", "ঃ", "ઃ",                 # visarga
    "्", "্", "્",                 # virama (Indo-Aryan scripts only)
    "‌", "‍",
}
_MERGE = str.maketrans({
    "ी": "ि", "ू": "ु", "ई": "इ", "ऊ": "उ",   # Devanagari long->short i/u
    "ী": "ি", "ূ": "ু", "ঈ": "ই", "ঊ": "উ",   # Bengali
    "ી": "િ", "ૂ": "ુ", "ઈ": "ઇ", "ઊ": "ઉ",   # Gujarati
    "য": "য়",                                                              # Bengali ya / yya
})


def strict_norm(s: str) -> str:
    s = U.normalize("NFC", s)
    s = re.sub(r"[\s,.\-।*\"'`]+", " ", s).strip()
    return s


def skeleton(s: str) -> str:
    """Orthography-insensitive key: drops nasalisation marks, nukta, virama,
    vowel-length contrasts on i/u, doubled letters, spaces and punctuation."""
    s = U.normalize("NFD", s).translate(_MERGE)
    s = "".join(ch for ch in s if ch not in _DROP and (U.category(ch)[0] in "LM"))
    return re.sub(r"(.)\1+", r"\1", s)


# Extra hand-added variants of very common words (all standard spellings).
_EXTRA = {"hi": {"छः": ["छह", "छे"], "हज़ार": ["हजार"], "एक सौ": ["सौ"]},
          "mr": {}, "bn": {"আটাশি": ["অষ্টাশি", "অষ্টআশি"]}, "gu": {}, "ta": {}, "te": {}}


@lru_cache(maxsize=None)
def _variant_table(lang: str):
    """primary token -> set of skeletons seen for the same value in any reference."""
    tab: dict[str, set[str]] = {}
    probe = list(range(0, 100)) + [100 * k for k in range(1, 10)] + [10**3, 10**5, 10**7]
    for n in probe:
        prim = L.words(n, lang)
        alts = L.alt_words(n, lang)
        ptoks = prim.split()
        for alt in alts:
            atoks = alt.split()
            if len(atoks) == len(ptoks):
                for p, a in zip(ptoks, atoks):
                    tab.setdefault(p, set()).add(skeleton(a))
            tab.setdefault(prim, set()).add(skeleton(alt))
        for p in ptoks:
            tab.setdefault(p, set()).add(skeleton(p))
        tab.setdefault(prim, set()).add(skeleton(prim))
    for k, vs in _EXTRA.get(lang, {}).items():
        tab.setdefault(k, set()).update(skeleton(v) for v in vs)
    return tab


def lenient_match(out: str, n: int, lang: str) -> bool:
    tab = _variant_table(lang)
    target = skeleton(strict_norm(out))
    if not target:
        return False
    whole = {skeleton(L.words(n, lang))} | {skeleton(a) for a in L.alt_words(n, lang)}
    if target in whole:
        return True
    toks = L.words(n, lang).split()

    @lru_cache(maxsize=None)
    def rec(i: int, pos: int) -> bool:
        if i == len(toks):
            return pos == len(target)
        opts = set(tab.get(toks[i], {skeleton(toks[i])}))
        if i + 1 < len(toks):                       # two-token units such as "एक सौ"
            for v in tab.get(toks[i] + " " + toks[i + 1], ()):
                if target.startswith(v, pos) and rec(i + 2, pos + len(v)):
                    return True
        return any(target.startswith(v, pos) and rec(i + 1, pos + len(v)) for v in opts if v)

    return rec(0, 0)


def char_sim(a: str, b: str) -> float:
    a, b = strict_norm(a), strict_norm(b)
    if not a and not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1 - prev[-1] / max(len(a), len(b))


# ---------------------------------------------------------------------------
# Item-level scoring
# ---------------------------------------------------------------------------


def score(item: dict, out: str) -> dict:
    t = item["answer_type"]
    out = (out or "").strip()
    if t == "int" or t == "decimal":
        pred = extract_number(out)
        ok = pred is not None and pred == Fraction(item["answer"])
        return dict(correct=ok, pred=None if pred is None else str(float(pred)) if pred.denominator != 1 else str(pred.numerator))
    if t == "time":
        pred = extract_time(out)
        return dict(correct=pred == item["answer"], pred=pred)
    if t == "grouped":
        pred = extract_grouped(out)
        return dict(correct=pred == item["answer"], pred=pred)
    if t == "words":
        n, lang = item["meta"]["n"], item["lang"]
        first = out.splitlines()[0] if out else ""
        strict = strict_norm(first) == strict_norm(item["answer"])
        len_ok = strict or lenient_match(first, n, lang)
        return dict(correct=len_ok, strict=strict, sim=char_sim(first, item["answer"]), pred=first)
    raise ValueError(t)


# ---------------------------------------------------------------------------
# Error taxonomy (only for wrong answers with a parsable prediction)
# ---------------------------------------------------------------------------


def _groups(n: int):
    """Indian-system groups: (crore, lakh, thousand, hundred, rest)."""
    c, n = divmod(n, 10**7)
    l, n = divmod(n, 10**5)
    t, n = divmod(n, 10**3)
    h, r = divmod(n, 100)
    return c, l, t, h, r


def w2d_error(gold: int, pred) -> str:
    if pred is None:
        return "no-number"
    if pred.denominator != 1 or pred < 0:
        return "other"
    p = pred.numerator
    c, l, t, h, r = _groups(gold)
    # Western re-scaling: lakh read as million and/or crore read as billion / 100 million
    for lv in (10**5, 10**6):
        for cv in (10**7, 10**8, 10**9):
            if (lv, cv) != (10**5, 10**7) and p == c * cv + l * lv + t * 1000 + h * 100 + r:
                return "scale-substitution"
    # concatenation of the spoken groups without zero padding
    cat = "".join(str(x) for x in (c, l, t) if x) + (f"{h}{r:02d}" if h else (str(r) if r else ""))
    if cat and p == int(cat) and p != gold:
        return "dropped-zero-padding"
    g, s = str(gold), str(p)
    if len(g) != len(s):
        if s.strip("0") == g.strip("0"):
            return "magnitude"
        return "length"
    gp, pp = _groups(gold), _groups(p)
    diff = [i for i in range(5) if gp[i] != pp[i]]
    if len(diff) == 1:
        a, b = gp[diff[0]], pp[diff[0]]
        if a < 100 and b < 100:
            if a % 10 == b % 10 and abs(a // 10 - b // 10) == 1:
                return "decade-shift"
            if a // 10 == b // 10:
                return "unit-digit"
        return "one-group"
    return "multi-group"


def frac_error(item: dict, pred) -> str:
    m = item["meta"]
    if pred is None:
        return "no-number"
    word, n, mult = m["word"], m["n"], m["mult"]
    if item["answer_type"] == "time":
        return "time-other"
    base = {"one_q": 1, "one_h": 1, "two_h": 2}.get(word, n)
    cands = {}
    cands["fraction-ignored"] = {Fraction(base) * mult, Fraction(base + 1) * mult}
    if word == "minus_q":
        cands["wrong-direction"] = {(n + Fraction(1, 4)) * mult, (n + Fraction(3, 4)) * mult, (n + Fraction(1, 2)) * mult}
    else:
        cands["wrong-direction"] = {(base - Fraction(1, 4)) * mult, (base - Fraction(1, 2)) * mult}
    true = Fraction(item["answer"])
    frac = true / mult - int(true / mult)
    if mult > 1:
        cands["fraction-on-unit"] = {base * mult + frac, base * mult + frac * 100, base * mult + 1, base * mult + frac * 10}
    others = {Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)} - {frac}
    cands["wrong-fraction"] = {(base + o) * mult for o in others} | {(base - 1 + o) * mult for o in others}
    for k, vs in cands.items():
        if pred in vs:
            return k
    if mult > 1 and true != 0:
        ratio = pred / true
        if ratio in {Fraction(10), Fraction(1, 10), Fraction(100), Fraction(1, 100), Fraction(1000), Fraction(1, 1000)}:
            return "magnitude"
    return "other"


def conv_error(item: dict, pred) -> str:
    m = item["meta"]
    if item["answer_type"] == "grouped":
        return "grouping"
    if pred is None:
        return "no-number"
    true, v = Fraction(item["answer"]), Fraction(m["v"])
    if pred == v:
        return "identity (units treated as equal)"
    if true != 0 and pred * true == v * v:
        return "inverse factor"
    r = pred / true
    k = 0
    while r.denominator == 1 and r.numerator % 10 == 0 and r != 0:
        r /= 10
        k += 1
    while r.numerator == 1 and r.denominator % 10 == 0:
        r *= 10
        k -= 1
    if r == 1 and k != 0:
        return "power-of-ten slip"
    return "other"
