"""Number-word lexicons and verbalizers used to build GintiEval.

Primary reference verbalizer: AI4Bharat `indic-numtowords` (cardinals).
Secondary references (used only to widen the set of accepted spellings when
scoring digits->words): `indic-num2words` (module `num_to_words`) and
`num2words` where the language is supported.
"""
from __future__ import annotations

import unicodedata as U
from fractions import Fraction

from indic_numtowords import num2words as _ai4b
from num_to_words import num_to_word as _inw2
from num2words import num2words as _n2w

# code -> (English name, script name, typology of the 1-99 system)
LANGS = {
    "hi": ("Hindi", "Devanagari", "fused"),
    "bn": ("Bengali", "Bengali", "fused"),
    "mr": ("Marathi", "Devanagari", "fused"),
    "gu": ("Gujarati", "Gujarati", "fused"),
    "ta": ("Tamil", "Tamil", "compositional"),
    "te": ("Telugu", "Telugu", "compositional"),
}

# Evident typos / foreign-orthography artefacts in the primary verbalizer.
_PATCHES = {
    "ta": [("ஓன்", "ஒன்")],        # long-O typo in 81 and 89 (cf. 71-79, 91)
    "mr": [("हज़ार", "हजार")],      # Marathi does not use the nukta here
    "bn": [("অষ্টআশি", "আটাশি")],  # non-standard spelling of 88; both secondary references give আটাশি
}


def nfc(s: str) -> str:
    return U.normalize("NFC", " ".join(s.split()))


def words(n: int, lang: str) -> str:
    """Primary reference verbalization of a non-negative integer."""
    if lang == "hi-Latn":
        return roman_hindi(n)
    if lang == "en-IN":
        return _clean_en(_n2w(n, lang="en_IN"))
    if lang == "en":
        return _clean_en(_n2w(n, lang="en"))
    s = nfc(_ai4b(n, lang=lang))
    for a, b in _PATCHES.get(lang, []):
        s = s.replace(nfc(a), nfc(b))
    return nfc(s)


def _clean_en(s: str) -> str:
    return " ".join(s.replace(",", " ").split())


def alt_words(n: int, lang: str) -> list[str]:
    """Verbalizations from the secondary references (may be empty)."""
    out = []
    try:
        out.append(nfc(_inw2(n, lang=lang).replace(",", " ")))
    except Exception:
        pass
    try:
        out.append(nfc(_n2w(n, lang=lang).replace(",", " ")))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Romanized Hindi (informal "Hinglish" spellings, hand-curated by the authors)
# ---------------------------------------------------------------------------
_RH = """shunya ek do teen chaar paanch chhah saat aath nau
das gyarah barah terah chaudah pandrah solah satrah atharah unnees
bees ikkees baees teis chaubees pachchees chhabbees sattaees atthaees untees
tees ikattees battees taintees chauntees paintees chhattees saintees adtees untaalees
chaalees iktaalees bayaalees taintaalees chauvaalees paintaalees chhiyaalees saintaalees adtaalees unchaas
pachaas ikyaavan baavan tirepan chauvan pachpan chhappan sattaavan atthaavan unsath
saath iksath baasath tirsath chaunsath painsath chhiyaasath sadsath adsath unhattar
sattar ikhattar bahattar tihattar chauhattar pachhattar chhihattar sathattar athhattar unyaasi
assi ikyaasi bayaasi tiraasi chauraasi pachaasi chhiyaasi sattaasi atthaasi navaasi
nabbe ikyaanave baanave tiraanave chauraanave pachaanave chhiyaanave sattaanave atthaanave ninyaanave""".split()
assert len(_RH) == 100 and len(set(_RH)) == 100


def roman_hindi(n: int) -> str:
    if n < 100:
        return _RH[n]
    parts = []
    for size, name in ((10**7, "crore"), (10**5, "lakh"), (10**3, "hazaar"), (100, "sau")):
        q, n = divmod(n, size)
        if q:
            parts.append(f"{_RH[q]} {name}")
    if n:
        parts.append(_RH[n])
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Vernacular fraction words ("sawa / dedh / dhai / paune / saadhe" family)
# ---------------------------------------------------------------------------
Q, H = Fraction(1, 4), Fraction(1, 2)

FRACTION_LEX = {
    # plus_q N = N + 1/4 ; minus_q N = N - 1/4 ; plus_h N = N + 1/2
    # one_q = 1.25, one_h = 1.5, two_h = 2.5 are stand-alone words.
    "hi": dict(one_q="सवा", one_h="डेढ़", two_h="ढाई", plus_q="सवा", minus_q="पौने", plus_h="साढ़े",
               mult={100: "सौ", 10**3: "हज़ार", 10**5: "लाख", 10**7: "करोड़"}, oclock="बजे"),
    "hi-Latn": dict(one_q="sawa", one_h="dedh", two_h="dhai", plus_q="sawa", minus_q="paune", plus_h="saadhe",
                    mult={100: "sau", 10**3: "hazaar", 10**5: "lakh", 10**7: "crore"}, oclock="baje"),
    # For Bengali and Marathi we avoid the fused hundred forms and keep the
    # multipliers whose fraction constructions are uncontroversial.
    "bn": dict(one_q="সোয়া এক", one_h="দেড়", two_h="আড়াই", plus_q="সোয়া", minus_q="পৌনে", plus_h="সাড়ে",
               mult={10**3: "হাজার", 10**5: "লাখ", 10**7: "কোটি"}, oclock=None),
    "mr": dict(one_q="सव्वा", one_h="दीड", two_h="अडीच", plus_q="सव्वा", minus_q="पावणे", plus_h="साडे",
               mult={10**3: "हजार", 10**5: "लाख", 10**7: "कोटी"}, oclock=None),
}

FRACTION_LANG_NAME = {"hi": "Hindi", "hi-Latn": "Hindi written in the Latin script (romanized Hindi)",
                      "bn": "Bengali", "mr": "Marathi"}


def fraction_items(lang: str):
    """Yield (phrase, value: Fraction, kind, meta) for one language."""
    L = FRACTION_LEX[lang]
    base = "hi" if lang == "hi" else lang
    w = (lambda k: words(k, base)) if lang != "hi-Latn" else roman_hindi
    heads = [("one_q", L["one_q"], Fraction(5, 4), None),
             ("one_h", L["one_h"], Fraction(3, 2), None),
             ("two_h", L["two_h"], Fraction(5, 2), None)]
    for k in range(2, 13):
        heads.append(("plus_q", f'{L["plus_q"]} {w(k)}', k + Q, k))
        heads.append(("minus_q", f'{L["minus_q"]} {w(k)}', k - Q, k))
        if k >= 3:
            heads.append(("plus_h", f'{L["plus_h"]} {w(k)}', k + H, k))
    for kind, head, val, k in heads:
        yield head, val, "bare", dict(word=kind, n=k, mult=1)
        for m, mw in L["mult"].items():
            yield f"{head} {mw}", val * m, "scaled", dict(word=kind, n=k, mult=m)
    if L["oclock"]:
        oc = L["oclock"]
        yield f'{L["one_h"]} {oc}', "1:30", "time", dict(word="one_h", n=None, mult=None)
        yield f'{L["two_h"]} {oc}', "2:30", "time", dict(word="two_h", n=None, mult=None)
        for k in range(1, 13):
            yield f'{L["plus_q"]} {w(k)} {oc}', f"{k}:15", "time", dict(word="plus_q", n=k, mult=None)
            prev = 12 if k == 1 else k - 1
            yield f'{L["minus_q"]} {w(k)} {oc}', f"{prev}:45", "time", dict(word="minus_q", n=k, mult=None)
            if k >= 3:
                yield f'{L["plus_h"]} {w(k)} {oc}', f"{k}:30", "time", dict(word="plus_h", n=k, mult=None)


def indian_group(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])
