"""Tokenisation for the inverted index.

Deliberately simple and language-aware rather than clever. Two rules drive the design:

1. **No English stemmer or stoplist on Danish text.** EB-NeRD is Danish; applying English
   linguistic resources to it would silently degrade the index and is the single easiest
   way to produce numbers that look plausible and are wrong.
2. **No stemming at all, for now.** Stemming is a hyper-parameter with a real trade-off
   (recall up, precision down) and it belongs in an ablation with a measured number next to
   it, not as an unexamined default baked into the tokeniser.
"""

from __future__ import annotations

import re

# Split on anything that is not a letter or digit. \w with re.UNICODE keeps Danish æ, ø, å
# and accented characters as word characters, which a naive [a-z0-9]+ would destroy.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Deliberately short lists: only words so frequent that they carry no discriminative signal.
# BM25's IDF already down-weights common terms, so an aggressive stoplist mostly removes
# terms the scoring would have handled anyway, at the cost of losing phrases like "The Who".
ENGLISH_STOPWORDS = frozenset("""
a an and are as at be but by for from has have he in is it its of on or she that the
their they this to was were will with you your i we us our
""".split())

DANISH_STOPWORDS = frozenset("""
af alle andet at blev blive bliver da de dem den denne der deres det dette dig din
disse dog du efter eller en end er et for fra ham han hans har havde have hende hendes
her hos hun hvad hvis hvor i ikke ind jeg jer kunne man mange med meget men mig min
mine mit ned noget nogle nu når og også om op os over på selv sig sin sine sit skal
skulle som sådan der til ud under var ved vi vil ville vor være været
""".split())

STOPWORDS = {"en": ENGLISH_STOPWORDS, "da": DANISH_STOPWORDS, None: frozenset()}


def tokenize(text: str | None, lang: str | None = "en", min_length: int = 2) -> list[str]:
    """Lowercase, split on non-word characters, drop stopwords and very short tokens.

    `min_length=2` drops single characters, which in news text are almost always list
    markers or fragments of a split contraction rather than content.

    A null or empty text yields an empty list, never None — callers concatenate the result
    and should not have to guard for it.
    """
    if not text:
        return []
    stop = STOPWORDS.get(lang, frozenset())
    return [
        tok for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) >= min_length and tok not in stop
    ]


def tokenize_fields(*fields: str | None, lang: str | None = "en") -> list[str]:
    """Tokenise several fields into one bag, e.g. title + abstract.

    Concatenating rather than weighting is the honest default: field weighting is a real
    alternative (see SPEC.md) and it should be introduced as a measured ablation, not
    smuggled in as a constant here.
    """
    out: list[str] = []
    for f in fields:
        out.extend(tokenize(f, lang=lang))
    return out
