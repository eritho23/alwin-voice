from __future__ import annotations

import re

_QUESTION_PREFIXES = (
    "vad ",
    "hur ",
    "var ",
    "när ",
    "vem ",
    "vilken ",
    "vilket ",
    "vilka ",
    "varför ",
    "what ",
    "why ",
    "who ",
    "when ",
    "where ",
    "which ",
    "how ",
    "can you ",
    "could you ",
    "would you ",
    "will you ",
    "do you ",
    "is it ",
    "are you ",
    "can i ",
    "could i ",
    "kan du ",
    "kan ni ",
    "skulle du ",
    "vill du ",
    "menar du ",
    "ursäkta ",
    "ursakta ",
)

_CLARIFICATION_PHRASES = (
    "vad menar du",
    "kan du upprepa",
    "kan du förklara",
    "kan du forklara",
    "kan du säga",
    "kan du saga",
    "menar du",
    "förlåt",
    "forlat",
    "ursäkta",
    "ursakta",
)

_NOISE_WORDS = {
    "eh",
    "ehm",
    "hmm",
    "hm",
    "mm",
    "mmm",
    "um",
    "umm",
    "uh",
    "uhm",
    "öh",
    "öhm",
}

_WORD_RE = re.compile(r"[a-zåäö]+", re.IGNORECASE)


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def is_clear_question_or_clarification(text: str) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False

    if not any(char.isalpha() for char in normalized):
        return False

    words = _words(normalized)
    if not words:
        return False

    if all(word in _NOISE_WORDS for word in words):
        return False

    if len(words) == 1:
        return words[0] in {"vad", "hur", "var", "när", "vem", "why", "what"}

    if re.search(r"(.)\1{3,}", normalized):
        return False

    if normalized.endswith("?"):
        return True

    if any(normalized.startswith(prefix) for prefix in _QUESTION_PREFIXES):
        return True

    if any(phrase in normalized for phrase in _CLARIFICATION_PHRASES):
        return True

    return False
