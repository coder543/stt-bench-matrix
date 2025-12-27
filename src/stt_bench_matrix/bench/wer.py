from __future__ import annotations

import re


_TAG_RE = re.compile(r"<[^>]+>")
_BRACKET_RE = re.compile(r"\[[^\]]+\]")
_NON_WORD_RE = re.compile(r"[^a-z0-9']+")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = _TAG_RE.sub(" ", text)
    text = _BRACKET_RE.sub(" ", text)
    text = _NON_WORD_RE.sub(" ", text)
    return " ".join(text.split())


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    if not hyp_words:
        return 1.0
    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1] / float(len(ref_words))
