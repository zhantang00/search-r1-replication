
from __future__ import annotations

import re
import string

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def extract_answer(response_text: str) -> str | None:
    """Return the content of the last complete <answer>...</answer> block, or None."""
    matches = _ANSWER_RE.findall(response_text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str | None, golden_answers: list[str]) -> float:
    if prediction is None:
        return 0.0
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(gold) for gold in golden_answers))


def score_trajectory(traj, fallback_to_text: bool = False) -> float:
    """Fill traj.prediction and traj.reward in place and return the reward.

    `fallback_to_text` is only used by the direct-inference baseline: if the
    model ignored the answer tags, its whole (stripped) response is scored.
    """
    prediction = extract_answer(traj.response_text)
    if prediction is None and fallback_to_text:
        prediction = traj.response_text.strip()
    traj.prediction = prediction
    traj.reward = exact_match(prediction, traj.golden_answers)
    return traj.reward


def _self_test() -> None:
    # Normalization
    assert normalize_answer("The Beatles!") == "beatles"
    assert normalize_answer("  An   apple ") == "apple"
    assert normalize_answer("McComb, Mississippi") == "mccomb mississippi"
    # Extraction: last complete block wins; whitespace stripped
    assert extract_answer("<think>x</think><answer> Paris </answer>") == "Paris"
    assert extract_answer("<answer>a</answer> ... <answer>b</answer>") == "b"
    assert extract_answer("no tags here") is None
    assert extract_answer("<answer>unfinished") is None
    assert extract_answer("<answer>multi\nline</answer>") == "multi\nline"
    # EM against several golds (Bamboogle "Answer 2" case)
    golds = ["Mildred Warner", "Mildred Gale"]
    assert exact_match("mildred gale", golds) == 1.0
    assert exact_match("Mildred", golds) == 0.0
    assert exact_match(None, golds) == 0.0
    assert exact_match("the 1999", ["1999"]) == 1.0
    print("scoring.py self-test passed")


if __name__ == "__main__":
    _self_test()
