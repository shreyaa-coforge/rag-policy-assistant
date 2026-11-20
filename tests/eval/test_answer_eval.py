import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# These tests call the real OpenAI API, so they need a key (from the environment or .env)
load_dotenv()
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set"
)

EVAL_SET = json.loads((Path(__file__).parent / "eval_set.json").read_text(encoding="utf-8"))
KNOWN_FAILURES = {
    "How long do I have to submit an expense report?",
    "What is the daily meal allowance when traveling for business?",
}
XFAIL = pytest.mark.xfail(
    strict=False,
    reason="Planted data quality issue: outdated 2023 expense policy in corpus. See DIAGNOSIS.md",
)
CASES = [
    pytest.param(
        case, id=case["question"], marks=XFAIL if case["question"] in KNOWN_FAILURES else ()
    )
    for case in EVAL_SET
]

failures = []  # (question, expected, actual answer) for the summary


@pytest.fixture(scope="module", autouse=True)
def summary():
    yield
    passed = len(EVAL_SET) - len(failures)
    print(f"\nAnswer accuracy: {passed}/{len(EVAL_SET)} passed")
    for question, expected, answer in failures:
        print(f"\n- Question: {question}\n  Expected: {expected}\n  Answer:   {answer}")


@pytest.mark.parametrize("case", CASES)
def test_answer(rag_module, case):
    answer = rag_module.answer(case["question"])["answer"]

    if case["type"] == "out_of_scope":
        expected = ["I don't know"]
    else:
        expected = case["expected_keywords"]
    missing = [keyword for keyword in expected if keyword.lower() not in answer.lower()]

    if missing:
        failures.append((case["question"], expected, answer))
    assert not missing, f"Missing {missing} in answer: {answer}"
