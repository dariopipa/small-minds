from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

_GSM8K_TARGET = re.compile(r"####\s*(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)")
_GSM8K_STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
_AGREEMENT_LABELS = {
    "all_six": "All six gave the same answer",
    "four_or_five": "Four or five gave the same answer",
    "unique_one_to_three": "One answer won with only one to three votes",
    "tied_top": "Two or more answers tied for the most votes",
    "no_usable_answer": "No usable answer was extracted",
}
_STRATEGIES = {"self_consistency", "society_of_minds", "role_based_svj"}


def _rate(numerator: int | float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _number(answer: str) -> Decimal | None:
    try:
        value = Decimal(answer.replace("$", "").replace(",", "").strip())
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def _expected_answer(sample: Any, benchmark: str) -> str | Decimal | None:
    expected = sample.expected_answer
    if benchmark == "gsm8k":
        matches = _GSM8K_TARGET.findall(str(expected))
        return _number(matches[-1]) if matches else None
    if benchmark == "arc_challenge_chat":
        answer = str(expected).strip().upper()
        return answer if answer in {"A", "B", "C", "D"} else None
    if benchmark == "boolq" and expected in {0, 1}:
        return "yes" if expected else "no"
    return None


def _correct(answer: str | None, expected: str | Decimal, benchmark: str) -> bool:
    if answer is None:
        return False
    if benchmark == "gsm8k":
        return _number(answer) == expected
    return answer.strip().casefold() == str(expected).casefold()


def _gsm8k_candidate_correct(response: str, expected: str) -> bool:
    """Apply the recorded GSM8K strict metric after final-response normalization.

    The strategy's extractor canonicalizes marked numbers before handing the
    selected raw response to lm-eval. Apply that same operation to each candidate.
    Its numeric fallback cannot produce a strict match, so no fallback is needed
    here. Preserve marker placement, whitespace and currency symbols exactly.
    """

    def normalize_marked_number(match: re.Match[str]) -> str:
        original = match.group(1)
        value = _number(original)
        assert value is not None
        normalized = format(value, "f") if value != 0 else "0"
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        if "$" in original:
            normalized = (
                f"-${normalized[1:]}"
                if normalized.startswith("-")
                else f"${normalized}"
            )
        return match.group(0)[: match.start(1) - match.start()] + normalized

    normalized_response = _GSM8K_TARGET.sub(normalize_marked_number, response)
    match = _GSM8K_STRICT.search(normalized_response)
    if match is None:
        return False
    answer = match.group(1)
    # These are the regexes_to_ignore saved in the experiment's GSM8K metric.
    for pattern in (r",", r"\$", r"(?s).*#### ", r"\.$"):
        answer = re.sub(pattern, "", answer)
        expected = re.sub(pattern, "", expected)
    return answer.casefold() == expected.casefold()


def _responses(sample: Any) -> list[dict[str, Any]] | None:
    trace = sample.strategy_result
    if not isinstance(trace, dict):
        return None
    responses = trace.get("agent_responses")
    if not isinstance(responses, list) or not responses:
        return None
    if any(
        not isinstance(response, dict)
        or not isinstance(response.get("extracted_response"), (str, type(None)))
        for response in responses
    ):
        return None
    return responses


def _answer(response: dict[str, Any]) -> str | None:
    # Saved JSON omits null optional fields, including unsuccessful extractions.
    answer = response.get("extracted_response")
    return answer if isinstance(answer, str) and answer.strip() else None


def _majority(answers: list[str | None]) -> str | None:
    """Counter preserves encounter order when the leading vote is tied."""
    votes = Counter(answer for answer in answers if answer is not None)
    return votes.most_common(1)[0][0] if votes else None


def _agreement_category(answers: list[str | None]) -> str:
    votes = Counter(answer for answer in answers if answer is not None)
    if not votes:
        return "no_usable_answer"
    highest = max(votes.values())
    if highest == 6:
        return "all_six"
    if highest >= 4:
        return "four_or_five"
    if list(votes.values()).count(highest) > 1:
        return "tied_top"
    return "unique_one_to_three"


def _rounds(
    responses: list[dict[str, Any]],
) -> dict[int, list[str | None]] | None:
    rounds: dict[int, list[dict[str, Any]]] = {}
    for response in responses:
        round_id = response.get("round_id")
        if type(round_id) is not int or round_id not in {1, 2}:
            return None
        rounds.setdefault(round_id, []).append(response)
    if 1 not in rounds:
        return None
    for agents in rounds.values():
        if any(type(agent.get("agent_id")) is not int for agent in agents):
            return None
        if len(agents) != 3 or {agent.get("agent_id") for agent in agents} != {
            1,
            2,
            3,
        }:
            return None
    return {
        round_id: [_answer(agent) for agent in agents]
        for round_id, agents in rounds.items()
    }


def _self_consistency(
    sample: Any,
    benchmark: str,
    expected: str | Decimal,
    responses: list[dict[str, Any]],
    counts: Counter[str],
    agreements: dict[str, Counter[str]],
) -> str | None:
    if len(responses) not in {6, 7}:
        return "expected six initial candidates and at most one tie-break candidate"
    if benchmark == "gsm8k" and any(
        not isinstance(response.get("response"), str) for response in responses[:6]
    ):
        return "missing raw candidate response needed for GSM8K strict scoring"
    answers = [_answer(response) for response in responses[:6]]
    if benchmark == "gsm8k":
        counts["individual_correct"] += sum(
            _gsm8k_candidate_correct(response["response"], sample.expected_answer)
            for response in responses[:6]
        )
    else:
        counts["individual_correct"] += sum(
            _correct(answer, expected, benchmark) for answer in answers
        )
    counts["final_correct"] += int(sample.metric)
    category = agreements[_agreement_category(answers)]
    category["observations"] += 1
    category["correct"] += int(sample.metric)
    return None


def _society_of_minds(
    sample: Any,
    benchmark: str,
    expected: str | Decimal,
    responses: list[dict[str, Any]],
    counts: Counter[str],
) -> str | None:
    rounds = _rounds(responses)
    if rounds is None:
        return "expected three distinct agents in initial round and revision round"
    if 2 not in rounds:
        return None
    trace = sample.strategy_result
    if "initial_extracted_response" in trace and not isinstance(
        trace["initial_extracted_response"], (str, type(None))
    ):
        return "malformed stored initial extraction"
    counts["revised"] += 1
    final = rounds[2]
    counts["unanimous_after_revision"] += (
        all(answer is not None for answer in final) and len(set(final)) == 1
    )
    # Older logs omit the initial majority; reconstruct it exactly as the
    # strategy did, including its first-encountered answer rule for tied votes.
    initial = (
        trace["initial_extracted_response"]
        if "initial_extracted_response" in trace
        else _majority(rounds[1])
    )
    before_correct = _correct(initial, expected, benchmark)
    after_correct = sample.metric == 1
    counts["before_correct"] += before_correct
    counts["after_correct"] += int(sample.metric)
    counts["corrected"] += not before_correct and after_correct
    counts["damaged"] += before_correct and not after_correct
    return None


def _svj(
    sample: Any,
    benchmark: str,
    expected: str | Decimal,
    responses: list[dict[str, Any]],
    counts: Counter[str],
) -> str | None:
    if any(not isinstance(response.get("agent_role"), str) for response in responses):
        return "missing or malformed Solver, Verifier or Judge role"
    roles = {response.get("agent_role"): response for response in responses}
    if len(responses) != 3 or set(roles) != {"solver", "verifier", "judge"}:
        return "expected one Solver, Verifier and Judge response"
    if not isinstance(
        sample.strategy_result.get("extracted_response"), (str, type(None))
    ):
        return "malformed saved final extraction"
    solver = _answer(roles["solver"])
    verifier = _answer(roles["verifier"])
    final = _answer(sample.strategy_result)
    solver_correct = _correct(solver, expected, benchmark)
    final_correct = sample.metric == 1
    if final_correct and solver is None:
        counts["missing_to_correct"] += 1
    elif final_correct and not solver_correct:
        counts["wrong_to_correct"] += 1
    elif solver_correct and not final_correct:
        counts["correct_to_wrong"] += 1
    if solver is not None and verifier is not None and solver != verifier:
        counts["disagreements"] += 1
        if final == solver:
            counts["matched_solver"] += 1
        elif final == verifier:
            counts["matched_verifier"] += 1
        else:
            counts["matched_neither"] += 1
    return None


def calculate_strategy_behavior(groups: Iterable[Any]) -> dict[str, Any]:
    """Return JSON-safe Tables 4.7–4.12 and explicit trace coverage.

    ``groups`` use the ``analysis.Group`` interface. Each table pools question ×
    repetition observations. Self-Consistency agreement additionally pools all
    supplied models and benchmarks. Incomplete or unsupported trace structures
    are excluded with a warning, never classified as missing extracted answers.
    """
    data: dict[str, Any] = {
        "self_consistency_accuracy": [],
        "self_consistency_agreement": [],
        "society_of_minds_revision": [],
        "society_of_minds_accuracy": [],
        "svj_transitions": [],
        "svj_disagreement": [],
        "trace_coverage": [],
        "warnings": [],
    }
    agreements: dict[str, Counter[str]] = {
        category: Counter() for category in _AGREEMENT_LABELS
    }
    has_self_consistency = False
    for group in groups:
        if group.strategy not in _STRATEGIES:
            continue
        counts: Counter[str] = Counter()
        excluded: Counter[str] = Counter()
        total = 0
        for run in group.runs:
            for sample in run.samples:
                total += 1
                responses = _responses(sample)
                expected = _expected_answer(sample, group.benchmark)
                if responses is None:
                    excluded["missing or malformed intermediate trace"] += 1
                    continue
                if expected is None:
                    excluded["unsupported or malformed expected answer"] += 1
                    continue
                if sample.metric not in {0, 1}:
                    excluded["final benchmark metric must be binary"] += 1
                    continue
                if group.strategy == "self_consistency":
                    reason = _self_consistency(
                        sample, group.benchmark, expected, responses, counts, agreements
                    )
                elif group.strategy == "society_of_minds":
                    reason = _society_of_minds(
                        sample, group.benchmark, expected, responses, counts
                    )
                else:
                    reason = _svj(sample, group.benchmark, expected, responses, counts)
                if reason:
                    excluded[reason] += 1
                else:
                    counts["observations"] += 1
        identity = {"model": group.model, "benchmark": group.benchmark}
        observations = counts["observations"]
        data["trace_coverage"].append(
            {
                **identity,
                "strategy": group.strategy,
                "total_observations": total,
                "analyzed_observations": observations,
                "excluded_observations": sum(excluded.values()),
                "exclusion_reasons": dict(excluded),
            }
        )
        for reason, count in excluded.items():
            data["warnings"].append(
                f"{group.model} / {group.benchmark} / {group.strategy}: "
                f"excluded {count} of {total} observations ({reason})."
            )
        if group.strategy == "self_consistency":
            has_self_consistency = True
            individual_accuracy = _rate(counts["individual_correct"], observations * 6)
            final_accuracy = _rate(counts["final_correct"], observations)
            data["self_consistency_accuracy"].append(
                {
                    **identity,
                    "observations": observations,
                    "individual_responses": observations * 6,
                    "individual_accuracy": individual_accuracy,
                    "final_accuracy": final_accuracy,
                    "gain": final_accuracy - individual_accuracy
                    if final_accuracy is not None and individual_accuracy is not None
                    else None,
                }
            )
        elif group.strategy == "society_of_minds":
            revised = counts["revised"]
            data["society_of_minds_revision"].append(
                {
                    **identity,
                    "observations": observations,
                    "revised": revised,
                    "revision_rate": _rate(revised, observations),
                    "unanimous_after_revision": counts["unanimous_after_revision"],
                    "agreement_rate": _rate(
                        counts["unanimous_after_revision"], revised
                    ),
                }
            )
            data["society_of_minds_accuracy"].append(
                {
                    **identity,
                    "revised": revised,
                    "before_accuracy": _rate(counts["before_correct"], revised),
                    "after_accuracy": _rate(counts["after_correct"], revised),
                    "gain": _rate(
                        counts["after_correct"] - counts["before_correct"], revised
                    ),
                    "corrected": counts["corrected"],
                    "damaged": counts["damaged"],
                }
            )
        else:
            data["svj_transitions"].append(
                {
                    **identity,
                    "observations": observations,
                    "wrong_to_correct": counts["wrong_to_correct"],
                    "missing_to_correct": counts["missing_to_correct"],
                    "correct_to_wrong": counts["correct_to_wrong"],
                }
            )
            data["svj_disagreement"].append(
                {
                    **identity,
                    "observations": observations,
                    "disagreements": counts["disagreements"],
                    "matched_solver": counts["matched_solver"],
                    "matched_verifier": counts["matched_verifier"],
                    "matched_neither": counts["matched_neither"],
                }
            )
    if has_self_consistency:
        for category, label in _AGREEMENT_LABELS.items():
            counts = agreements[category]
            data["self_consistency_agreement"].append(
                {
                    "category": category,
                    "label": label,
                    "observations": counts["observations"],
                    "correct": int(counts["correct"]),
                    "final_accuracy": _rate(counts["correct"], counts["observations"]),
                }
            )
    return data
