import json
from pathlib import Path
from typing import Any

from configs.experiments import Experiment


def read_calls(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_samples(
    experiment: Experiment,
    raw_samples: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    calls_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for call in calls:
        calls_by_prompt.setdefault(call["prompt"], []).append(call)

    samples: dict[str, dict[str, Any]] = {}
    for raw in raw_samples:
        sample_id = str(raw["doc_id"])
        if sample_id not in samples:
            prompt = str(_first(raw["arguments"]))
            samples[sample_id] = {
                "record_type": "sample",
                "sample_id": raw["doc_id"],
                "question_id": f"{experiment.benchmark_name}:{raw['doc_id']}",
                "document": raw["doc"],
                "prompt": prompt,
                "expected_answer": raw["target"],
                "lm_eval_response": _first(raw["resps"]),
                "hashes": {
                    "document": raw["doc_hash"],
                    "prompt": raw["prompt_hash"],
                    "target": raw["target_hash"],
                },
                "evaluations": {},
            }
        samples[sample_id]["evaluations"][raw.get("filter", "none")] = {
            "response": raw["filtered_resps"],
            "metrics": {metric: raw[metric] for metric in raw["metrics"]},
        }

    records = []
    for sample in samples.values():
        matching_calls = calls_by_prompt.pop(sample["prompt"], [])
        records.append(
            {
                **sample,
                "status": (
                    "completed"
                    if any("result" in call for call in matching_calls)
                    else "failed"
                ),
                "calls": matching_calls,
            }
        )

    if calls_by_prompt:
        records.append(
            {
                "record_type": "unscored_calls",
                "status": "unscored",
                "calls": [call for group in calls_by_prompt.values() for call in group],
            }
        )
    return records


def _first(value: Any) -> Any:
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    return value
