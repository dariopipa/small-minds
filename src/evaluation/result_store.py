import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from configs.experiments import Experiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def prepare_experiment_directory(
    output_root: str,
    started_at: datetime,
    requested_dir: Path | None,
) -> tuple[Path, str]:
    experiment_id = f"{started_at:%Y-%m-%d_%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    output_dir = requested_dir or Path(output_root) / experiment_id
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "experiment.json"
    if metadata_path.exists():
        experiment_id = json.loads(metadata_path.read_text(encoding="utf-8"))[
            "experiment_id"
        ]
    else:
        write_json(
            metadata_path,
            {"experiment_id": experiment_id, "created_at": started_at.isoformat()},
        )
    return output_dir, experiment_id


def completed_run_exists(
    output_dir: Path,
    experiment: Experiment,
    repetition: int,
) -> bool:
    run_dir = (
        output_dir
        / experiment.benchmark_name
        / experiment.strategy_name
        / f"run-{repetition:03d}"
    )
    run_path = run_dir / "run.json"
    return (
        run_path.is_file()
        and (run_dir / "samples.jsonl").is_file()
        and json.loads(run_path.read_text(encoding="utf-8")).get("status")
        == "completed"
    )


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write("\n")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for value in values:
            output.write(json.dumps(value, ensure_ascii=False) + "\n")
