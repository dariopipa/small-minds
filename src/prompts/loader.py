from pathlib import Path

PROMPT_ROOT = Path(__file__).resolve().parent


def load_prompt(path: str | Path) -> str:
    path = Path(path)

    if not path.is_absolute():
        path = PROMPT_ROOT / path

    prompt = path.with_suffix(".txt").read_text(encoding="utf-8").strip()

    if not prompt:
        raise ValueError(f"Prompt is empty: {path}")

    return prompt
