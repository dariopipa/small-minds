from pathlib import Path

PROMPT_ROOT = Path(__file__).resolve().parent


def read_prompt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompt


def load_prompt(role: str, *context: str) -> str:
    return read_prompt(PROMPT_ROOT.joinpath(role, *context, "prompt.txt"))
