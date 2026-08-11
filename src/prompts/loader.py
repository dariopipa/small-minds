from pathlib import Path

from common.exceptions import ConfigurationError

PROMPT_ROOT = Path(__file__).resolve().parent


def read_prompt(path: Path) -> str:
    if not path.is_file():
        raise ConfigurationError(f"Prompt file does not exist: {path}")

    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Could not read prompt file {path}: {exc}") from exc

    if not prompt:
        raise ConfigurationError(f"Prompt file is empty: {path}")

    return prompt


def load_prompt(*parts: str) -> str:
    return read_prompt(PROMPT_ROOT.joinpath(*parts).with_suffix(".txt"))
