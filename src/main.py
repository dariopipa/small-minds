import argparse
import logging
import sys
from pathlib import Path

from api.app import create_app
from common.exceptions import (
    ConfigurationError,
    ModelLoadException,
    ModelNotFoundException,
)
from common.logging_config import configure_logging
from evaluation.runner import run_evaluation

logger = logging.getLogger(__name__)

app = create_app()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the configured experiments")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--benchmark",
        help="Run every configured strategy for one benchmark",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        help="Create or continue an experiment in this directory",
    )

    args = parser.parse_args()

    if args.repetitions is not None and args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    return args


def main() -> None:
    args = parse_args()
    configure_logging()

    try:
        run_evaluation(
            repetitions=args.repetitions,
            question_limit=args.limit,
            benchmark=args.benchmark,
            experiment_dir=args.experiment_dir,
        )
    except KeyboardInterrupt:
        sys.exit(1)
    except ConfigurationError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    except (ModelLoadException, ModelNotFoundException) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
