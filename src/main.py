import logging
import sys

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


def main():
    configure_logging()

    try:
        run_evaluation()
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
