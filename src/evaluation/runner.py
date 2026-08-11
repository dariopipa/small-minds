import logging
from typing import Any

from configs.context import build_llm_eval_config, load_experiment
from evaluation.lm_eval_harness import LLMEvalHarness

logger = logging.getLogger(__name__)


def run_evaluation() -> dict[str, Any]:
    provider, config, experiment = load_experiment()
    eval_config = build_llm_eval_config(
        provider,
        experiment,
        question_limit=config.run.questions,
    )
    logger.info(
        "Evaluation started: experiment=%s backend=%s tasks=%s fewshot=%d limit=%s",
        experiment.name,
        eval_config.backend,
        ",".join(eval_config.tasks),
        eval_config.num_fewshot,
        eval_config.limit,
    )
    eval_harness = LLMEvalHarness(config=eval_config)
    results = eval_harness.evaluate()

    logger.info(
        "Evaluation finished: tasks=%s",
        ",".join(results.get("results", {})),
    )

    return results
