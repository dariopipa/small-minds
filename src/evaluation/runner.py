import logging
from typing import Any

from configs.context import load_experiment_context, load_llm_eval_config
from evaluation.lm_eval_harness import LLMEvalHarness

logger = logging.getLogger(__name__)


def run_evaluation() -> dict[str, Any]:
    context = load_experiment_context()
    eval_config = load_llm_eval_config(context)
    logger.info(
        "Evaluation started: experiment=%s backend=%s tasks=%s fewshot=%d limit=%s",
        context.selected.name,
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
