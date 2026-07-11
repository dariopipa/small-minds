from typing import Any

from lm_eval.evaluator import simple_evaluate  # type: ignore[import-untyped]

from evaluation.lm_eval_config import LLMEvalHarnessConfig
from evaluation.evaluator_interface import EvaluatorI


class LLMEvalHarness(EvaluatorI):
    def __init__(self, config: LLMEvalHarnessConfig):
        self.config = config

    def evaluate(self) -> dict[str, Any]:
        return simple_evaluate(
            model=self.config.backend,
            model_args=self.config.model_args.model_dump(exclude_none=True),
            system_instruction=self.config.system_instruction,
            tasks=self.config.tasks,
            num_fewshot=self.config.num_fewshot,
            batch_size=self.config.batch_size,
            limit=self.config.limit,
            log_samples=self.config.log_samples,
            write_out=self.config.write_out,
            bootstrap_iters=self.config.bootstrap_iters,
            # Arguments for model generation.
            # gen_kwargs=self.config.gen_kwargs.model_dump(exclude_none=True),
        )
