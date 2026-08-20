from typing import Any

from lm_eval.evaluator import simple_evaluate  # type: ignore[import-untyped]

from evaluation.base import Evaluator
from evaluation.lm_eval_config import LLMEvalHarnessConfig


class LLMEvalHarness(Evaluator):
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
            log_samples=True,
            write_out=self.config.write_out,
            bootstrap_iters=self.config.bootstrap_iters,
            cache_requests=self.config.cache_requests,
        )
