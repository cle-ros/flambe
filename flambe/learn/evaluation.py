from typing import Iterable, Dict  # noqa: F401

import torch

from flambe.compile import Component
from flambe.dataset import Dataset
from flambe.model import Model
from flambe.logging import log


class Evaluation(Component):
    """Implement an Evaluator block.

    An `Evaluator` takes as input data, and a model and executes
    the evaluation. This is a single step `Component` object.

    """

    def __init__(self,
                 dataset: Dataset,
                 model: Model,
                 eval_train: bool = False,
                 eval_val: bool = False,
                 eval_test: bool = True) -> None:
        """Initialize the evaluator.

        Parameters
        ----------
        dataset : Dataset
            The dataset to run evaluation on
        model : Module
            The model to train
        metric_fn: Metric
            The metric to use for evaluation
        eval_sampler : Optional[Sampler]
            The sampler to use over validation examples. By default
            it will use `BaseSampler` with batch size 16 and without
            shuffling.
        eval_data: str
            The data split to evaluate on: one of train, val or test
        device: str, optional
            The device to use in the computation.

        """
        # Select right device
        if torch.cuda.is_available():
            model.cuda()

        samplers: Dict[str, Iterable] = {}
        if eval_train:
            samplers['Train'] = model.sampler(dataset.train, train=False)
        if eval_val:
            samplers['Validation'] = model.sampler(dataset.val, train=False)
        if eval_test:
            samplers['Test'] = model.sampler(dataset.test, train=False)

        self.samplers = samplers
        self.model = model
        self.dataset = dataset

    def run(self) -> bool:
        """Run the evaluation.

        Returns
        ------
        bool
            Whether the component should continue running.

        """
        self.model.eval()

        for name, sampler in self.samplers.items():
            # Compute metrics
            metrics = map(self.model.batch_eval, sampler)
            aggregate_metrics = self.model.aggregate(metrics)
            # Log everything
            for metric, value in aggregate_metrics.items():
                log(f'{name}/{metric}', value, 0)  # type: ignore

        _continue = False
        return _continue
