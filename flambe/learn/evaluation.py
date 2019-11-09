import torch

from flambe.dataset import Dataset
from flambe.model import Model
from flambe.logging import log


class Evaluation(Stage):
    """Implement an Evaluator block.

    An `Evaluator` takes as input data, and a model and executes
    the evaluation. This is a single step `Component` object.

    """

    def __init__(self,
                 dataset: Dataset,
                 model: Model,
                 eval_train: bool = False,
                 eval_val: bool = False,
                 eval_test: bool = True,
                 num_cpus: int = 1,
                 num_gpus: int = 1) -> None:
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
        if num_gpus > 0 and torch.cuda.is_available():
            model.cuda()

        samplers = {}
        if eval_train:
            samplers['Train'] = model.sampler(dataset.train, training=False)
        if eval_val:
            samplers['Validation'] = model.sampler(dataset.val, training=False)
        if eval_test:
            samplers['Test'] = model.sampler(dataset.test, training=False)

        self.samplers = samplers
        self.model = model
        self.dataset = dataset

    def run(self):
        """Run the evaluation.

        Returns
        ------
        bool
            Whether the component should continue running.

        """
        self.model.eval()

        for name, sampler in self.samplers.items():
            # Compute metrics
            metrics = map(self.model.batch_metric, sampler)
            metrics = self.model.aggregate_metrics(metrics)
            # Log everything
            for metric, value in metrics.items():
                log(f'{name}/{metric}', value, 0)  # type: ignore
