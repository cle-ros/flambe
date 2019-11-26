import warnings
from typing import Dict, List, Optional, Any, Iterable  # noqa: F401

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_

from flambe.compile import Component
from flambe.dataset import Dataset

from flambe.model import Model
from flambe.logging import log
from flambe.optim.optimizer import Optimizer
from flambe.optim.scheduler import LRScheduler


class Training(Component):
    """Implement a Pytorch Training stage.

    The `Training` stage takes as input a dataset and a model and
    executes training, thhrough its ``run`` and ``step`` methods.
    See the ``Dataset``, and ``Model`` interfaces to understand how
    to use this object.

    """

    def __init__(self,
                 dataset: Dataset,
                 model: Model,
                 optimizer: Optimizer,
                 iter_scheduler: Optional[LRScheduler] = None,
                 eval_scheduler: Optional[LRScheduler] = None,
                 eval_freq: float = 0.01,
                 max_epoch: float = 1.0,
                 max_iter: Optional[int] = None,
                 gradient_accumulation: int = 1,
                 max_grad_norm: Optional[float] = None,
                 max_grad_abs_val: Optional[float] = None,
                 num_cpus: int = -1,
                 fp16: bool = False,
                 fp16_opt_level: str = 'O1',
                 eval_first: bool = True,
                 eval_train_set: bool = False) -> None:
        """Initialize a Training stage.

        Parameters
        ----------
        dataset : Dataset
            The dataset to use in training the model
        model : Model
            The model to train
        optimizer: Optimizer
            The optimizer to use
        iter_scheduler: LRScheduler
            A scheduler to apply on every batch
        eval_scheduler: LRScheduler
            A scheduler to apply on every evaluation step
        eval_freq: float, optional
            How often to run validation
        max_epoch : int, optional
            The maximum of passes over the data
        max_iter: int, optional
            The maximum number of iterations to run during training.
            If provided, overrides max_epoch.
        gradient_accumulation: int, optional
            Number of batches to pass through the model before
            calling optimizer.step. Requires the sampler to have
            drop_last set to True. (default set to 1 so optimizer.step
            is called after every batch)
        max_grad_norm : float, optional
            Maximum Euclidean norm of gradient after clipping.
        max_grad_abs_val: float, optional
            Maximum absolute value of all gradient vector components
            after clipping.
        fp16: bool, optional
            Whether to run training with half precision. Requires Apex
            to be installed. Default ``False``.
        fp16_opt_level: str, optional
            Apex AMP optimization level: one of ['O0', 'O1', 'O2', 'O3']
        eval_first: bool, optional
            Wether to evaluate the model before training starts.
            Default ``True.
        eval_train_set: bool, optional
            Whether to also run evaluation on the training data.
            Default ``False.``

        """
        # Select right device
        num_gpus = torch.cuda.device_count()

        if num_gpus > 0 and torch.cuda.is_available():
            model.cuda()

        # Enable half precision
        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Apex must be installed to use half precision.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

        # Use multi GPU
        if num_gpus > 1:
            model = torch.nn.DataParallel(model)

        self.dataset = dataset
        self.gradient_accumulation = gradient_accumulation
        self.max_grad_norm = max_grad_norm
        self.max_grad_abs_val = max_grad_abs_val

        self._train_iterator = None
        self.train_sampler: Iterable = model.sampler(dataset.train, train=True)
        self.val_sampler: Iterable = model.sampler(dataset.val, train=False)
        if eval_train_set:
            self.eval_train_sampler: Iterable = model.sampler(dataset.train, train=False)

        self.model = model
        self.optimizer = optimizer
        if isinstance(optimizer, Optimizer) and not optimizer.initialized:
            optimizer.initialize(filter(lambda p: p.requires_grad, model.parameters()))

        self.iter_scheduler = iter_scheduler
        if isinstance(iter_scheduler, LRScheduler) and not iter_scheduler.initialized:
            iter_scheduler.initialize(optimizer)

        self.eval_scheduler = eval_scheduler
        if isinstance(eval_scheduler, LRScheduler) and not eval_scheduler.initialized:
            eval_scheduler.initialize(optimizer)

        self.global_iter = 0
        self.global_epoch = 0
        self.best_metric: Optional[float] = None
        self.max_epoch = max_epoch if max_iter is None else float('inf')
        self.max_iter = max_iter or float('inf')
        self.eval_freq = eval_freq
        self.eval_first = eval_first
        self.eval_train_set = eval_train_set

    def train(self):
        """Run a training step over the training data."""
        # Compute the loss
        accumulated_loss = 0.0
        accumulated_count = 0.0
        for _ in range(self.gradient_accumulation):
            try:
                batch = next(self._train_iterator)
            except StopIteration:
                # End of an epoch, create a new iterator
                self.global_epoch += 1
                self._train_iterator = iter(self.train_sampler)

            loss = self.model.batch_train(batch)
            accumulated_count += int(loss.size(0))
            accumulated_loss += loss.sum().item()

            if self.num_gpus > 1 or loss.size(0) > 1:
                loss = loss.mean()
            if self.gradient_accumulation > 1:
                loss = loss / self.gradient_accumulation
                if accumulated_count == 1:
                    warnings.warn("Loss is a single scalar and \
                    you are using gradient accumulation. This can result in wrong computation \
                    of the accumulated loss. If your batch size is 1, you can ignore this warning.")

            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:  # type: ignore
                    scaled_loss.backward()
            else:
                loss.backward()

        # Clip gradients if necessary
        if self.fp16:
            parameters = amp.master_params(self.optimizer)  # type: ignore
        else:
            parameters = self.model.parameters()
        if self.max_grad_norm:
            clip_grad_norm_(parameters, self.max_grad_norm)
        if self.max_grad_abs_val:
            clip_grad_value_(parameters, self.max_grad_abs_val)

        # Log everything
        self.global_iter += 1

        log(f'Training/Loss', accumulated_loss / accumulated_count, self.global_iter)
        log(f'Training/Gradient_Norm', self.model.gradient_norm, self.global_iter)
        log(f'Training/Parameter_Norm', self.model.parameter_norm, self.global_iter)

        # Optimize
        self.optimizer.step()
        if self.iter_scheduler is not None:
            learning_rate = self.iter_scheduler.get_lr()[0]  # type: ignore
            log(f'Training/LR', learning_rate, self.global_iter)
            self.iter_scheduler.step()  # type: ignore

        # Zero the gradients when exiting a train step
        self.optimizer.zero_grad()

    def evaluate(self) -> None:
        """Run an evaluation step over the validation data."""
        self.model.eval()

        # Compute metrics
        metrics = map(self.model.batch_eval, self.val_sampler)
        aggregate_metrics = self.model.aggregate_metrics(metrics)

        # Update best model
        val_metric = self.model.val_metric(aggregate_metrics)
        if self.best_metric is None or val_metric > self.best_metric:
            self.best_metric = val_metric
            self.best_model = self.model.state_dict()

        # Update scheduler
        if self.eval_scheduler is not None:
            if isinstance(self.eval_scheduler, ReduceLROnPlateau):
                self.eval_scheduler.step(val_metric)
            else:
                # torch's _LRScheduler.step DOES have a default value
                # so passing in no args is fine; it will automatically
                # compute the current epoch
                self.scheduler.step()  # type: ignore

        # Log everything
        for metric, value in aggregate_metrics.items():
            log(f'Validation/{metric}', value, self.global_iter)  # type: ignore

        if self.eval_train_set:
            # Compute metrics
            metrics = map(self.model.batch_metric, self.eval_train_sampler)
            aggregate_metrics = self.model.aggregate_metrics(metrics)
            for metric, value in aggregate_metrics.items():
                log(f'TrainingEval/{metric}', value, self.global_iter)  # type: ignore

        self.model.train()

    def run(self) -> bool:
        _continue = True

        # Initialize the training iterator
        if self.global_iter == 0:
            self._train_iterator = iter(self.train_sampler)  # type: ignore
        # Evaluate on start
        if self.global_iter == 0 and self.eval_first:
            with torch.no_grad():
                self.evaluate()
        # Train and evaluate during training
        if self.global_iter < self.max_iter and self.global_epoch < self.max_epoch:
            self.train()
            if self.global_iter % int(self.max_iter * self.eval_freq):
                with torch.no_grad():
                    self.evaluate()
        # Load best model
        if self.global_iter == self.max_iter or self.global_epoch == self.max_epoch:
            _continue = False
            self.model.eval()
            self.model.load_state_dict(self.best_model)

        return _continue

    def _state(self,
               state_dict,
               prefix: str,
               local_metadata: Dict[str, Any]):
        state_dict[prefix + 'global_iter'] = self.global_iter
        state_dict[prefix + 'global_epoch'] = self.global_epoch
        state_dict[prefix + 'optimizer'] = self.optimizer.state_dict()
        if self.iter_scheduler is not None:
            state_dict[prefix + 'iter_scheduler'] = self.iter_scheduler.state_dict()
        if self.eval_scheduler is not None:
            state_dict[prefix + 'eval_scheduler'] = self.eval_scheduler.state_dict()
        return state_dict

    def _load_state(self,
                    state_dict,
                    prefix: str,
                    local_metadata: Dict[str, Any],
                    strict: bool,
                    missing_keys: List[Any],
                    unexpected_keys: List[Any],
                    error_msgs: List[Any]) -> None:
        self.global_iter = state_dict[prefix + 'global_iter']
        self.global_epoch = state_dict[prefix + 'global_epoch']
        self.optimizer.load_state_dict(state_dict[prefix + 'optimizer'])
        if self.iter_scheduler is not None:
            iter_scheduler_state = state_dict[prefix + 'iter_scheduler']
            self.iter_scheduler.load_state_dict(iter_scheduler_state)  # type: ignore
        if self.eval_scheduler is not None:
            eval_scheduler_state = state_dict[prefix + 'eval_scheduler']
            self.eval_scheduler.load_state_dict(eval_scheduler_state)  # type: ignore
        # Useful when loading the model after training
        done = self.global_iter >= self.max_iter
        if done:
            self.model.load_state_dict(self.best_model)
