import warnings
from abc import abstractmethod
from typing import Dict, List, Optional, Any, Iterable, TypeVar  # noqa: F401

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_

from flambe.logging import log
from flambe.learn.task import Task


Batch = TypeVar('Batch', bound=Any)


class Training(Task):
    """A PyTorch training task template.

    Implement this interface to use run training and evaluation of
    a model in PyTorch. This interface requires implementing 3
    constructors and 3 methods. The interface are designed in a way
    that the internal repesentation of input examples, and of a batch
    can be completely generics:

    - ``build_model``: constuct the nn.Module to train. The model
        is passed to the other methods to execute computation.
    - ``build_optimizers``: given the model, create the optimizers
        to use to optimize the model. Should be returned as a dict.
    - ``build_schedulers``: given the optimizers, create the learning
        rate schedulers. Should be returns as a dict.
    - ``sampler``: takes an iterable of example data points and returns
        an iterable of batches, where a batch can be any object.
    - ``train_step``: takes the model, and a batch and returns the
        loss per sample as a tensor of size B (i.e batch size)
    - ``eval_step``: takes an iterable over batches and returns the
        metric to use for validation. Higher should indicate better.

    """

    def __init__(self,
                 max_epoch: float = 1.0,
                 max_iter: Optional[int] = None,
                 gradient_accumulation: int = 1,
                 max_grad_norm: Optional[float] = None,
                 max_grad_abs_val: Optional[float] = None,
                 validation_freq: float = 0.01,
                 eval_only: bool = False,
                 eval_on_start: List[str] = ['val'],
                 eval_on_end: List[str] = ['val'],
                 eval_during_training: List[str] = ['val'],
                 num_cpus: int = -1,
                 fp16: bool = False,
                 fp16_opt_level: str = 'O1') -> None:
        """Initialize a Training task.

        Parameters
        ----------
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
        self.gradient_accumulation = gradient_accumulation
        self.max_grad_norm = max_grad_norm
        self.max_grad_abs_val = max_grad_abs_val
        self.global_iter = 0
        self.global_epoch = 0
        self.best_metric: Optional[float] = None
        self.max_epoch = max_epoch if max_iter is None else float('inf')
        self.max_iter = max_iter or float('inf')
        self.validation_freq = validation_freq
        self.eval_only = eval_only
        self.eval_on_start = eval_on_start
        self.eval_on_end = eval_on_end
        self.eval_during_training = eval_during_training

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build the model, containing all parameters to train.

        Returns
        -------
        nn.Module
            The model to train.

        """
        pass

    @abstractmethod
    def build_optimizers(self, model: nn.Module) -> Dict[str, Optimizer]:
        """Build the model, containing all parameters to train.

        Parameters
        ----------
        model: nn.Module
            The model to train

        Returns
        -------
        List[Optimizer]
            A list of optimizers to call on every batch

        """
        pass

    @abstractmethod
    def build_schedulers(self,
                         optimizers: Dict[str, Optimizer],
                         mode: str = 'iter') -> Dict[str, _LRScheduler]:
        """Build the model, containing all parameters to train.

        Parameters
        ----------
        model: nn.Module
            The model to train

        Returns
        -------
        List[Optimizer]
            A list of optimizers to call on every batch

        """
        pass

    @abstractmethod
    def sample_batches(self, split: str = 'train', train: bool = True) -> Iterable[Batch]:
        """Get an iterable of batches of data.

        This method is used to create batches of data from an
        iterable of examples. Both the input examples and the
        output batches can take any arbitary form.

        Parameters
        ----------
        train: bool
            Whether the batches are meant for training or evaluation.
            This is important as it defines the input to the
            ``batch_loss`` and ``batch_metrics`` methods.

        Returns
        -------
        Iterable[Batch]
            An iterable of batches, each of which will be passed to
            either the ``batch_loss`` or ``batch_metrics`` methods.

        """
        pass

    @abstractmethod
    def train_step(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """Compute loss on the given batch during training.

        Given a batch, this method computes a training step
        by providing the output loss. The loss should be returned
        as a tensor of size B (i.e batch size). This is to ensure
        correct computation when using gradient accumulation steps.

        ``Important``: this method should *NOT* call the backward
        method, as this is generally done in the object using the model.

        Parameters
        ----------
        model: nn.Module
            The model.
        batch: Batch
            A batch of data to train over. The batch can take any form.

        Returns
        -------
        torch.Tensor
            Should contain at least one element being the loss for the
            batch. The recommended default key is ``loss``.

        """
        pass

    @abstractmethod
    def val_step(self, model: nn.Module, batches: Iterable[Batch]) -> Dict[str, float]:
        """Compute metrics on the validation set, provided in batches.

        Returns a single float value to be used for model selection
        and early stopping. A higher number should indicate improvement.
        For metrics where lower means "better", smply return the
        negative value of the metric.

        Parameters
        ----------
        model: nn.Module
            The model.
        batch: Batch
            A batch of data to evaluate. The batch can take any form.

        Returns
        -------
        Dict[str, float]
            Set of validation metrics.

        """
        pass

    def val_metric(self, metrics: Dict[str, float]) -> float:
        """Compute metrics on the validation set, provided in batches.

        Returns a single float value to be used for model selection
        and early stopping. A higher number should indicate improvement.
        For metrics where lower means "better", smply return the
        negative value of the metric.

        Parameters
        ----------
        batch: Batch
            A batch of data to evaluate. The batch can take any form.

        Returns
        -------
                float
            The validation metric to pick the best performing model.

        """
        pass

    def setup(self):

        model = self.build_model()

        # Select right device
        num_gpus = torch.cuda.device_count()

        if num_gpus > 0 and torch.cuda.is_available():
            model.cuda()

        if not eval_only:
            optimizers = self.build_optimizers(model)

        # Enable half precision
        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Apex must be installed to use half precision.")
            if eval_only:
                model = amp.initialize(model, opt_level=fp16_opt_level)
            else:
                model, optimizers = amp.initialize(model, optimizers, opt_level=fp16_opt_level)

        # Use multi GPU
        if num_gpus > 1:
            model = torch.nn.DataParallel(model)

        self.train_sampler: Iterable = self.sample_batches(split='train', train=True)
        self.val_sampler: Iterable = self.sample_batches(split='val', train=False)
        if eval_train_set:
            self.eval_train_sampler: Iterable = self.sample_batches(split='train', train=False)

        self._train_iterator = iter(self.train_sampler)  # type: ignore

        self.model = model
        self.optimizers = optimizers
        self.iter_schedulers = self.build_schedulers(optimizers, mode='iter')
        self.eval_schedulers = self.build_schedulers(optimizers, mode='eval')

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

            loss = self.train_step(self.model, batch)
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
        for _, optimizer in self.optimizers.items():
            optimizer.step()

        for name, scheduler in self.iter_schedulers.items():
            learning_rate = scheduler.get_lr()[0]  # type: ignore
            log(f'Training/LR_{name}', learning_rate, self.global_iter)
            scheduler.step()  # type: ignore

        # Zero the gradients when exiting a train step
        for _, optimizer in self.optimizers.items():
            optimizer.zero_grad()

    def evaluate(self) -> None:
        """Run an evaluation step over the validation data."""
        self.model.eval()

        # Compute metrics
        val_metrics = self.val_step(self.model, self.val_sampler)
        _val_metric = self.val_metric(val_metrics)
        # Update best model
        if self.best_metric is None or _val_metric > self.best_metric:
            self.best_metric = _val_metric
            self.best_model = self.model.state_dict()

        # Update scheduler
        for name, scheduler in self.eval_schedulers.items():
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(_val_metric)
            else:
                # torch's _LRScheduler.step DOES have a default value
                # so passing in no args is fine; it will automatically
                # compute the current epoch
                scheduler.step()  # type: ignore
            learning_rate = scheduler.get_lr()[0]  # type: ignore
            log(f'Training/LR_{name}', learning_rate, self.global_iter)

        # Log everything
        log(f'Validation/metric', val_metric, self.global_iter)  # type: ignore
        for name, metric in val_metrics:
            log(f'Validation/{name}', metric, self.global_iter)  # type: ignore
        if self.eval_train_set:
            train_metrics = self.val_step(self.model, self.eval_train_sampler)
            train_metric = self.val_metric(train_metrics)
            log(f'TrainingEval/{metric}', train_metric, self.global_iter)  # type: ignore
            for name, metric in train_metrics:
                log(f'TrainingEval/{name}', metric, self.global_iter)  # type: ignore

        self.model.train()

    def run(self) -> bool:
        _continue = True

        # Initialize the training iterator
        if self.global_iter == 0:
            self.setup()
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

    def metric(self) -> Optional[float]:
        """Override this method to enable scheduling."""
        return self._best_metric

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
