from abc import abstractmethod
from typing import Dict, Iterable, Optional, List, Any, TypeVar

import torch

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from flambe.nn import Module


Example = TypeVar('Example', bound=Any)
Batch = TypeVar('Batch', bound=Any)


class Model(Module):
    """Base model interface.

    Implement this inerface to use a custom Model with the training
    and evaluation routines provided in the ``learn`` module. This
    interface requires implementing only 4 methods, with a few extra
    being optional. The interface are designed in a way that the
    internal representation of input examples, and of a batch can
    be completely generics:

    - ``sampler``: takes an iterable of example data points and returns
        an iterable of batches, where a batch can be anything.
    - ``batch_train``: takes a batch and returns a dictionary which
        contains the loss, and other useful training metrics.
    - ``batch_eval``: takes a batch and returns a dictionary of results.
        The values in the dictionary can be an abitrary object.
    - ``aggregate``: takes a list of outputs from ``batch_eval``, and
        aggregates results into a single output for the full dataset.

    Optionally, this interface also offers the following methods:

    - ``initialize``: run additional model related initialization,
        which may require to compute statistics over the dataset. For
        example, generating vocabularies in NLP models.
    - ``compare``: takes outputs from the ``aggregate`` method, and
        returns which of the two sets of results is best.
    - ``optimizer``: returns an initialized optimizer for this model.
        This object will be passed to the scheduler methods below.
    - ``iter_scheduler``: returns a learning rate scheduler to call
        ``step`` on after every optimization step.
    - ``eval_scheduler``: returns a learning rate scheduler to call
        ``step`` on after every evluation step.

    """

    @abstractmethod
    def sampler(self, data: Iterable[Example], train: bool = True) -> Iterable[Batch]:
        """Get an iterable of batches of data.

        This method is used to create batches of data from an
        iterable of examples. Both the input examples and the
        output batches can take any arbitary form.

        Parameters
        ----------
        data: Iterable[Example]
            An iterable of examples. The type of an ``Example`` depends
            on the input dataset provided.
        train: bool
            Whether the batches are meant for training or evaluation.
            This is important as it defines the input to the
            ``batch_train`` and ``batch_eval`` methods.

        Returns
        -------
        Iterable[Batch]
            An iterable of batches, each of which will be passed to
            either the ``batch_train`` or ``batch_eval`` methods.

        """
        pass

    @abstractmethod
    def batch_train(self, batch: Batch) -> Dict[str, Any]:
        """Compute loss on the given batch.

        Given a batch, this method computes a training step
        by providing the output loss. By default, the key used by
        typical Flambé stages is ``loss``, but that can generally be
        overriden directly in these objects. This generic interface
        allows for other stages to require extra keys that can be used
        for more complex loss computation.

        ``Important``: this method should *NOT* call the backward
        method, as this is generally done in the object using the model.

        Parameters
        ----------
        batch: Batch
            A batch of data to train over. The batch can take any form.

        Returns
        -------
        Dict[str, Any]
            Should contain at least one element being the loss for the
            batch. The recommended default key is ``loss``.

        """
        pass

    @abstractmethod
    def batch_eval(self, batch: Batch) -> Dict[str, Any]:
        """Compute validation metrics on the given batch.

        Given a batch, this method computes a set of evaluation metrics
        which can be used to compare two models at different stages of
        training.

        Parameters
        ----------
        batch: Batch
            A batch of data to evaluate. The batch can take any form.

        Returns
        -------
        Dict[str, Any]
            Dictionary contraining results. This is passed to the
            ``aggregate`` method, which computes the final metrics
            for the whole dataset.

        """
        pass

    @abstractmethod
    def aggregate(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate evaluation metrics for the full dataset.

        Recieves a list of results where each result is an output of
        the ``batch_eval`` method, each of which corresponds to a
        a batch of data. This method is used to aggregate results
        into a single result dictionary for the whole dataset.

        Parameters
        ----------
        metrics: List[Dict[str, Any]]
            List of restults as given by the ``batch_eval`` method. 

        Returns
        -------
        Dict[str, Any]

        """
        pass

    def initialize(self, dataset: Iterable[Example]):
        """Extra initialization based on the dataset.

        This method is generally necessary for most models as it will
        be used to create things like vocabularies for NLP models. It
        is however not required per say, since some models may not
        require it. Does nothing by default.

        Parameters
        ----------
        Iterable[Example]
            An iterable of examples, which can be any arbitrary objects.

        """
        pass

    def compare(self, metrics: Dict[str, Any], other: Dict[str, Any]) -> bool:
        """Compare this model's metrics to another's.

        This method recieves the output of the aggregate method, and
        if used to determine which of two models performed best so
        thhat early stopping can be done. The default implementation
        returns ``True``, meaning that the latest model is always
        picked, but you can easily override this behavior.

        Parameters
        ----------
        metrics: Dict[str, Any]
            Dictionary of metrics provided by the ``aggregate`` method,
            for the first model.
        other: Dict[str, Any]
            Dictionary of metrics provided by the ``aggregate`` method,
            for the other model to compare agaisnt.

        Returns
        -------
        bool
            ``True`` if the first model outperforms the second, and
            ``False`` otherwise.

        """
        return True

    def optimizer(self) -> Optimizer:
        """Get an initialized optimizer.

        Returns
        -------
        Optimizer
            A Pytorch optimizer to call on every iteration.

        """
        return torch.optim.Adam(self.parameters())

    def iter_scheduler(self, optimizer: Optimizer) -> Optional[_LRScheduler]:
        """The learning rate schedulers to be called on every batch.

        Parameters
        ----------
        optimizer: Optimizer
            The optimizer returned by the ``optimizer`` method.

        Returns
        -------
        Optional[_LRScheduler]
            An optional learning rate scheduler. Default is ``None``.

        """
        return None

    def eval_scheduler(self, optimizer: Optimizer) -> Optional[_LRScheduler]:
        """The learning rate schedulers to be called on every eval.

        Parameters
        ----------
        optimizer: Optimizer
            The optimizer returned by the ``optimizer`` method.

        Returns
        -------
        Optional[_LRScheduler]
            An optional learning rate scheduler. Default is ``None``.

        """
        return None
