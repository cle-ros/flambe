from abc import abstractmethod
from typing import Dict, Iterable, List, Any, TypeVar

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

    - ``build``: run additional model related initialization,
        which may require to compute statistics over the dataset. For
        example, generating vocabularies in NLP models.
    - ``sampler``: takes an iterable of example data points and returns
        an iterable of batches, where a batch can be anything.
    - ``batch_train``: takes a batch and returns a dictionary which
        contains the loss, and other useful training metrics.
    - ``batch_eval``: takes a batch and returns a dictionary of results.
        The values in the dictionary can be an abitrary object.
    - ``aggregate``: takes a list of outputs from ``batch_eval``, and
        aggregates results into a single output for the full dataset.
     - ``val_metric``: takes outputs from the ``aggregate`` method, and
        returns the value of the metric to use for model selection.

    """

    @classmethod
    @abstractmethod
    def build(cls, dataset: Iterable[Example], **kwargs) -> None:
        """Build a new model based on a dataset.

        This method is generally necessary for most models as it will
        be used to create things like vocabularies for NLP models,
        which may affect the model architecture (i.e input size).

        Parameters
        ----------
        Iterable[Example]
            An iterable of examples, which can be any arbitrary objects.

        """
        pass

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
        by providing the output loss. Should contain a key named
        ``loss``. This generic interface allows for other stages to
        require extra keys that can be used for more complex loss
        computation.

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
        global_step: int
            Global step to use for logging

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
        into a single result dictionary for the whole dataset. This
        method is also used to log different metrics during evaluation.

        Parameters
        ----------
        metrics: List[Dict[str, Any]]
            List of restults as given by the ``batch_eval`` method.

        Returns
        -------
        Dict[str, Any]

        """
        pass

    @abstractmethod
    def val_metric(self, metrics: Dict[str, Any]) -> float:
        """Select the metrc to be used for model selection.

        This method recieves the output of the aggregate method, and
        returns a single float value to be used for model selection
        and early stopping. A higher number should indicate improvement.
        For metrics where lower means "better", smply return the
        negative value of the metric.

        Parameters
        ----------
        metrics: Dict[str, Any]
            Dictionary of metrics provided by the ``aggregate`` method.

        Returns
        -------
        float
            The validation metric to pick the best performing model.

        """
        pass
