from typing import Optional

import torch.nn as nn

from flambe.dataset import TabularDataset
from flambe.nn import Encoder
from flambe.metric import Accuracy
from flambe.field import LabelField
from flambe.task import Training
from flambe.optim.optimizer import Optimizer
from flambe.optim.lr_scheduler import LRScheduler
from flambe.nlp.classification import TextClassification


class ImageClassifier(Encoder):
    """Implements a simple image classifier."""

    def __init__(self, encoder: Encoder, output_size: int, dropout: float = 0) -> None:
        """A simple image classifier."""
        super().__init__()
        self.output_size = output_size
        self.encoder = encoder
        self.drop = nn.Dropout(dropout)
        self.output_layer = nn.Linear(encoder.output_dim, output_size)

    @property
    def input_dim(self) -> int:
        """Get the size of the last dimension of an input."""
        return self.encoder.input_dim

    @property
    def output_dim(self) -> int:
        """Get the size of the last dimension of an output."""
        return self.output_size

    def forward(self, data):  # type: ignore
        """Run a forward pass from shape (B x H) to (B x O)."""
        return self.output_layer(self.drop(self.encoder(data).flatten(1)))


class ImageClassification(TextClassification):
    """An image classification task.

    Performs image classifcation training and evaluation.
    Takes as input a dataset and an encoder, and constructs
    a simple ImageClassifier. You may pass your custom fields
    or used the defaults. The loss is computer via a cross entropy,
    and the validation metric is accuracy.

    """

    def __init__(self,
                 dataset: TabularDataset,
                 encoder: Optional[Encoder] = None,
                 dropout: float = 0,
                 build_vocabularies: bool = True,
                 train_batch_size: int = 32,
                 val_batch_size: int = 32,
                 optimizer: Optional[Optimizer] = None,
                 iter_scheduler: Optional[LRScheduler] = None,
                 eval_scheduler: Optional[LRScheduler] = None,
                 label_field: Optional[LabelField] = None,
                 model: Optional[ImageClassifier] = None,
                 **kwargs) -> None:
        """Initalize a TextClassification task.

        Parameters
        ----------
        dataset : TabularDataset
            The input dataset
        encoder : Encoder
            An encoder
        dropout : float, optional
            Dropout to apply between the encoder and output layer.
            Default ``0``.
        build_vocabularies : bool, optional
            Whether the fields should expand their vocabulary Using
            the training data. Default ``True``.
        train_batch_size : int, optional
            The batch size to use during. Default ``32``.
        val_batch_size : int, optional
            The batch size to use during evaluation. Default ``32``.
        optimizer : Optional[Optimizer], optional
            The optimizer to use. Should be provided for training.
        iter_scheduler : Optional[LRScheduler], optional
            A learning rate scheduler to call on every training step.
        eval_scheduler : Optional[LRScheduler], optional
            A learning rate scheduler to call on every validation step.
        label_field : Optional[LabelField], optional
            A custom label field to apply to the label inputs.
        model : Optional[ImageClassifier], optional
            A custom model. Overrides ``encoder`.

        See the ``Training`` parent class for other keyword arguments.

        """
        Training.__init__(self, **kwargs)

        label_field = label_field or LabelField()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.dataset = dataset
        self.label_field = label_field
        transforms = {'image': {'field': label_field, 'column': 1}}
        self.dataset._set_transforms(transforms, do_setup=build_vocabularies)  # type: ignore

        self.encoder = encoder
        self._model = model  # type: ignore
        self.dropout = dropout
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.metric_fn = Accuracy()

        self.optimizer = optimizer
        self.eval_scheduler = eval_scheduler
        self.iter_scheduler = iter_scheduler

    def build_model(self):
        """Build the model, containing all parameters to train."""
        if self._model is not None:
            return self._model
        output_size = self.label_field.vocab_size
        return ImageClassifier(self.encoder, output_size, self.dropout)
