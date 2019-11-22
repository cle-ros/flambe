from typing import Tuple, Dict, Iterable, Callable

import torch.nn as nn
from torch import Tensor

from flambe.dataset import TabularDataset
from flambe.sampler import BaseSampler
from flambe.model import Model
from flambe.nn import Embedder
from flambe.metric import Accuracy


Batch = Tuple[Tensor, Tensor]


class TextClassifier(Model):
    """Implements a standard classifier.

    The classifier is composed of an encoder module, followed by
    a fully connected output layer, with a dropout layer in between.

    Attributes
    ----------
    embedder: Embedder
        The embedder layer
    output_layer : Module
        The output layer, yields a probability distribution over targets
    drop: nn.Dropout
        the dropout layer

    """

    def __init__(self,
                 embedder: Embedder,
                 ouput_size: int,
                 dropout: float = 0,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 lower: bool = False,
                 multilabel: bool = False,
                 multilabel_sep: str = ',',
                 tokenizer: Optional[Tokenizer] = None,
                 text_field: Optional[TextField] = None,
                 label_field: Optional[LabelField] = None) -> None:
        """Initialize the TextClassifier model.

        Parameters
        ----------
        embedder: Embedder
            The embedder layer
        output_layer : Module
            The output layer, yields a probability distribution
        dropout : float, optional
            Amount of dropout to include between layers (defaults to 0)

        """
        super().__init__()

        self.embedder = embedder
        self.drop = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embedder.hidden_size, ouput_size)

        self.text_field = text_field or TextField(tokenizer, lower=lower)
        self.label_field = label_field or LabelField(multilabel)

        self.loss = nn.CrossEntropyLoss()
        self.metric = Accuracy()

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

    def build(cls, dataset, **kwargs):
        """Build the model from a dataset."""
        obj = cls(**kwargs)
        # Build vocabularies
        text, label = zip(*dataset)
        obj.text_field.setup(text)
        obj.label_field.setup(label)
        return obj

    def forward(self, data: Tensor) -> Tensor:
        """Run a forward pass through the network."""
        encoding = self.embedder(data)
        preds = self.output_layer(self.drop(encoding))
        return preds

    def sampler(self,
                dataset: TabularDataset[str, str],
                train: bool = True) -> Iterable[Batch]:
        """Sample batches of data for training or validation."""
        dataset.set_transform(self.text_field, column=0)
        dataset.set_transform(self.label_field, column=1)

        batch_size = self.train_batch_size if train else self.eval_batch_size
        return BaseSampler(dataset, suffle=train, batch_size=batch_size)

    def batch_train(self, batch: Batch) -> Dict[str, Tensor]:
        """Compute loss on the given batch."""
        text, labels = batch
        preds = self.forward(text.to(self.device))
        loss = self.loss(preds, labels.to(self.device))
        return {'loss': loss, 'preds': preds}

    def batch_eval(self, batch: Batch) -> Dict[str, Tensor]:
        """Compute validation metrics on the given batch."""
        text, labels = batch
        preds = self.forward(source.to(self.device))
        loss = self.loss(preds, target.to(self.device))
        accuracy = self.accuracy(preds, target)
        return {'accuracy': accuracy.item(), 'loss': loss.item()}

    def compare(self, metrics: Dict[str, float], other: Dict[str, float]) -> bool:
        """Compare this model's metrics to another's."""
        return metrics['Accuracy'] > other['Accuracy']
