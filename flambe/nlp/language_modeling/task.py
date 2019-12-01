from typing import Dict, Optional, Any

import torch.nn as nn

from flambe.dataset import TabularDataset
from flambe.nn import Encoder, Embedder, Embeddings, MixtureOfSoftmax
from flambe.metric import Perplexity, BPC
from flambe.field import TextField
from flambe.task import Training
from flambe.optim.optimizer import Optimizer
from flambe.optim.lr_scheduler import LRScheduler


class LanguageModel(Encoder):

    def __init__(self, embedder, output_size, dropout=0, mixture_of_softmax=0) -> None:
        """A simple language model."""
        super().__init__()
        self.output_size = output_size
        self.embedder = embedder
        self.drop = nn.Dropout(dropout)

        hidden_size = embedder.output_dim
        if mixture_of_softmax > 0:
            self.output_layer = MixtureOfSoftmax(hidden_size,
                                                 output_size,
                                                 k=mixture_of_softmax,
                                                 use_activation=False)
        else:
            self.output_layer = nn.Linear(hidden_size, output_size)  # type: ignore

    @property
    def input_dim(self) -> int:
        """Get the size of the last dimension of an input."""
        return self.embedder.input_dim

    @property
    def output_dim(self) -> int:
        """Get the size of the last dimension of an output."""
        return self.output_size

    def forward(self, data, state=None):  # type: ignore
        """Run a forward pass from shape (B x S) to (B x S x H)."""
        return self.output_layer(self.drop(self.embedder(data, state=state))), state


class LanguageModeling(Training):
    """Implement an language modeling task.

    This model can be used to language modeling, as well as other
    sequential classification tasks. The full sequence predictions
    are produced by the model, effectively making the number of
    examples the batch size multiplied by the sequence length.

    """

    def __init__(self,
                 dataset: TabularDataset,
                 encoder: Optional[Encoder] = None,
                 dropout: float = 0,
                 mixture_of_softmax: int = 0,
                 tie_weights: bool = False,
                 ordered: bool = True,
                 use_bpc: bool = False,
                 build_vocabularies: bool = True,
                 train_batch_size: int = 32,
                 val_batch_size: int = 32,
                 train_unroll_size: int = 100,
                 val_unroll_size: int = 100,
                 embedding_args: Dict[str, Any] = None,
                 optimizer: Optional[Optimizer] = None,
                 iter_scheduler: Optional[LRScheduler] = None,
                 eval_scheduler: Optional[LRScheduler] = None,
                 text_field: Optional[TextField] = None,
                 embedder: Optional[Embedder] = None,
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
        mixture_of_softmax: int, optional
            Number of softmax heads. By default ``0``.
        tie_weights: bool, optional
            Tie the weights between the input and output layers.
            Weights cannot be tied when using a mixture of softmax.
            Default ``False``.
        ordered: bool, optional
            Whether the sampling should be done in the order of the
            corpus. See ``CorpusSampler`` for more details.
        use_bpc: bool, optional
            Whether to use bits per character over perplexity.
            Default ``False``.
        build_vocabularies : bool, optional
            Whether the fields should expand their vocabulary Using
            the training data. Default ``True``.
        tie_weights : bool, optional
            If true, the input and output layers share the same weights
        train_batch_size : int, optional
            The batch size to use during. Default ``32``.
        val_batch_size : int, optional
            The batch size to use during evaluation. Default ``32``.
        train_unroll_size : int, optional
            The unroll size to use during. Only used if ``oredered``
            is set. Default ``32``.
        val_unroll_size : int, optional
            The unroll size to use during evaluation. Only used if
            ``oredered`` is set. Default ``32``.
        embedding_args : Dict[str, Any], optional
            Keyword arguments to pass the ``Embeddings`` constructor.
        optimizer : Optional[Optimizer], optional
            The optimizer to use. Should be provided for training.
        iter_scheduler : Optional[LRScheduler], optional
            A learning rate scheduler to call on every training step.
        eval_scheduler : Optional[LRScheduler], optional
            A learning rate scheduler to call on every validation step.
        text_field : Optional[TextField], optional
            A custom text field to apply to the text inputs.
        embedder : Optional[Embedder], optional
            A custom embedder. Overrides ``encoder`` and ``pooling``.

        See the ``Training`` parent class for other keyword arguments.

        """
        super().__init__(**kwargs)

        text_field = text_field or TextField()

        # Build vocabularies
        if build_vocabularies:
            text, = zip(*dataset)
            text_field.setup(text)

        self.dataset = dataset
        self.embedding_args = embedding_args or dict()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_unroll_size = train_unroll_size
        self.val_unroll_size = val_unroll_size

        self.text_field = text_field

        if encoder is None and embedder is None:
            raise ValueError("At least one of encoder or embedder must be provided.")

        self.state = None
        self.encoder = encoder
        self.embedder = embedder
        self.dropout = dropout

        self.ordered = ordered
        if tie_weights and mixture_of_softmax > 0:
            raise ValueError("Cannot tie weights for a mixture of softmax.")

        self.tie_weights = tie_weights
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.metric_fn = BPC() if use_bpc else Perplexity()
        self.metric_name = 'bpc' if use_bpc else 'perplexity'

        self.optimizer = optimizer
        self.eval_scheduler = eval_scheduler
        self.iter_scheduler = iter_scheduler

    def build_model(self):
        """Build the model, containing all parameters to train."""
        input_size = output_size = self.label_field.vocab_size
        if self.embedder is None:
            embedding_dim = self.encoder.input_dim
            padding_idx = self.text_field.vocab[self.text_field.pad]
            embeddings = Embeddings(input_size, embedding_dim, padding_idx, **self.embedding_args)
            self.embedder = Embedder(embeddings, self.encoder, self.pooling, self.embedding_dropout)

        model = LanguageModel(self.embedder, output_size, self.dropout, self.mixture_of_softmax)
        if self.tie_weights:
            model.embedder.tie_weights(model.output_layer)

    def build_optimizers(self, model):
        """Build the model, containing all parameters to train."""
        if self.optimizer is None:
            raise ValueError("Using the task in training mode but not optimizer was provided.")
        self.optimizer.initialize(model)
        return {'optimizer': self.optimizer}

    def build_schedulers(self, optimizers, mode='iter'):
        """Build the model, containing all parameters to train."""
        schedulers = dict()
        if self.iter_scheduler is not None and mode == 'iter':
            self.iter_scheduler.initialize(optimizers['optimizer'])
            schedulers['iter_scheduler'] = self.iter_scheduler
        elif self.eval_scheduler is not None and mode == 'eval':
            self.eval_scheduler.initialize(optimizers['optimizer'])
            schedulers['eval_scheduler'] = self.eval_scheduler
        return schedulers

    def sample_batches(self, split='train', train=True):
        """Get an iterable of batches of data."""
        self.dataset.add_feature_hook('text', self.text_field, columns=0)
        self.dataset.add_feature_hook('label', self.label_field, columns=1)

        batch_size = self.train_batch_size if train else self.val_batch_size
        if self.ordered:
            unroll_size = self.train_unroll_size if train else self.val_unroll_size
            return CorpusSampler(self.dataset, unroll_size=unroll_size, batch_size=batch_size)
        else:
            return BaseSampler(self.dataset, shuffle=train, batch_size=batch_size)

    def batch_compute(self, batch):
        source = batch[0][:, :-1]
        targets = batch[0][:, 1:]
        preds = model(source, self.state)
        if isinstance(preds, tuple):
            preds, state = preds
            self.state = state.detach()
        mask = (target != self.pad_index).view(-1).byte()
        preds = preds.view(-1, preds.size(2))[mask]
        targets = targets.view(-1)[mask]
        return preds, targets

    def train_step(self, model, batch):
        """Compute loss on the given batch during training."""
        text, = batch
        preds, labels = self.batch_compute(model, text.to(self.device))
        loss = self.loss_fn(preds, labels)
        return loss

    def val_step(self, model, batches):
        """Compute metrics on the validation set, given in batches."""
        total_loss, metric, total_count = 0, 0, 0
        for batch in batches:
            text, = batch
            preds, labels = self.batch_compute(model, text.to(self.device))
            total_count += preds.size(0)
            total_loss += self.loss_fn(preds, labels).sum().item()
            total_metric += self.metric_fn(preds, labels).sum().item()
        loss = total_loss / total_count
        metric = total_metric / total_count
        return {'loss': loss, self.metric_name: metric}

    def val_metric(self, metrics):
        return - metrics[self.metric_name]  # lower is better
