from typing import Tuple, Optional, Union

import torch.nn as nn
from torch import Tensor


from flambe.nn import Embedder, Module
from flambe.metric import Perplexity, BPC


class LanguageModel(Module):
    """Implement an LanguageModel model for sequential classification.

    This model can be used to language modeling, as well as other
    sequential classification tasks. The full sequence predictions
    are produced by the model, effectively making the number of
    examples the batch size multiplied by the sequence length.

    """

    def __init__(self,
                 embedder: Embedder,
                 output_layer: Module,
                 dropout: float = 0,
                 pad_index: int = 0,
                 tie_weights: bool = False,
                 tie_weight_attr: str = 'embedding',
                 ordered: bool = True,
                 use_bpc: bool = False) -> None:
        """Initialize the LanguageModel model.

        Parameters
        ----------
        embedder: Embedder
            The embedder layer
        output_layer : Decoder
            Output layer to use
        dropout : float, optional
            Amount of droput between the encoder and decoder,
            defaults to 0.
        pad_index: int, optional
            Index used for padding, defaults to 0
        tie_weights : bool, optional
            If true, the input and output layers share the same weights
        tie_weight_attr: str, optional
            The attribute to call on the embedder to get the weight
            to tie. Only used if tie_weights is ``True``. Defaults
            to ``embedding``. Multiple attributes can also be called
            by adding another dot: ``embeddings.word_embedding``.

        """
        super().__init__()

        self.embedder = embedder
        self.output_layer = output_layer
        self.drop = nn.Dropout(dropout)

        self.pad_index = pad_index
        self.tie_weights = tie_weights

        if tie_weights:
            module = self.embedder
            for attr in tie_weight_attr.split('.'):
                module = getattr(module, attr)
            self.output_layer.weight = module.weight

        self.use_bpc = use_bpc
        self.loss = nn.CrossEntropyLoss()
        self.metric = BPC() if use_bpc else Perplexity()

    def forward(self, data: Tensor) -> Tensor:
        """Run a forward pass through the network."""
        encoding, _ = self.embedder(data)
        pred = self.output_layer(self.drop(encoding))
        return pred

    def batch_train(self, batch: Batch) -> Dict[str, Any]:
        """Compute loss on the given batch."""
        flat_mask = mask.view(-1).byte()
            flat_encodings = encoding.view(-1, encoding.size(2))[flat_mask]
            flat_targets = target.view(-1)[flat_mask]
            flat_pred = self.output_layer(self.drop(flat_encodings))
            return flat_pred, flat_targets
        source, target = batch
        predictions = self.forward(source)
        return {'loss' : self.loss(predictions, target)}

    def batch_eval(self, batch: Batch) -> Dict[str, Any]:
        """Compute validation metrics on the given batch."""
        source, target = batch
        preds = self.forward(source)

        loss = self.loss(preds, target)
        metric = self.metric(preds, target)
        return {f'{self.metric}': metric.item(), 'loss': loss.item()}

    def sampler(self, dataset: Dataset, training: bool = True) -> Iterable[Batch]:
        """Sample batches of data for training or validation."""
        batch_size = self.train_batch_size if training else self.eval_batch_size
        if self.ordered:
            return CorpusSampler(dataset, batch_size=batch_size)
        else:
            return BaseSampler(dataset, shuffle=training, batch_size=batch_size)

    def compare(self, metrics: Dict[str, float], other: Dict[str, float]) -> bool:
        """Compare this model's metrics to another's."""
        metric_name = f'{self.metric}'
        # Lower is better for language modeling
        return metrics[metric_name] < other[metric_name]
