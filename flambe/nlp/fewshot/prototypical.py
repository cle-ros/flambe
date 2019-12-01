from typing import Dict, Optional, Any

from flambe.dataset import TabularDataset
from flambe.sampler import EpisodicSampler
from flambe.nn import Encoder, Embedder
from flambe.nn.distance import get_distance_module, get_mean_module
from flambe.nlp.classification import TextClassification


class PrototypicalTextClassifier(Encoder):
    """Implements a standard classifier."""

    def __init__(self, embedder, distance='euclidean', detach_mean=False) -> None:
        """Initialize the TextClassifier model."""
        super().__init__()

        self.embedder = embedder
        self.distance_module = get_distance_module(distance)
        self.mean_module = get_mean_module(distance)
        self.detach_mean = detach_mean

    @property
    def input_dim(self) -> int:
        """Get the size of the last dimension of an input."""
        return self.embedder.input_dim

    @property
    def output_dim(self) -> int:
        """Get the size of the last dimension of an output."""
        return self.embedder.output_dim

    def compute_prototypes(self, support, label):
        """Set the current prototypes used for classification."""
        means_dict: Dict[int, Any] = {}
        for i in range(support.size(0)):
            means_dict.setdefault(int(label[i]), []).append(support[i])

        means = []
        n_means = len(means_dict)

        for i in range(n_means):
            # Ensure that all contiguous indices are in the means dict
            supports = torch.stack(means_dict[i], dim=0)
            if supports.size(0) > 1:
                mean = self.mean_module(supports).squeeze(0)
            else:
                mean = supports.squeeze(0)
            means.append(mean)

        prototypes = torch.stack(means, dim=0)
        return prototypes

    def forward(self, query, support=None, support_label=None):
        """Perform a forward pass through the model."""
        query_encoding = self.embedder(query)

        if support is None:
            return query_encoding

        else:
            if support_label is None:
                raise ValueError("No labels provided for the support set.")
            if self.detach_mean:
                support = support.detach()
                support_label = support_label.detach()  # type: ignore

            support_encoding = self.embedder(support)
            prototypes = self.compute_prototypes(support_encoding, support_label)
            dist = self.distance_module(query_encoding, prototypes)
            return - dist


class PrototypicalTextClassification(TextClassification):
    """A text classification task.

    Performs text classifcation training and evaluation.
    Takes as input a dataset and an encoder, and constructs
    a simple TextClassifier. You may pass your custom fields
    or used the defaults. The loss is computer via a cross entropy,
    and the validation metric is accuracy.

    """

    def __init__(self,
                 dataset: TabularDataset,
                 distance: str = 'euclidean',
                 detach_mean: bool = False,
                 dropout: float = 0,
                 build_vocabularies: bool = True,
                 n_classes: Optional[int] = None,
                 n_support: int = 5,
                 n_query: int = 10,
                 n_episodes: int = 1000,
                 balance_query: bool = True,
                 **kwargs) -> None:
        """Initalize a TextClassification task.

        Parameters
        ----------
        dataset : TabularDataset
            The input dataset
        distance: str, optional
            One of [euclidean, cosine, hyperbolic].
            Default ``euclidean``.
        detatch_mean: bool, optional
            Whether to detach gradients with respect to the mean.
            Default ``False``.
        n_classes: int, optional
            The number of classes to sampler per episode. Default is
            ``None``, which means all classes are used every episode.
        n_support: int, optional
            Number of points to sample per class as supports.
            Default ``5``.
        n_query: int, optional
            Number of points to sample per class as query if
            ``balance_query`` is set. Otherwise the total number
            of query points. Default ``10``.
        n_episodes: int, optional
            Number of episodes to consider as one epoch. This is the
            number of episodes used for evaluation.
        balance_query: bool, optional
            If set, ``n_query`` is considered to be the number
            of points to sampler per class as query. If
            ``balance_query`` is ``False``, then ``n_query`` is treated
            as the total number of query points across all classes.

        See the ``TextClassification`` parent class for other
        keyword arguments.

        """
        super().__init__(dataset, **kwargs)
        self.distance = distance
        self.n_episodes = n_episodes
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query
        self.balance_query = balance_query
        self.detach_mean = detach_mean

    def build_model(self):
        """Build the model, containing all parameters to train."""
        if self._model is not None:
            return self._model
        if self.embedder is None:
            input_size = self.text_field.vocab_size
            embedding_dim = self.encoder.input_dim
            padding_idx = self.text_field.vocab[self.text_field.pad]
            embeddings = Embeddings(input_size, embedding_dim, padding_idx, **self.embedding_args)
            self.embedder = Embedder(embeddings, self.encoder, self.pooling, self.embedding_dropout)
        return PrototypicalTextClassifier(self.embedder, self.distance, self.detach_mean)

    def sample_batches(self, split='train', train=True):
        """Get an iterable of batches of data."""
        pad_index = self.text_field.vocab[self.text_field.pad]
        return EpisodicSampler(getattr(self.dataset, split),
                               self.n_support,
                               self.n_query,
                               self.n_episodes,
                               n_classes=n_classes,
                               pad_index=pad_index)

    def train_step(self, model, batch):
        """Compute loss on the given batch during training."""
        batch = (tensor.to(self.device) for tensor in batch)
        query, query_label, support, support_label = batch
        preds = model(query, support, support_label)
        loss = self.loss_fn(preds, query_label)
        return loss

    def val_step(self, model, batches):
        """Compute metrics on the validation set, given in batches."""
        total_loss, total_acc, total_count = 0, 0, 0
        for batch in batches:
            batch = (tensor.to(self.device) for tensor in batch)
            query, query_label, support, support_label = batch
            preds = model(query, support, support_label)
            total_count += preds.size(0)
            total_loss += self.loss_rn(preds, query_label).sum().item()
            total_acc += self.metric_fn(preds, labels).sum().item()
        loss = total_loss / total_count
        accuracy = total_acc / total_count
        return {'loss': loss, 'accuracy': accuracy}
