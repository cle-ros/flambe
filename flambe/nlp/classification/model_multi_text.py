from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from flambe.nn import Embedder, Module


class DualTextClassifier(Module):
    """Implements a standard classifier.

    The classifier is composed of an encoder module, followed by
    a fully connected output layer, with a dropout layer in between.

    Attributes
    ----------
    embedder_1: Embedder
        The embedder layer
    embedder_2: Embedder
        The embedder layer
    output_layer : Module
        The output layer, yields a probability distribution over targets
    drop: nn.Dropout
        the dropout layer
    loss: Metric
        the loss function to optimize the model with
    metric: Metric
        the dev metric to evaluate the model on

    """

    def __init__(self,
                 embedder_1: Embedder,
                 embedder_2: Embedder,
                 output_layer: Module,
                 dropout: float = 0) -> None:
        """Initialize the TextClassifier model.

        Parameters
        ----------
        embedder_1: Embedder
            The embedder layer for the first sentence
        embedder_2: Embedder
            The embedder layer for the second sentence
        output_layer : Module
            The output layer, yields a probability distribution
        dropout : float, optional
            Amount of dropout to include between layers (defaults to 0)

        """
        super().__init__()

        self.embedder_1 = embedder_1
        self.embedder_2 = embedder_2
        self.output_layer = output_layer

        self.drop = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self,
                input_1: Tensor,
                input_2: Tensor,
                target: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Run a forward pass through the network.

        Parameters
        ----------
        input_1: Tensor
            The first sentence (premise)
        input_2: Tensor
            The second sentence (hypothesis)
        target: Tensor, optional
            The targets, optional

        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor]
            The output predictions, and optionally the targets

        """
        encoding_1 = self.embedder_1(input_1)
        encoding_2 = self.embedder_2(input_2)
        if isinstance(encoding_1, tuple):
            encoding_1 = encoding_1[0]
        else:
            encoding_1 = encoding_1
        if isinstance(encoding_2, tuple):
            encoding_2 = encoding_2[0]
        else:
            encoding_2 = encoding_2

        pred = self.output_layer(self.drop(torch.cat((encoding_1, encoding_2), dim=1)))
        return (pred, target) if target is not None else pred
