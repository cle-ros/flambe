from typing import Optional, List

import torch
import torch.nn as nn

from flambe.nn.module import Encoder


class MLPEncoder(Encoder):
    """Implements a multi layer feed forward network.

    This module can be used to create output layers, or
    more complex multi-layer feed forward networks.

    Attributes
    ----------
    seq: nn.Sequential
        the sequence of layers and activations

    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 n_layers: int = 1,
                 dropout: float = 0.,
                 output_activation: Optional[nn.Module] = None,
                 hidden_size: Optional[int] = None,
                 hidden_activation: Optional[nn.Module] = None) -> None:
        """Initializes the FullyConnected object.

        Parameters
        ----------
        input_size: int
            Input_dimension
        output_size: int
            Output dimension
        n_layers: int, optional
            Number of layers in the network, defaults to 1
        dropout: float, optional
            Dropout to be used before each MLP layer.
            Only used if n_layers > 1.
        output_activation: nn.Module, optional
            Any PyTorch activation layer, defaults to None
        hidden_size: int, optional
            Hidden dimension, used only if n_layers > 1.
            If not given, defaults to the input_size
        hidden_activation: nn.Module, optional
            Any PyTorch activation layer, defaults to None

        """
        super().__init__()

        # Gather the layers in a list to pass to Sequential
        layers: List[nn.Module] = []

        # Add the hidden_layers
        if n_layers == 1 or hidden_size is None:
            hidden_size = input_size

        if n_layers > 1:

            # Add the first hidden layer
            layers.append(nn.Linear(input_size, hidden_size))
            if hidden_activation is not None:
                layers.append(hidden_activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            for _ in range(1, n_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                if hidden_activation is not None:
                    layers.append(hidden_activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_size, output_size))
        if output_activation is not None:
            layers.append(output_activation)

        self.input_size = input_size
        self.output_size = output_size
        self.seq = nn.Sequential(*layers)

    @property
    def input_dim(self) -> int:
        """Get the size of the last dimension of an input.

        Returns
        -------
        int
            The size of the last dimension of an input.

        """
        return self.input_size

    @property
    def output_dim(self) -> int:
        """Get the size of the last dimension of an output.

        Returns
        -------
        int
            The size of the last dimension of an output.

        """
        return self.output_size

    def forward(self, data: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Performs a forward pass through the network.

        Parameters
        ----------
        data: torch.Tensor
            input to the model of shape (batch_size, input_size)

        Returns
        -------
        output: torch.Tensor
            output of the model of shape (batch_size, output_size)

        """
        return self.seq(data)
