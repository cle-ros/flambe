import torch

from flambe import Component


class Optimizer(torch.optim.optimizer.Optimizer, Component):
    """Adapter to the Pytorch Optimizer class.

    This object allows an optimizer to be instantiated without
    model parameters, so that the model can be safely put on the
    correct device by its wrapper object.

    """

    _cls: torch.optim.optimizer.Optimizer

    def __init__(self, parameters=None, **kwargs):
        self.initialized = False
        if parameters is not None:
            self.initialized = True
            super().__init__(parameters, **kwargs)
        else:
            self.kwargs = kwargs

    def initialize(self, parameters):
        if self.initialized:
            raise ValueError("Optimizer already initialized with parameters.")

        self.initialized = True
        self._cls.__init__(self, parameters, **self.kwargs)
        del self.kwargs


class Adam(Optimizer):
    _cls: torch.optim.Adam


class SGD(Optimizer):
    _cls: torch.optim.SGD
