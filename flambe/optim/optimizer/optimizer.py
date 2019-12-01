from typing import Type

import torch

from flambe import Component


class Optimizer(Component):
    """Adapter to the Pytorch Optimizer class.

    This object allows an optimizer to be instantiated without
    model parameters, so that the model can be safely put on the
    correct device by its wrapper object.

    """

    _cls: Type[torch.optim.optimizer.Optimizer]

    def __init__(self, params=None, **kwargs):
        self.initialized = False
        if parameters is not None:
            self.initialized = True
            self._cls.__init__(self, params=parameters, **kwargs)
        else:
            self.kwargs = kwargs

    def initialize(self, params):
        if self.initialized:
            raise ValueError("Optimizer already initialized with parameters.")

        self.initialized = True
        self._cls.__init__(self, params=params, **self.kwargs)
        del self.kwargs


class Adam(Optimizer, torch.optim.Adam):
    _cls = torch.optim.Adam


class SGD(Optimizer, torch.optim.SGD):
    _cls = torch.optim.SGD


class AdamW(Optimizer, torch.optim.AdamW):
    _cls = torch.optim.AdamW


class SparseAdam(Optimizer, torch.optim.SparseAdam):
    _cls = torch.optim.SparseAdam


class Adadelta(Optimizer, torch.optim.Adadelta):
    _cls = torch.optim.Adadelta


class Adamax(Optimizer, torch.optim.Adamax):
    _cls = torch.optim.Adamax


class ASGD(Optimizer, torch.optim.ASGD):
    _cls = torch.optim.ASGD


class LBFGS(Optimizer, torch.optim.LBFGS):
    _cls = torch.optim.LBFGS


class RMSprop(Optimizer, torch.optim.RMSprop):
    _cls = torch.optim.RMSprop


class Rprop(Optimizer, torch.optim.Rprop):
    _cls = torch.optim.Rprop
