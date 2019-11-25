from typing import Type

import torch
from flambe.compile import Component


class LRScheduler(Component):

    _cls: Type[torch.optim.lr_scheduler._LRScheduler]

    def __init__(self, optimizer=None, **kwargs):
        self.initialized = False
        if optimizer is not None:
            self.initialized = True
            self._cls.__init__(self, optimizer=optimizer, **kwargs)
        else:
            self.kwargs = kwargs

    def initialize(self, optimizer):
        if self.initialized:
            raise ValueError("Scheduler already initialized with optimizer.")

        self.initialized = True
        self._cls.__init__(self, optimizer=optimizer, **self.kwargs)
        del self.kwargs

    def state_dict(self):
        state_dict = super().state_dict()
        del state_dict['_schema']
        del state_dict['_saved_kwargs']
        del state_dict['_extensions']
        return state_dict


class LambdaLR(LRScheduler):
    _cls = torch.optim.lr_scheduler.LambdaLR


class StepLR(LRScheduler):
    _cls = torch.optim.lr_scheduler.StepLR


class MultiStepLR(LRScheduler):
    _cls = torch.optim.lr_scheduler.MultiStepLR


class ExponentialLR(LRScheduler):
    _cls = torch.optim.lr_scheduler.ExponentialLR


class CosineAnnealingLR(LRScheduler):
    _cls = torch.optim.lr_scheduler.CosineAnnealingLR


class ReduceLROnPlateau(LRScheduler):
    _cls = torch.optim.lr_scheduler.ReduceLROnPlateau  # type: ignore


class CyclicLR(LRScheduler):
    _cls = torch.optim.lr_scheduler.CyclicLR


class OneCycleLR(LRScheduler):
    _cls = torch.optim.lr_scheduler.OneCycleLR  # type: ignore


class CosineAnnealingWarmRestarts(LRScheduler):
    _cls = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
