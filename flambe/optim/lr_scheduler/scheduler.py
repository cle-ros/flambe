from typing import Type

from torch.optim import lr_scheduler
from flambe.compile import Component


class LRScheduler(Component, lr_scheduler._LRScheduler):

    _cls: Type[lr_scheduler._LRScheduler]

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


class LambdaLR(LRScheduler, lr_scheduler.LambdaLR):
    _cls = lr_scheduler.LambdaLR


class StepLR(LRScheduler, lr_scheduler.StepLR):
    _cls = lr_scheduler.StepLR


class MultiStepLR(LRScheduler, lr_scheduler.MultiStepLR):
    _cls = lr_scheduler.MultiStepLR


class ExponentialLR(LRScheduler, lr_scheduler.ExponentialLR):
    _cls = lr_scheduler.ExponentialLR


class CosineAnnealingLR(LRScheduler, lr_scheduler.CosineAnnealingLR):
    _cls = lr_scheduler.CosineAnnealingLR


class ReduceLROnPlateau(LRScheduler, lr_scheduler.ReduceLROnPlateau):  # type: ignore
    _cls = lr_scheduler.ReduceLROnPlateau  # type: ignore


class CyclicLR(LRScheduler, lr_scheduler.CyclicLR):
    _cls = lr_scheduler.CyclicLR


class OneCycleLR(LRScheduler, lr_scheduler.OneCycleLR):  # type: ignore
    _cls = lr_scheduler.OneCycleLR  # type: ignore


class CosineAnnealingWarmRestarts(LRScheduler, lr_scheduler.CosineAnnealingWarmRestarts):
    _cls = lr_scheduler.CosineAnnealingWarmRestarts
