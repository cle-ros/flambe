import torch
from flambe.compile import Component


class LRScheduler(torch.optim.lr_scheduler._LRScheduler, Component):

    _cls: torch.optim.lr_scheduler._LRScheduler

    def __init__(self, optimizer=None, **kwargs):
        self.initialized = False
        if optimizer is not None:
            self.initialized = True
            super().__init__(optimizer, **kwargs)
        else:
            self.kwargs = kwargs

    def initialize(self, optimizer):
        if self.initialized:
            raise ValueError("Scheduler already initialized with optimizer.")

        self.initialized = True
        self._cls.__init__(self, optimizer, **self.kwargs)
        del self.kwargs

    def state_dict(self):
        state_dict = super().state_dict()
        del state_dict['_schema']
        del state_dict['_saved_kwargs']
        del state_dict['_extensions']
        return state_dict


class LambdaLR(LRScheduler, torch.optim.lr_scheduler.LambdaLR):
    pass
