from abc import abstractmethod
from typing import Iterable, Dict, TypeVar, Any

from torch import Tensor
from flambe.dataset import Dataset

from flambe.nn import Module


Data = TypeVar('Data', bound=Any)
Batch = TypeVar('Batch', bound=Any)


class Model(Module):

    @abstractmethod
    def transform(self, data: Data) -> Data:
        """Compute loss on the batch"""
        pass

    @abstractmethod
    def train_batch(self, batch: Batch) -> Tensor:
        """Compute loss on the batch"""
        pass

    @abstractmethod
    def eval_batch(self, batch: Batch) -> Dict[str, Tensor]:
        """Compute metrics on the batch"""
        pass

    @abstractmethod
    def train_sampler(self, data: Data) -> Iterable[Batch]:
        """Sample batches of data."""
        pass

    @abstractmethod
    def eval_sampler(self, data: Data) -> Iterable[Batch]:
        """Sample batches of data"""
        pass
