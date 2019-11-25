import typing

from abc import abstractmethod

from flambe.compile import Component


class Metric(Component):
    """Base Metric interface.

    Objects implementing this interface should take in a sequence of
    examples and provide as output a processd list of the same size.

    """

    @typing.no_type_check
    @abstractmethod
    def compute(self, *args, **kwargs):
        """Computes the metric."""
        pass

    def __call__(self, *args, **kwargs):
        """Makes Featurizer a callable."""
        return self.compute(*args, **kwargs)

    def __str__(self) -> str:
        """Return the name of the Metric (for use in logging)."""
        return self.__class__.__name__
