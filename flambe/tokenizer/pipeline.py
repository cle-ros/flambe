from typing import List, Callable

from flambe.tokenizer import Tokenizer


class PipelineTokenizer(Tokenizer):
    """Base tokenizer implementation that receives a set of
    transformations to be applied to the input string.

    """
    def __init__(self,
                 transformations: List[Callable[[str], str]],
                 **kwargs) -> None:
        """
        Parameters
        ----------
        transformations: List[Callable[[str], str]]
            The list of transformations to be applied. For example:
            [lambda x: x.lower()]

        """
        super().__init__(**kwargs)

        if transformations is None:
            raise ValueError("No transformations were passed.")

        self.transformations = transformations

    def tokenize(self, example: str) -> List[str]:
        """Tokenize an input example by applying the transformations
        in the given order.

        Parameters
        ----------
        example : str
            The input example, as a string

        Returns
        -------
        List[str]
            The output word tokens, as a list of strings

        """
        for transformation in self.transformations:
            example = transformation(example)

        return example
