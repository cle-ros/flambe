from typing import List, Callable

from flambe.tokenizer import Tokenizer


class PipelineTokenizer(Tokenizer):
    """Base tokenizer implementation that receives a set of
    transformations to be applied to the input string.

    This class is useful when the input string needs some
    considerable amount of transformations before converting
    it in a list of tokens. This transformations could include
    lowecasing, removing numbers, currencies, etc.

    """
    def __init__(self,
                 transformations: List[Callable[[str], str]],
                 final_step: Callable[[str], List[str]],
                 **kwargs) -> None:
        """
        Parameters
        ----------
        transformations: List[Callable[[str], str]]
            The list of transformations to be applied to the string.
            For example: [lambda x: x.lower()]
            Each transformation should be a function that receives
            a string and returns a string.
        final_step: Callablep[[str], List[str]]
            Callable that receives the string and does the final
            tokenization returning a list of strings.

        """
        super().__init__(**kwargs)

        if transformations is None or final_step is None:
            raise ValueError("You need to provide 'transformations' and 'final_step'")

        self.transformations = transformations
        self.final_step = final_step

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

        return self.final_step(example)
