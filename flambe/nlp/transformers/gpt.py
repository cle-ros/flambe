"""
Intergation of the transformers openai and gpt2 modules.

Note that these objects are only to be used to load
pretrained models. The pytorch-transformers library
wasn't designed to train these models from scratch.

"""
from typing import Optional

import transformers as pt

from flambe.nlp.transformers.utils import TransformerTextField, TransformerEmbedder


class GPTTextField(TransformerTextField):
    """Integrate the transformers OpenAIGPTTokenizer.

    Currently available aliases:
        . `openai-gpt`

    """

    _cls = pt.OpenAIGPTTokenizer


class GPTEmbedder(TransformerEmbedder):
    """Integrate the transformers OpenAIGPTmodel.

    Currently available aliases:
        . `openai-gpt`

    """

    _cls = pt.OpenAIGPTModel


class GPT2TextField(TransformerTextField):
    """Integrate the transformers GPT2Tokenizer.

    Currently available aliases:
        . `gpt2`
        . `gpt2-medium`
        . `gpt2-large`

    """

    _cls = pt.GPT2Tokenizer


class GPT2Embedder(TransformerEmbedder):
    """Integrate the transformers GPT2Model.

    Currently available aliases:
        . `gpt2`
        . `gpt2-medium`
        . `gpt2-large`

    """

    _cls = pt.GPT2Model

    def __init__(self,
                 alias: str,
                 cache_dir: Optional[str] = None,
                 padding_idx: Optional[int] = None,
                 pool: bool = False, **kwargs) -> None:

        if pool:
            raise ValueError('GPT2 does not support pooling.')

        super().__init__(alias, cache_dir, padding_idx, pool, **kwargs)
