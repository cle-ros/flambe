"""
Intergation of the transformers roberta module.

Note that these objects are only to be used to load
pretrained models. The pytorch-transformers library
wasn't designed to train these models from scratch.

"""

import transformers as pt

from flambe.nlp.transformers.utils import TransformerTextField, TransformerEmbedder


class RobertaTextField(TransformerTextField):
    """Integrate the transformers RobertaTokenizer.

    Currently available aliases:
        . `roberta-base`
        . `roberta-large`
        . `roberta-large-mnli`

    """

    _cls = pt.RobertaTokenizer


class RobertaEmbedder(TransformerEmbedder):
    """Integrate the transformers RobertaModel.

    Currently available aliases:
        . `roberta-base`
        . `roberta-large`
        . `roberta-large-mnli`

    """

    _cls = pt.RobertaModel
