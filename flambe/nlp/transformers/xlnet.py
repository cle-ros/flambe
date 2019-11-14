"""
Intergation of the transformers xlnet module.

Note that these objects are only to be used to load
pretrained models. The pytorch-transformers library
wasn't designed to train these models from scratch.

"""

import transformers as pt

from flambe.nlp.transformers.utils import TransformerTextField, TransformerEmbedder


class XLNetTextField(TransformerTextField):
    """Integrate the transformers XLNetTokenizer.

    Currently available aliases:
        . `xlnet-base-cased`
        . `xlnet-large-cased`

    """

    _cls = pt.XLNetTokenizer


class XLNetEmbedder(TransformerEmbedder):
    """Integrate the transformers XLNetModel.

    Currently available aliases:
        . `xlnet-base-cased`
        . `xlnet-large-cased`

    """

    _cls = pt.XLNetModel
