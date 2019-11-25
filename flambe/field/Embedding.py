from typing import Optional, Dict, Set
from collections import OrderedDict as odict

import torch
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import temporary_file

from flambe.field import Field
from flambe.tokenizer import Tokenizer, WordTokenizer


def build_glove(pretrained, binary):
    with temporary_file('temp.txt') as temp:
        glove2word2vec(pretrained, temp)
        return KeyedVectors.load_word2vec_format(temp, binary=binary)

def build_word2vec(pretrained, binary):
    return KeyedVectors.load_word2vec_format(pretrained,
                                              binary=binary)
def build_gensim(pretrained):
    """
    embeddings_format : str, optional
            The format of the input embeddings, should be one of:
            'glove', 'word2vec', 'fasttext' or 'gensim'. The latter can
            be used to download embeddings hosted on gensim on the fly.
            See https://github.com/RaRe-Technologies/gensim-data
            for the list of available embedding aliases.
        embeddings_binary : bool, optional
            Whether the input embeddings are provided in binary format,
            by default False
    :param pretrained:
    :return:
    """
    try:
        model = KeyedVectors.load(pretrained)
    except FileNotFoundError:
        model = api.load(pretrained)
    return model