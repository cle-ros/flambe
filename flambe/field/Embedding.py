import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import temporary_file


def Embedding():
    def vector_size(self):
        pass

    def __getitem__(self, key):
        pass


def ExternalEmbedding(Embedding):
    def __init__(self, delegate):
        self.delegate = delegate

    def vector_size(self):
        return self.delegate.vector_size()

    def __getitem__(self, key):
        return self.delegate[key]


def build_glove(pretrained: str, binary: bool):
    with temporary_file('temp.txt') as temp:
        glove2word2vec(pretrained, temp)
        return ExternalEmbedding(KeyedVectors.load_word2vec_format(temp, binary=binary))


def build_word2vec(pretrained: str, binary: bool):
    return ExternalEmbedding(KeyedVectors.load_word2vec_format(pretrained,
                                                               binary=binary))


def build_gensim(pretrained: str):
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
    return ExternalEmbedding(model)
