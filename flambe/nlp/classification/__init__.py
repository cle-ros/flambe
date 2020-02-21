# type: ignore[attr-define]

from flambe.nlp.classification.datasets import SSTDataset, TRECDataset, NewsGroupDataset
from .model import TextClassifier
from .model_multi_text import DualTextClassifier
from .datasets_nli import SNLIDataset, MultiNLIDataset, SCIDataset, \
    WNLIDataset, QQPDataset, QNLIDataset, RTEDataset

__all__ = [
    'TextClassifier',
    'SSTDataset',
    'TRECDataset',
    'NewsGroupDataset',
    'SNLIDataset',
    'MultiNLIDataset',
    'SCIDataset',
    'WNLIDataset',
    'QQPDataset',
    'QNLIDataset',
    'RTEDataset',
    'DualTextClassifier'
]
