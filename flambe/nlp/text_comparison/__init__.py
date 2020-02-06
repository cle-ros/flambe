from flambe.nlp.text_comparison.datasets_nli import SNLIDataset, MultiNLIDataset, SCIDataset, WNLIDataset, \
    QQPDataset, QNLIDataset, RTEDataset
from flambe.nlp.text_comparison.datasets_ranking_dialog_prediction import ConvAI2Dataset
from flambe.nlp.text_comparison.model import DualTextClassifier

__all__ = ['SNLIDataset', 'MultiNLIDataset', 'DualTextClassifier', 'SCIDataset', 'WNLIDataset', 'QQPDataset',
           'QNLIDataset', 'RTEDataset', 'ConvAI2Dataset']
