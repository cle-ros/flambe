from typing import List, Tuple, Optional, Dict, Union
import abc
import os
import json
import csv
from flambe.nlp.common import CompressedDataTabularDataset


class NLIDataset(CompressedDataTabularDataset, metaclass=abc.ABCMeta):
    """
    The base class for the NLI class of tasks.
    """

    NAMED_COLS = ('text_1', 'text_2', 'label')

    @classmethod
    def _load_file(cls,
                   path: str,
                   sep: Optional[str] = '\t',
                   header: Optional[str] = None,
                   columns: Optional[Union[List[str], List[int]]] = None,
                   encoding: Optional[str] = 'utf-8') -> Tuple[List[Tuple], Optional[List[str]]]:
        """Load data from the given path."""
        data = []
        if path.endswith('jsonl') or path.endswith('json'):
            with open(path, 'r') as file:
                for line in file:
                    sample = json.loads(line.strip())
                    if not sample['gold_label'].strip() in \
                           ['entailment', 'neutral', 'contradiction']:
                        continue
                    data.append((sample['sentence1'],
                                 sample['sentence2'],
                                 sample['gold_label'].strip()))
        elif path.endswith('tsv'):  # tab-separated table
            with open(path, 'r') as file:
                reader = csv.reader(file, dialect='excel-tab')
                header = next(reader)
                appendix, tail = (['n/a'], -2) if len(header) == 3 else ([], -3)
                data = filter(lambda x: len(x) == len(header), reader)
                data = [[*sample[tail:], *appendix] for sample in data]
        else:
            raise ValueError(f'Unrecognized filetype for {os.path.split(path)[-1]}.')
        return data, None

    @staticmethod
    @abc.abstractmethod
    def _get_dataset_files(**kwargs) -> Tuple[str, str, str]:
        pass


class GLUEDataset(NLIDataset):
    @staticmethod
    def _get_dataset_files(variation='basic', **kwargs):
        return 'train.tsv', 'dev.tsv', 'test.tsv'


class SNLIDataset(NLIDataset):
    """ SNLI

        See: https://nlp.stanford.edu/projects/snli/

        Cite:
        @inproceedings{snli:emnlp2015,
            Author = {Bowman, Samuel R. and Angeli, Gabor and Potts,
            Christopher, and Manning, Christopher D.},
            Booktitle = {Proceedings of the 2015 Conference on Empirical
            Methods in Natural Language Processing (EMNLP)},
            Publisher = {Association for Computational Linguistics},
            Title = {A large annotated corpus for learning natural
            language inference},
            Year = {2015}
        }
    """

    NAME = 'SNLI'
    URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"

    @staticmethod
    def _get_dataset_files(**kwargs):
        return ['snli_1.0_train.jsonl', 'snli_1.0_dev.jsonl', 'snli_1.0_test.jsonl']


class MultiNLIDataset(NLIDataset):
    """MultiNLI.
        See https://www.nyu.edu/projects/bowman/multinli/

        Cite:
        @InProceedings{N18-1101,
            author = "Williams, Adina
                    and Nangia, Nikita
                    and Bowman, Samuel",
            title = "A Broad-Coverage Challenge Corpus for
                   Sentence Understanding through Inference",
            booktitle = "Proceedings of the 2018 Conference of
                       the North American Chapter of the
                       Association for Computational Linguistics:
                       Human Language Technologies, Volume 1 (Long
                       Papers)",
            year = "2018",
            publisher = "Association for Computational Linguistics",
            pages = "1112--1122",
            location = "New Orleans, Louisiana",
            url = "http://aclweb.org/anthology/N18-1101"
        }
    """

    NAME = 'MultiNLI'
    URL = "https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip"

    @staticmethod
    def _get_dataset_files(variation='basic', **kwargs):
        if not variation in ('basic', 'matched', 'mismatched'):
            raise ValueError(f'Only supported variations for MultiNLI '
                             f'are basic/matched and mismatched. '
                             f'Got {variation}.')
        mismatched = variation == 'mismatched'
        dev_file = 'multinli_1.0_dev_matched.jsonl' if not mismatched \
            else 'multinli_1.0_dev_mismatched.jsonl'
        print(f'MultiNLI: Using the {"matched" if not mismatched else "mismatched"} '
              f'version of the eval dataset.')
        return ['multinli_1.0_train.jsonl', dev_file, dev_file]


class SCIDataset(NLIDataset):
    """SCI.
        See https://nlp.stanford.edu/projects/sci/

        Cite:
        @inproceedings{cases-etal-2019-recursive,
            title = "Recursive Routing Networks: Learning to Compose
            Modules for Language Understanding",
            author = "Cases, Ignacio  and
              Rosenbaum, Clemens  and
              Riemer, Matthew  and
              Geiger, Atticus  and
              Klinger, Tim  and
              Tamkin, Alex  and
              Li, Olivia  and
              Agarwal, Sandhini  and
              Greene, Joshua D.  and
              Jurafsky, Dan  and
              Potts, Christopher  and
              Karttunen, Lauri",
            booktitle = "Proceedings of the 2019 Conference of the North
            {A}merican Chapter of the Association for Computational
            Linguistics: Human Language Technologies,
            Volume 1 (Long and Short Papers)",
            month = jun,
            year = "2019",
            address = "Minneapolis, Minnesota",
            publisher = "Association for Computational Linguistics",
            url = "https://www.aclweb.org/anthology/N19-1365",
            doi = "10.18653/v1/N19-1365",
            pages = "3631--3648",
        }
    """

    NAME = 'SCI'
    URL = "https://nlp.stanford.edu/projects/sci/sci_dataset.zip"

    @staticmethod
    def _get_dataset_files(variation='basic', **kwargs):
        if variation not in ('basic', 'joint', 'matched', 'mismatched', 'disjoint', 'nested'):
            raise ValueError(f'Only supported variations for SCI '
                             f'are basic/joint/matched, mismatched, '
                             f'disjoint and nested. Got {variation}.')
        if variation == 'mismatched':
            train_file = 'ci_latest_train_mismatch.json'
            dev_file = 'ci_latest_dev_joint.json'
            test_file = 'ci_latest_test_joint.json'
            type_str = 'mismatched'
        elif variation == 'disjoint':
            train_file = 'ci_latest_train_disjoint.json'
            dev_file = 'ci_latest_dev_disjoint.json'
            test_file = 'ci_latest_test_disjoint.json'
            type_str = 'disjoint'
        elif variation == 'nested':
            train_file = 'ci_latest_train_nested.json'
            dev_file = 'ci_latest_dev_nested.json'
            test_file = 'ci_latest_test_nested.json'
            type_str = 'nested'
        else:
            train_file = 'ci_latest_train_joint.json'
            dev_file = 'ci_latest_dev_joint.json'
            test_file = 'ci_latest_test_joint.json'
            type_str = 'basic'
        print(f'SCI: Using the {type_str} version of the eval dataset.')
        return train_file, dev_file, test_file


class WNLIDataset(GLUEDataset):
    """MultiNLI.
        See https://www.aclweb.org/anthology/W18-5446/
        https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html
    """

    NAME = 'WNLI'
    URL = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/" \
          "o/data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf"


class QNLIDataset(GLUEDataset):
    """QNLI.
        See https://www.aclweb.org/anthology/W18-5446/
        See https://rajpurkar.github.io/SQuAD-explorer/
    """

    NAME = 'QNLI'
    URL = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/" \
          "o/data%2FQNLIv2.zip?alt=media&token=6fdcf570-0fc5-4631-8456-9505272d1601"


class RTEDataset(GLUEDataset):
    """MultiNLI.
        See https://www.aclweb.org/anthology/W18-5446/
        See https://rajpurkar.github.io/SQuAD-explorer/
    """

    NAME = 'RTE'
    URL = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/" \
          "o/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb"


class QQPDataset(GLUEDataset):
    """QQP.
        See https://www.aclweb.org/anthology/W18-5446/
        See https://www.quora.com/q/quoradata/First-Quora-Dataset-
        Release-Question-Pairs
    """

    NAME = 'QQP'
    URL = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/" \
          "o/data%2FQQP.zip?alt=media&token=700c6acf-160d-4d89-81d1-de4191d02cb5"
