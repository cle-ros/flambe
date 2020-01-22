from typing import List, Tuple, Optional, Dict, Union
import abc
import os
import tempfile

import requests
from zipfile import ZipFile
import json

from flambe.dataset import TabularDataset
from flambe.field import Field


class NLIDataset(TabularDataset, metaclass=abc.ABCMeta):
    """
    The base class for the NLI class of tasks.
    """

    NAME = None
    URL = None

    def __init__(self,
                 cache: bool = True,
                 transform: Dict[str, Union[Field, Dict]] = None,
                 **kwargs) -> None:
        """Initialize the SSTDataset builtin.

        Parameters
        ----------
        binary: bool
            Set to true to train and evaluate in binary mode.
            Defaults to True.
        phrases: bool
            Set to true to train on phrases. Defaults to False.

        """

        # data handling
        # making sure it needs to be downloaded:
        for f in os.listdir(tempfile.gettempdir()):
            file = os.path.join(tempfile.gettempdir(), f)
            if os.path.isdir(file) and f.startswith(f'flambe_{self.NAME}'):
                tmp_folder = file
                break
        else:
            # downloading and unzipping into tmp folder
            tmp_folder = tempfile.mkdtemp(prefix=f'flambe_{self.NAME}')
            zip_path = os.path.join(tmp_folder, 'data.zip')
            data_file = requests.get(self.URL)
            with open(zip_path, 'wb') as outfile:
                outfile.write(data_file.content)
            with ZipFile(zip_path, 'r') as zipobj:
                for zip_info in zipobj.infolist():
                    if zip_info.filename[-1] == '/':
                        continue
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zipobj.extract(zip_info, tmp_folder)
                # zipobj.extractall(tmp_folder)

        # loading each dataset
        file_names = self._get_dataset_files(**kwargs)
        train_path = os.path.join(tmp_folder, file_names[0])
        dev_path = os.path.join(tmp_folder, file_names[1])
        test_path = os.path.join(tmp_folder, file_names[2])

        train, _ = self._load_file(train_path)
        val, _ = self._load_file(dev_path)
        test, _ = self._load_file(test_path)

        named_cols = ['premise', 'hypothesis', 'label']
        super().__init__(train, val, test, cache, named_cols, transform)

    @classmethod
    def _load_file(cls,
                   path: str,
                   sep: Optional[str] = '\t',
                   header: Optional[str] = None,
                   columns: Optional[Union[List[str], List[int]]] = None,
                   encoding: Optional[str] = 'utf-8') -> Tuple[List[Tuple], Optional[List[str]]]:
        """Load data from the given path."""
        data = []
        with open(path, 'r') as file:
            for line in file:
                sample = json.loads(line.strip())
                if not sample['gold_label'].strip() in ['entailment', 'neutral', 'contradiction']:
                    continue
                data.append((sample['sentence1'], sample['sentence2'], sample['gold_label'].strip()))
        return data, None

    @staticmethod
    @abc.abstractmethod
    def _get_dataset_files(**kwargs) -> Tuple[str, str, str]:
        pass


class SNLIDataset(NLIDataset):
    """ SNLI

        See: https://nlp.stanford.edu/projects/snli/

        Cite:
        @inproceedings{snli:emnlp2015,
            Author = {Bowman, Samuel R. and Angeli, Gabor and Potts, Christopher, and Manning, Christopher D.},
            Booktitle = {Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
            Publisher = {Association for Computational Linguistics},
            Title = {A large annotated corpus for learning natural language inference},
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
            raise ValueError(f'Only supported variations for MultiNLI are basic/matched and mismatched. '
                             f'Got {variation}.')
        mismatched = variation == 'mismatched'
        dev_file = 'multinli_1.0_dev_matched.jsonl' if not mismatched else 'multinli_1.0_dev_mismatched.jsonl'
        print(f'MultiNLI: Using the {"matched" if not mismatched else "mismatched"} version of the eval dataset.')
        return ['multinli_1.0_train.jsonl', dev_file, dev_file]


class SCIDataset(NLIDataset):
    """MultiNLI.
        See https://nlp.stanford.edu/projects/sci/

        Cite:
        @inproceedings{cases-etal-2019-recursive,
            title = "Recursive Routing Networks: Learning to Compose Modules for Language Understanding",
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
            booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
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
            raise ValueError(f'Only supported variations for SCI are basic/joint/matched, mismatched, '
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
