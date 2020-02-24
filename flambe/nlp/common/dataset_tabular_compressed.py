from typing import Dict, Union

import abc
import os
import tempfile

from urllib.parse import unquote, urlparse
import requests
import shutil

from flambe.dataset import TabularDataset
from flambe.field import Field


class CompressedDataTabularDataset(TabularDataset, metaclass=abc.ABCMeta):
    """Can download and unzip data files

    An extension to TabularDataset that supports downloading and
    extracting files
    """
    NAME = None
    URL = None
    EXTRA_TOKENS = None
    NAMED_COLS = ('text_1', 'text_2', 'label')

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
        tmp_folder = self._download_and_extract_files(**kwargs)

        # loading each dataset
        file_names = self._get_dataset_files(**kwargs)
        train_path = os.path.join(tmp_folder, file_names[0])
        dev_path = os.path.join(tmp_folder, file_names[1])
        test_path = os.path.join(tmp_folder, file_names[2])

        train, _ = self._load_file(train_path)
        val, _ = self._load_file(dev_path)
        test, _ = self._load_file(test_path)

        super().__init__(train, val, test, cache, self.NAMED_COLS, transform)

    @classmethod
    def _download_and_extract_files(cls, **kwargs):
        # data handling
        # making sure it needs to be downloaded:
        for f in os.listdir(tempfile.gettempdir()):
            folder = os.path.join(tempfile.gettempdir(), f)
            if os.path.isdir(folder) and f.startswith(f'flambe_{cls.NAME}_'):
                if not all([file in os.listdir(folder)
                            for file in cls._get_dataset_files(**kwargs)]):
                    # partial download or tmp cleanup, remove folder
                    shutil.rmtree(folder)
                    continue
                tmp_folder = folder
                break
        else:
            # downloading and unzipping into tmp folder
            # creating tmp folder
            tmp_folder = tempfile.mkdtemp(prefix=f'flambe_{cls.NAME}_')
            # figuring out filename
            file_name = unquote(urlparse(cls.URL).path).split('/')[-1]
            zip_path = os.path.join(tmp_folder, file_name)
            # download and write to disk
            data_file = requests.get(cls.URL)
            with open(zip_path, 'wb') as outfile:
                outfile.write(data_file.content)
            # unzip
            shutil.unpack_archive(zip_path, tmp_folder)
            # remove file
            os.remove(zip_path)
            # collect files to flatten folder structure
            walker = os.walk(tmp_folder)
            for data in walker:
                for files in data[2]:
                    try:
                        shutil.move(data[0] + os.sep + files, tmp_folder)
                    except shutil.Error:
                        # same folder
                        continue
        return tmp_folder
