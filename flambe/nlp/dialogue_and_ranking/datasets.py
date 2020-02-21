from typing import List, Tuple, Optional, Union
import re
import csv
import json
import random

from flambe.nlp.common import CompressedDataTabularDataset


class ConvAI2Dataset(CompressedDataTabularDataset):
    """ ConvAI2 dataset from the NeurIPS convai challenge

    See: http://convai.io/#personachat-convai2-dataset

    Cite:
    @article{DBLP:journals/corr/abs-1801-07243,
        author    = {Saizheng Zhang and
                     Emily Dinan and
                     Jack Urbanek and
                     Arthur Szlam and
                     Douwe Kiela and
                     Jason Weston},
        title     = {Personalizing Dialogue Agents: {I} have a dog,
                     do you have pets too?},
        journal   = {CoRR},
        volume    = {abs/1801.07243},
        year      = {2018},
        url       = {http://arxiv.org/abs/1801.07243},
        archivePrefix = {arXiv},
        eprint    = {1801.07243},
        timestamp = {Mon, 13 Aug 2018 16:46:43 +0200},
        biburl    =
            {https://dblp.org/rec/bib/journals/corr/abs-1801-07243},
    }
    """

    NAME = 'ConvAI2'
    URL = 'http://parl.ai/downloads/convai2/convai2_fix_723.tgz'
    EXTRA_TOKENS = ['<p1-', '-p1>', '<p2-', '-p2>']

    @staticmethod
    def _get_dataset_files(use_candidates=True, **kwargs):
        files = ['train_self_original.txt', 'valid_self_original.txt', 'valid_self_original.txt']
        if use_candidates:
            return files
        else:
            return [f.replace('.txt', '_no_cands.txt') for f in files]

    @classmethod
    def _load_file(cls,
                   path: str,
                   sep: Optional[str] = '\t',
                   header: Optional[str] = None,
                   columns: Optional[Union[List[str], List[int]]] = None,
                   encoding: Optional[str] = 'utf-8') -> Tuple[List[Tuple], Optional[List[str]]]:
        """Load data from the given path."""
        with open(path, 'r') as file:
            # 1st: parse file into different conversations
            conversations = []
            conv_ = []
            for line in file:
                if line.startswith('1 ') and len(conv_) > 0:
                    # a '1' at the beginning of a line marks a new
                    # conversation
                    conversations.append(conv_)
                    conv_ = []
                # remove pre- and suffixes
                line = line.rstrip('\n')
                line = re.split(r'^[0-9]+', line)[-1]
                if 'your persona' in line:
                    # the persona sentences
                    persona = line.split(':')[-1]
                    conv_.append(('', persona, ()))
                else:
                    # the conversation sentences
                    other_, target_, cands_ = re.split(r'\t+', line)
                    cands_ = re.split('\|', cands_)
                    conv_.append((other_, target_, cands_))
            conversations.append(conv_)

        # converting the conversations into a "regular" dataset format
        data = []
        for conv in conversations:
            context = ''
            for utterance in conv:
                # add conversation markers to the utterances
                other_ = ' <p1- ' + utterance[0] + ' -p1> ' if utterance[0] != '' else ''
                target_ = ' <p2- ' + utterance[1] + ' -p2> '
                if utterance[2]:
                    # append next p1 utterance to the concext
                    context += other_
                    # create the candidate sentences
                    cands = list(utterance[2])
                    random.shuffle(cands)
                    # create labels
                    label = cands.index(utterance[1])
                    data.append((context, cands, label))
                    context += target_
                else:
                    # persona specs
                    context += target_
        return data, None


class DSCT7Dataset(CompressedDataTabularDataset):
    """Dialog System Technology Challenges 7 (DSTC 7)

    Please see
    https://github.com/IBM/dstc-noesis

    Please cite
    @InProceedings{dstc19task1,
        title     = {DSTC7 Task 1:
                     Noetic End-to-End Response Selection},
        author    = {Chulaka Gunasekara, Jonathan K. Kummerfeld,
                     Lazaros Polymenakos, and Walter S. Lasecki},
        year      = {2019},
        booktitle = {7th Edition of the Dialog System Technology
                     Challenges at AAAI 2019},
        url       = {http://workshop.colips.org/dstc7/papers/
                     dstc7_task1_final_report.pdf},
        month     = {January},
    }

    If using the Ubuntu subtask, please also cite
    @Article{arxiv18disentangle,
        author    = {Jonathan K. Kummerfeld, Sai R. Gouravajhala,
                     Joseph Peper, Vignesh Athreya, Chulaka Gunasekara,
                     Jatin Ganhotra, Siva Sankalp Patel,
                     Lazaros Polymenakos, and Walter S. Lasecki},
        title     = {Analyzing Assumptions in Conversation
                     Disentanglement Research Through the Lens
                     of a New Dataset and Model},
        journal   = {ArXiv e-prints},
        archivePrefix = {arXiv},
        eprint    = {1810.11118},
        primaryClass = {cs.CL},
        year      = {2018},
        month     = {October},
        url       = {https://arxiv.org/pdf/1810.11118.pdf},
    }
    """
    NAME = 'DSCT7'
    URL = 'http://parl.ai/downloads/dstc7/dstc7.tar.gz'
    EXTRA_TOKENS = ['<p1-', '-p1>', '<p2-', '-p2>', '<empty>']

    @staticmethod
    def _get_dataset_files(augmented=False, **kwargs):
        train_file = 'ubuntu_train_subtask_1_augmented.json' \
            if augmented \
            else 'ubuntu_train_subtask_1.json'
        dev_file = 'ubuntu_dev_subtask_1.json'
        # test_file = 'ubuntu_test_subtask_1.json'
        print(f'DSCT7: Using the {"augmented" if augmented else "normal"} '
              f'version of the train dataset.')
        return [train_file, dev_file, dev_file]

    @classmethod
    def _load_file(cls,
                   path: str,
                   sep: Optional[str] = '\t',
                   header: Optional[str] = None,
                   columns: Optional[Union[List[str], List[int]]] = None,
                   encoding: Optional[str] = 'utf-8') -> Tuple[List[Tuple], Optional[List[str]]]:
        """Load data from the given path."""
        data = []
        encountered_speakers = set()
        # loading dataset
        with open(path) as file:
            samples = json.load(file)
        # parsing samples
        for sample in samples:
            # creating context
            context = ''
            for msf in sample['messages-so-far']:
                spkr = msf['speaker'][-1]
                encountered_speakers.add(spkr)
                context += f' <p{spkr}- ' + msf['utterance'] + f' -p{spkr}> '
            # creating candidates
            candidates = [c['utterance'] if len(c['utterance']) > 0 else '<empty>'
                          for c in sample['options-for-next']]
            random.shuffle(candidates)
            # creating label
            target = sample['options-for-correct-answers'][0]['utterance']
            target = target if len(target) > 0 else '<empty>'
            label = candidates.index(target)
            data.append((context, candidates, label))

        return data, None


class UbuntuDataset(CompressedDataTabularDataset):
    """The Ubuntu dialog corpus

    Please see
    https://arxiv.org/abs/1506.08909

    Please cite
    @article{lowe2015ubuntu,
        title={The ubuntu dialogue_and_ranking corpus: A large dataset for research
        in unstructured multi-turn dialogue_and_ranking systems},
        author={Lowe, Ryan and Pow, Nissan and Serban, Iulian
        and Pineau, Joelle},
        journal={arXiv preprint arXiv:1506.08909},
        year={2015}
    }
    """
    NAME = 'Ubuntu'
    URL = 'http://parl.ai/downloads/ubuntu/ubuntu.tar.gz'
    EXTRA_TOKENS = ['__eot__', '__eou__']
    NAMED_COLS = ['text_1', 'text_2']

    @staticmethod
    def extra_tokens_():
        return ['__eot__', '__eou__']

    @staticmethod
    def _get_dataset_files(augmented=False, **kwargs):
        train_file = 'train.csv'
        dev_file = 'valid.csv'
        test_file = 'test.csv'
        return [train_file, dev_file, test_file]

    @classmethod
    def _load_file(cls,
                   path: str,
                   sep: Optional[str] = '\t',
                   header: Optional[str] = None,
                   columns: Optional[Union[List[str], List[int]]] = None,
                   encoding: Optional[str] = 'utf-8') -> Tuple[List[Tuple], Optional[List[str]]]:
        """Load data from the given path."""
        # loading dataset
        with open(path, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            # ignoring distractors in valid and test.
            # Negatives are drawn from batch.
            # first col is context, 2nd is response
            data = [(sample[0], sample[1]) for sample in reader]
        return data, None
