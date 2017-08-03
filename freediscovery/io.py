# -*- coding: utf-8 -*-
import platform
import re

import pandas as pd


def parse_ground_truth_file(filename):
    """ Parse a ground truth file specified by a filename.
    Replace '/' by '\' when running in Windows """
    df = pd.read_csv(filename, sep='[\s\t]+',
                     names=['file_path', 'is_relevant'], engine='python')
    if platform.system() == 'Windows':
        df.file_path = df.file_path.map(lambda path: path.replace('/', '\\'))
    return df


def parse_smart_tokens(text):
    """
    Parse a dataset stored in the SMART tokenized format, used
    in particular for the RCV1-v2 dataset,
    http://www.jmlr.org/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm
    (cf. Appendix B.12.i.)

    Parameters
    ----------
    text : str
       the full text of the dataset

    Returns
    -------
    result : dict
       the parsed dataset in a OrderedDict, with document_ids as keys,
       and a string of tokens as values
    """

    data = None
    data_all = []
    data_key = None

    for line in text.splitlines():
        key_match = re.match('^\.(?P<key>[A-Z])\s?(?P<val>.*)', line)
        if key_match:
            data_key = key_match.group('key')
            if data_key == 'I':
                if data is not None:
                    data_all.append(data)
                data = {"I": int(key_match.group('val'))}
            else:
                data[data_key] = []
        else:
            if data_key is None:
                raise ValueError('Failed to parse index at the first line!')
            if data_key != 'I':
                data[data_key].append(line)

    if data is not None:
        data_all.append(data)

    for row in data_all:
        for key, val in row.items():
            if key != 'I':
                row[key] = ' '.join(val)

    df = pd.DataFrame(data_all)
    return df.set_index('I')
