# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pandas
import platform
from collections import OrderedDict


def parse_ground_truth_file(filename):
    """ Parse a ground truth file specified by a filename.
    Replace '/' by '\' when running in Windows """
    df = pandas.read_csv(filename, sep='[\s\t]+', names=['file_path', 'is_relevant'], engine='python')
    if platform.system() == 'Windows':
        df.file_path = df.file_path.map(lambda path: path.replace('/', '\\'))
    return df


def parse_rcv1_smart_tokens(text):
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

    res = OrderedDict()

    docid = None
    document_text = []

    for line in text.splitlines():
        if line.startswith('.I'):
            if docid is not None:
                res[docid] = document_text
            document_text = []
            _, docid = line.split(' ')
        elif line.startswith('.W') or not line:
            pass
        else:
            document_text += line.split(' ')

    if document_text and docid is not None:
        res[docid] = document_text

    return res


