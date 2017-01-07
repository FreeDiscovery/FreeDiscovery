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
    df = pandas.read_csv(filename, sep='[\s\t]+', names=['filename', 'is_relevant'], index_col=0, engine='python')
    if platform.system() == 'Windows':
        df.index = df.index.map(lambda path: path.replace('/', '\\'))
    return df


def parse_smart_tokens(text):

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


