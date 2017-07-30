# -*- coding: utf-8 -*-

import os.path
from pathlib import Path
import pickle

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_array_equal

from freediscovery.utils import dict2type, sdict_keys
from .base import (parse_res, V01, app, data_dir,
                   CACHE_DIR)
from sklearn.externals import joblib


@pytest.mark.parametrize('preprocess', [[],
                                           ['emails_ignore_header']])
def test_preprocessing_email_headers(app, preprocess):
    method = V01 + "/feature-extraction/"
    data = app.post_check(method, json={"preprocess": preprocess})
    dsid = data['id']
    method += dsid
    app.post_check(method, json={'dataset_definition': [{'file_path': os.path.join(data_dir, '0.7.6.28635.txt')}]})


    with (Path(CACHE_DIR) / 'ediscovery_cache' / dsid / 'vectorizer').open('rb') as fh:
        vectorizer = pickle.load(fh)


    pars = app.get_check(method)
    assert pars['preprocess'] == preprocess


    vocabulary = list(vectorizer.vocabulary_.keys())
    if not preprocess:
        assert 'baumbach' in vocabulary # a name in the To: header
    else:
        assert 'baumbach' not in vocabulary # a name in the To: header
