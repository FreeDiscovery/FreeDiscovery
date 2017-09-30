# -*- coding: utf-8 -*-
import os
import os.path
import shutil
import pytest


from freediscovery.engine.stop_words import _StopWordsWrapper
from freediscovery.tests.run_suite import check_cache


from freediscovery.stop_words import CUSTOM_STOP_WORDS as csw
from freediscovery.stop_words import COMMON_FIRST_NAMES as cfns

cache_dir = check_cache()

tested_stop_words = ['one', 'two', 'three', 'four', 'five', 'six']


@pytest.mark.parametrize('csw_name, csw_list', [
                        ('test_acstw', tested_stop_words),
                        ('common_first_names', cfns),
                        ('csw', csw)
                         ])
def test_custom_stop_words_param(csw_name, csw_list):
    """Test to save, retrieve and delete the custom stop words

       tested_stop_words - list of custom stop words,
       cfns - list of common first names imported from stop_words.py
              to use as a custom stop words in testing
    """

    # test to save and retrieve the custom stop words
    custom_sw = _StopWordsWrapper(cache_dir=cache_dir)
    custom_sw.save(name=csw_name, stop_words=csw_list)
    custom_stop_words = custom_sw.load(name=csw_name)
    assert(custom_stop_words == csw_list)
    assert(len(custom_stop_words) == len(csw_list))
    i = 0
    for word in custom_stop_words:
        assert word == csw_list[i]
        i += 1

    # test to delete the custom stop words
    custom_sw.delete(name=csw_name)
    with pytest.raises(FileNotFoundError):
        custom_sw.load(name=csw_name)

    # clearing from temporary testing data
    if os.path.isdir(os.path.join(cache_dir, 'stop_words')):
        shutil.rmtree(os.path.join(cache_dir, 'stop_words'), ignore_errors=True)
