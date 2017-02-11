#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import shutil

from unittest import SkipTest
from freediscovery.stop_words import (_StopWordsWrapper)


basename = os.path.dirname(__file__)
current_point = os.getcwd()

if os.name == 'nt': # on windows
    cache_dir = os.getcwd() + '\\acsw\\'
else:
    cache_dir = os.getcwd() + '/acsw/'

custom_sw = _StopWordsWrapper(cache_dir = cache_dir)

stop_words_for_test = ['one', 'two', 'three']

def test_custom_stop_words():
    """Test to save and retrive the stop_words list of strings"""
    custom_sw.save(name = 'test_acstw', stop_words = stop_words_for_test)
    if os.name == 'nt':
        assert(os.path.isfile(cache_dir + '\\stop_words\\test_acstw.pkl') == True)
    else:
        assert(os.path.isfile(cache_dir + '/stop_words/test_acstw.pkl') == True)
    if os.name == 'nt':
        with open(cache_dir + '\\stop_words\\test_acstw.pkl', 'r') as acsw_file:
            sw_list = acsw_file.read().split('\n')[:-1]
            assert(sw_list == stop_words_for_test)
    else:
        with open(cache_dir + '/stop_words/test_acstw.pkl', 'r') as acsw_file:
            sw_list = acsw_file.read().split('\n')[:-1]
            assert(sw_list == stop_words_for_test)
    # clearing from temporary testing data
    os.chdir(current_point)
    if os.name == 'nt':
        if os.path.isfile(os.getcwd() + '\\acsw\\stop_words\\test_acstw.pkl'):
            shutil.rmtree(os.getcwd() + '\\acsw\\', ignore_errors=True, onerror=None)
    else:
        if os.path.isfile(os.getcwd() + '/acsw/stop_words/test_acstw.pkl'):
            shutil.rmtree(os.getcwd() + '/acsw/', ignore_errors=True, onerror=None)
