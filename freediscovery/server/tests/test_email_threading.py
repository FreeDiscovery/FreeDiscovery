# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pytest
import json
import itertools
from unittest import SkipTest
from numpy.testing import assert_equal, assert_almost_equal

from .. import fd_app
from ...utils import _silent, dict2type, sdict_keys
from ...ingestion import DocumentIndex
from ...exceptions import OptionalDependencyMissing
from ...tests.run_suite import check_cache

from .base import (parse_res, V01, app, app_notest, get_features, get_features_lsi,
                   email_data_dir)

#=============================================================================#
#
#                     Email Parsing
#
#=============================================================================#

def parse_emails(app):
    method = V01 + "/email-parser/"
    pars = dict(data_dir=email_data_dir)

    data = app.post_check(method, json=pars)

    assert sorted(data.keys()) ==  ['filenames', 'id']
    dsid = data['id']

    return dsid, pars

def test_parse_emails(app):
    dsid, pars = parse_emails(app)

    method = V01 + "/email-parser/{}".format(dsid)
    data = app.get_check(method)
    for key, val in pars.items():
        if key in ['data_dir']:
            continue
        assert val == data[key]


def test_delete_parsed_emails(app):
    dsid, _ = parse_emails(app)

    method = V01 + "/email-parser/{}".format(dsid)
    app.delete_check(method)


def test_get_email_parser_all(app):
    method = V01 + "/email-parser/"
    data = app.get_check(method)
    for row in data:
        assert sorted(row.keys()) == sorted([ 'data_dir', 'id', 'encoding', 'n_samples']) 


def test_get_email_parser(app):
    dsid, _ = parse_emails(app)
    method = V01 + "/email-parser/{}".format(dsid)
    data = app.get_check(method)
    assert sorted(data.keys()) == \
             sorted(['data_dir', 'filenames', 'encoding', 'n_samples', 'type'])


def test_get_search_emails_by_filename(app):
    dsid, _ = parse_emails(app)

    method = V01 + "/email-parser/{}/index".format(dsid)
    for pars, indices in [
            ({ 'filenames': ['1', '2']}, [0, 1]),
            ({ 'filenames': ['5']}, [4])]:

        data = app.post_check(method, json=pars)
        assert sorted(data.keys()) ==  sorted(['index'])
        assert_equal(data['index'], indices)



#=============================================================================#
#
#                     Email Threading
#
#=============================================================================#


def test_api_thread_emails(app):

    dsid, _ = parse_emails(app)

    method = V01 + "/email-parser/{}".format(dsid)
    data = app.get_check(method)

    url = V01 + "/email-threading" 
    pars = { 'parent_id': dsid }

    data = app.post_check(url, json=pars)
    assert sorted(data.keys()) == sorted(['data', 'id'])
    mid = data['id']

    tree_ref = [ {'id': 0, 'parent': None, 'children': [
                  {'id': 1, 'children': [], 'parent': 0},
                  {'id': 2, 'parent': 0,  'children': [
                         {'id': 3, 'children': [], 'parent': 2},
                         {'id': 4, 'children': [], 'parent': 2}],
                         }]
                  }]

    def remove_subject_field(d):
        del d['subject']
        for el in d['children']:
            remove_subject_field(el)

    tree_res = data['data']
    for el in tree_res:
        remove_subject_field(el)

    assert data['data'] == tree_ref

    url += '/{}'.format(mid)
    data = app.get_check(url)
    assert sorted(data.keys()) == sorted(['group_by_subject'])

    app.delete_check(method)
