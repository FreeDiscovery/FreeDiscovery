# -*- coding: utf-8 -*-

from numpy.testing import assert_equal, assert_almost_equal

from .base import (parse_res, V01, app, app_notest, get_features, get_features_lsi,
                   email_data_dir)


#=============================================================================#
#
#                     Email Threading
#
#=============================================================================#


def test_api_thread_emails(app):

    dsid, pars, _ = get_features(app, parse_email_headers=True,
                                 dataset='fedora_ml_3k_subset')

    url = V01 + "/email-threading"
    pars = {'parent_id': dsid}

    data = app.post_check(url, json=pars)
    assert sorted(data.keys()) == sorted(['data', 'id'])
    mid = data['id']

    tree_ref = [{'id': 0, 'parent': None, 'children': [
                 {'id': 1, 'children': [], 'parent': 0},
                 {'id': 2, 'parent': 0,  'children': [
                        {'id': 3, 'children': [], 'parent': 2},
                        {'id': 4, 'children': [], 'parent': 2}]}]
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

    app.delete_check(url)
