import pytest

from .base import (parse_res, V01, app, app_notest,
                   get_features_cached)
from freediscovery.exceptions import WrongParameter

# ============================================================================#
#
#                     LSI
#
# ============================================================================#

def test_api_lsi(app):
    dsid, pars, _ = get_features_cached(app)
    method = V01 + "/feature-extraction/{}".format(dsid)
    res = app.get(method)
    assert res.status_code == 200
    data = parse_res(res)
    method = V01 + "/lsi/"
    res = app.get_check(method, data=dict(parent_id=dsid))

    lsi_pars = dict(n_components=101, parent_id=dsid)
    method = V01 + "/lsi/"
    data = app.post_check(method, json=lsi_pars)
    assert sorted(data.keys()) == ['explained_variance', 'id']
    lid = data['id']

    # checking again that we can load all the lsi models
    method = V01 + "/lsi/"
    data = app.get(method, data=dict(parent_id=dsid,))

    method = V01 + "/lsi/{}".format(lid)
    data = app.get_check(method)
    for key, vals in lsi_pars.items():
        assert vals == data[key]

    assert sorted(data.keys()) == sorted(["n_components", "parent_id"])

    for key in data.keys():
        assert data[key] == lsi_pars[key]

def test_api_lsi_custom_names(app):
    dsid, pars, _ = get_features_cached(app)
    mid_orig = 'tyds'
    lsi_pars = dict(n_components=2, parent_id=dsid, id=mid_orig)
    data = app.post_check(V01 + "/lsi/", json=lsi_pars)
    assert data['id'] == mid_orig
    method = V01 + "/lsi/{}".format(mid_orig)
    data = app.get_check(method)

    with pytest.raises(WrongParameter):
        data = app.post_check(V01 + "/lsi/", json=lsi_pars)

    with pytest.raises(WrongParameter):
        lsi_pars['id'] = '?+d'
        data = app.post_check(V01 + "/lsi/", json=lsi_pars)
