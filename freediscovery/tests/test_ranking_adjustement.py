import pytest
import numpy as np
from numpy.testing import assert_allclose

from ..ranking_adjustement import (_document_length_correction_full,
                                   _document_length_correction_diff)



@pytest.mark.parametrize("func", [_document_length_correction_full,
                                  _document_length_correction_diff])
def test_document_length_correction(func):
    np.random.seed(42)
    document_len = np.fmax(1, np.random.normal(1000, scale=3000, size=(10000)).astype('int'))
    document_len_mean = document_len.mean()
    document_len = np.concatenate((document_len,
                                   np.array([document_len_mean])))

    epsilon = 0.1
    res = func(document_len, epsilon, 1.0)
    if func.__name__ == '_document_length_correction_full':
        assert_allclose(res, 0, atol=0.01)

    # correction at l=0 (sort documents)
    res = func(document_len, epsilon, 0.7)
    idx_min = np.argmin(document_len)
    assert res[idx_min] < 0  # negative correction for small documents
    if func.__name__ == '_document_length_correction_full':
        assert_allclose(res[idx_min], -epsilon, rtol=0.65)

    # the pivot value (mean) returns zero deviation
    assert_allclose(res[-1], 0, atol=1e-5)
    # asymptotic value at infinity
    idx_max = np.argmax(document_len)
    if func.__name__ == '_document_length_correction_diff':
        assert_allclose(res[idx_max], 0, atol=1e-4)
    assert res[idx_max] > 0  # positive correction for very large documents

    # correction bound by epsilon
    assert -epsilon < res.min()
    assert res.max() < epsilon
