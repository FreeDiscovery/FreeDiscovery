
def _document_length_correction_full(doc_len, epsilon, alpha):
    """An additive correction for the document lenght to account for
    document lenght normalization.

    Warning: this function is experimental and shouldn't be used in
             production.

    Parameters
    ----------
    doc_len : ndarray
      document lenght array
    epsilon : float
      approximate value of the correction for doc_len == 0 and
      doc_len == + \inf
    alpha : float
      how much to penalize documents that are far away from the
      average document in length. alpha=1.0 no correction at all,
      alpha=0 correction by a constant value.
    """
    pivot = doc_len.mean()
    return epsilon*(1 - 1/(alpha + (1 - alpha)*(doc_len / pivot)))


def _document_length_correction_diff(doc_len, epsilon, alpha):
    """An additive correction for the document lenght to account for
    document lenght normalization.

    Warning: this function is experimental and shouldn't be used in
             production.

    Parameters
    ----------
    doc_len : ndarray
      document lenght array
    epsilon : float
      approximate value of the correction for doc_len == 0 and
      doc_len == + \inf
    alpha : float
      how much to penalize documents that are far away from the
      average document in length. alpha=1.0 no correction at all,
      alpha=0 correction by a constant value.
    """
    pivot = doc_len.mean()
    return epsilon*(1/(pivot*alpha + (1 - alpha)*doc_len) - 1/doc_len)
