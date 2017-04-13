Document similarity
-------------------

A number of algorithms in FreeDiscovery require computing similarity between documents, including Nearest Neighbors categorization, search and clustering. Four metrics (``cosine``, ``jaccard``, ``cosine_norm``, ``jaccard_norm``) are supported to compute the document similarity scores, specified by the `nn_metric` or `metric` input parameters. To illustrate the difference between these metrics on a more practical example, we can consider two documents,

- document ``A`` consisting of words ``"legal documents prodedure case"``
- document ``B`` consisting of words ``"legal documents"``


A more detailed description of different metrics can be found below,

- ``cosine`` metric computes to the `cosine similarity score <https://en.wikipedia.org/wiki/Cosine_similarity>`_. This metric is always internally used by FreeDiscovery. For vectorized documents with positive term frequencies the cosine similarity is in the ``[0, 1]`` range, however in general (in the LSI space and with hashed feature extraction) the domain of definition is ``[-1, 1]``. ``cosine_similarity(A, B) = 0.71``.
- ``jaccard`` metric, aims to scale the cosine similarity in a way to approximately match the results of `Jaccard similarity <https://en.wikipedia.org/wiki/Jaccard_index>`_,

  .. math:: S_jaccard = \frac{S_{cosine]}{2 - S_{cosine}}
   

  .. image:: ../_static/cosine_similarity_scaling.svg

  For positive vectors, the results are in the ``[0, 1]`` range, or in general in the ``[-1/3, 1]`` range. ``jaccard_similarity(A, B) = 0.54``.
  **Note:** the exact jaccard similarity in this case is ``0.5``. 

- ``cosine-positive`` is the equal to 
  .. math:: max(S_cosine, 0)
