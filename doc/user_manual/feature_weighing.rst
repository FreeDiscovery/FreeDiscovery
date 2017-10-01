Feature extraction
==================

For a general introduction to feature extraction with textual documents see the :ref:`scikit-learn documentation <text_feature_extraction>`.

.. _tfidf_section:

TF-IDF schemes
--------------

SMART TF-IDF schemes
^^^^^^^^^^^^^^^^^^^^

FreeDiscovery extends :class:`sklearn.feature_extraction.text.TfidfTransformer` with a larger number of TF-IDF weighting and normalization schemes in :class:`~freediscovery.feature_weighting.SmartTfidfTransformer`. It follows the `SMART Information Retrieval System <https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System>`_ notation,

  .. image:: ../_static/tf_idf_weighting.svg 

The different options are descibed in more detail in the table below,

+----------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| **Term frequency**                                                                                                                                       | **Document frequency**                                                                            | **Normalization**                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| **n** (natural): :math:`{{\text{tf}}_{t,d}}`                                                                                                             | **n** (no): 1                                                                                     | **n** (none): 1                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| **l** (logarithm): :math:`1+log({\displaystyle {\text{tf}}_{t,d}})`                                                                                      | **t** (idf): :math:`log{\displaystyle {\tfrac {N}{df_{t}}}}`                                      | **c** (cosine): :math:`{\displaystyle {\sqrt{\Sigma_ {t\epsilon d}{w_{t}^{2}}}}}`                 |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| **a** (augmented): :math:`0.5 + {\displaystyle {\tfrac {0.5\times {\text{tf}}_{t,d}}{{\text{max(tf}}_{t,d})}}}`                                          | **s** (smoothed idf):                                                                             | **l** (length): :math:`{\displaystyle  \Sigma_{t\epsilon d}{ |w_{t}| }}`                          |
|                                                                                                                                                          |                                                                                                   |                                                                                                   |
|                                                                                                                                                          | :math:`log{\displaystyle {\tfrac {N + 1}{df_{t } + 1}}}`                                          |                                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| **b** (boolean): :math:`{\displaystyle {\begin{cases}1,&{\text{if tf}}_{t,d}>0\\0,&{\text{otherwise}}\end{cases}}}`                                      | **p** (prob idf): :math:`{\displaystyle {\text{log}}{\tfrac {N-df_{t}}{df_{t}}}}`                 | **u** (unique): :math:`{\displaystyle  \Sigma_ {t\epsilon d} \textbf{bool}\left(|w_{t}|\right) }` |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| **L** (log average): :math:`{\displaystyle {\tfrac {1+{\text{log}}({\text{tf}}_{t,d})}{1+{\text{log}}({\text{avg}}_{t\epsilon d}({\text{tf}}_{t,d}))}}}` | **d** (smoothed prob idf): :math:`{\displaystyle {\text{log}}{\tfrac {N+1-df_{t}}{df_{t} + 1}}}`  |                                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+

Pivoted document length normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to standard TF-IDF normalizations above, pivoted normalization was proposed by Singal et al. as a way to avoid over-penalising long documents. It can be enabled with the ``weighting='???p'`` parameter. For each document the normalization term :math:`V_{\textbf{d}}` is replaced by,

.. math::
  
   {\displaystyle (1 - \alpha) \textbf{avg} \left( V_{\textbf{d}}\right)  + \alpha  V_{\textbf{d}}}

where :math:`\alpha` (``norm_alpha``) is a user defined parameter, such as :math:`\alpha \in [0, 1]`. If ``norm_alpha=1`` the pivot cancels out and this case corresponds to regular TF-IDF normalization.

See the example on :ref:`optimize_tfidf_scheme_example` for a more practical illustration.

.. admonition:: References

    *  C.D. Manning, P. Raghavan, H. Schütze.  `"Document and query weighting schemes"
       <https://nlp.stanford.edu/IR-book/html/htmledition/document-and-query-weighting-schemes-1.html>`_ , 2008.
    * A. Singhal, C. Buckley, and M. Mitra. `"Pivoted document length normalization."
      <https://ecommons.cornell.edu/bitstream/handle/1813/7217/95-1560.pdf?sequence=1>`_ 1996
