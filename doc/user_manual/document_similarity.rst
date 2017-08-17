Document similarity
===================


A number of algorithms in FreeDiscovery require computing similarity between documents, including Nearest Neighbors categorization, search and clustering. 

1. TF-IDF weighting schemes
---------------------------
FreeDiscovery supports a large range of TF-IDF weighting schemes via the `weighting="xxxx"` parameter, that follows the `SMART Information Retrieval System <https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System>`_ notation,

  .. image:: ../_static/tf_idf_weighting.svg 

the different options are descibed in more detail below,

+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| **Term frequency**                                                                                                                                       | **Document frequency**                                                            | **Normalization**                                                                                 |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| **n** (natural): :math:`{{\text{tf}}_{t,d}}`                                                                                                             | **n** (no): 1                                                                     | **n** (none): 1                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| **l** (logarithm): :math:`1+log({\displaystyle {\text{tf}}_{t,d}})`                                                                                      | **t** (idf): :math:`log{\displaystyle {\tfrac {N}{df_{t}}}}`                      | **c** (cosine): :math:`{\displaystyle {\sqrt{\Sigma_ {t\epsilon d}{w_{t}^{2}}}}}`                 |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| **a** (augmented): :math:`0.5 + {\displaystyle {\tfrac {0.5\times {\text{tf}}_{t,d}}{{\text{max(tf}}_{t,d})}}}`                                          | **s** (smoothed idf):                                                             | **l** (length): :math:`{\displaystyle  \Sigma_{t\epsilon d}{ |w_{t}| }}`                          |
|                                                                                                                                                          |                                                                                   |                                                                                                   |
|                                                                                                                                                          | :math:`log{\displaystyle {\tfrac {N + 1}{df_{t } + 1}}}`                          |                                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| **b** (boolean): :math:`{\displaystyle {\begin{cases}1,&{\text{if tf}}_{t,d}>0\\0,&{\text{otherwise}}\end{cases}}}`                                      | **p** (prob idf): :math:`{\displaystyle {\text{log}}{\tfrac {N-df_{t}}{df_{t}}}}` | **u** (unique): :math:`{\displaystyle  \Sigma_ {t\epsilon d} \textbf{bool}\left(|w_{t}|\right) }` |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| **L** (log average): :math:`{\displaystyle {\tfrac {1+{\text{log}}({\text{tf}}_{t,d})}{1+{\text{log}}({\text{avg}}_{t\epsilon d}({\text{tf}}_{t,d}))}}}` |                                                                                   |                                                                                                   |
+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+

when pivoted normalization is enabled i.e. ``weighting='???p'``, for each document it's norm :math:`V_{\textbf{d}}` is replaced by,

.. math::
  
   {\displaystyle (1 - \alpha) \textbf{avg} \left( V_{\textbf{d}}\right)  + \alpha  V_{\textbf{d}}}

where ``alpha`` is a user defined parameter, such as :math:`\alpha \in [0, 1]`.


2. Example of term weighting
----------------------------

To illustrate the difference between these metrics on a more practical example, we can consider two documents,

- document ``A`` consisting of words ``"legal documents prodedure case"``
- document ``B`` consisting of words ``"legal documents"``


