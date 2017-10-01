# Authors: Roman Yurchak
#
# License: BSD 3 clause

import numpy as np


def _sample_tree(node, min_similarity):
    """Perform the first step of optimal sampling"""
    from itertools import chain

    if node['cluster_similarity'] > min_similarity or not node.children:
        return [node]
    else:
        return list(chain.from_iterable(
                    _sample_tree(child, min_similarity)
                    for child in node.children))


def compute_optimal_sampling(htree, min_similarity, min_coverage):
    """
    Given a Birch hierarchical tree, compute the optmal sampling,
    with the following steps

      1. For each node, walk through subclusters, and append those
         that have a cluster_similarity > min_similarity,
         recursively iterate through children until
         either this condition is reached or maximum depth is reached.
      2. Sort obtained subclusters by size, and select
         the first N veryfing the coverage requirement
      3. For each resulting subcluster, keep only the document closest
         to the centroid remove children, and the document.

    .. warning::
        This function does modify the original htree
    """
    if not (0 <= min_coverage <= 1):
        raise ValueError('min_coverage={} must be in [0, 1]'.format(
                         min_coverage))

    total_documents = htree['cluster_size']  # root node
    limit_size = total_documents*min_coverage
    # Step 1
    stree = _sample_tree(htree, min_similarity)

    # Step 2
    stree2 = []
    cum_size = 0
    for row in sorted(stree, key=lambda x: x['cluster_size'], reverse=True):
        cum_size += row['cluster_size']
        # Step 3:

        sampled_doc_idx = np.argmax(row['document_similarity'])
        row['document_similarity'] = [row['document_similarity'][sampled_doc_idx]]
        row['document_id_accumulated'] = [row['document_id_accumulated'][sampled_doc_idx]]

        stree2.append(row)
        if cum_size > limit_size:
            # required coverage reached
            break

    return stree2
