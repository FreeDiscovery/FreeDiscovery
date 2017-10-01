"""
.. _exploring_birch_example:

Exploring BIRCH cluster hierarchy
=================================

An example illustrating how to explore the cluster hierarchy
computed by the BIRCH algorithm.

In this example, we use a
:ref:`patched verson <birch_section>` of :class:`sklearn.cluster.Birch`
that allows to store the id of samples belonging to each subcluster.
This modified version is available from :class:`freediscovery.cluster.Birch`.

Building the cluster hierarchy
------------------------------

We start by computing BIRCH clustering on some random structured data,
"""

import numpy as np
from sklearn.datasets import make_blobs
from freediscovery.cluster import Birch, birch_hierarchy_wrapper

rng = np.random.RandomState(42)

X, y = make_blobs(n_samples=1000, n_features=10, random_state=rng)


cluster_model = Birch(threshold=0.9, branching_factor=20,
                      compute_sample_indices=True)
cluster_model.fit(X)

###############################################################################
#
# Next we wrap each subcluster in the cluster hierarchy
# (``cluster_model.root_``) with the
# :class:`~freediscovery.cluster.BirchSubcluster` class
# that allows easier manipulation of the hierarchical tree.

htree, _ = birch_hierarchy_wrapper(cluster_model)
print('Total number of subclusters:', htree.tree_size)


###############################################################################
#
# Visualizing the hierarchy
# -------------------------
# We can now visualize the cluster hierarchy,

htree.display_tree()

###############################################################################
#
# We have a hierarchy 2 levels deep, with 78 sub-clusters and a total
# of 1000 samples.
#
# For instance, let's consider the subcluster with  ``cluster_id=34``.
# We can access it by id with the flattened representation of the hierarchy,

sc = htree.flatten()[34]
print(sc)

###############################################################################
#
# Each subcluster is a dictionary linked inside the hierarchy via the
# `parent` / `children` attributes
# (cf documentation of :class:`~freediscovery.cluster.BirchSubcluster`).
# The ids of the samples contained in a subcluster are stored under the
# ``document_id_accumulated`` key. We can perform any calculations with the
# samples in a given cluster by indexing the original dataset `X`,

print('cluster_id', sc['cluster_id'])
print('document_id_accumulated', sc['document_id_accumulated'])
sc_centroid = X[sc['document_id_accumulated'], :].mean(axis=0)
print(sc_centroid)

###############################################################################
#
# For instance, we can select only subclusters that are one level deep
# (this includes ``cluster_id=34``) and compute their centroids,

htree_depth_1 = [sc for sc in htree.flatten() if sc.current_depth == 1]

for sc in htree_depth_1:
    sc['centroid'] = X[sc['document_id_accumulated'], :].mean(axis=0)

print('Centroid for cluster_id=34:\n', htree.flatten()[34]['centroid'])


###############################################################################
#
# Custom calculations in the hierarchy
# ------------------------------------
#
# While for a number of computations, it is sufficient to iterate through
# a flattened tree, sometimes the output of the calculation need to account
# for data from any number of other subclusters in the tree (e.g. all the
# children). In this case we can subclass
# :class:`~freediscovery.cluster.BirchSubcluster` to add out custom recursive
# function. Here we will add a function that for any subcluster computes the
# the maximum depth of the tree spanned by its children subclusters,

from freediscovery.cluster import BirchSubcluster


class NewBirchSubcluster(BirchSubcluster):

    @property
    def max_tree_depth(self):
        if self.children:
            return 1 + max(sc.max_tree_depth for sc in self.children)
        else:
            return 0


###############################################################################
#
# by re-wrapping the cluster hierarchy with this container, we get,

htree_new, _ = birch_hierarchy_wrapper(cluster_model,
                                       container=NewBirchSubcluster)

print('Tree depth from the root node:', htree_new.max_tree_depth)

print('Tree depth from cluster_id=34:', htree_new.flatten()[34].max_tree_depth)
