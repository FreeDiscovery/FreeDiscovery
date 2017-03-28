# -*- coding: utf-8 -*-

import numpy as np

from freediscovery.cluster.utils import centroid_similarity
from freediscovery.externals.jwzthreading import Container

class _HCContainer(Container):

    @property
    def document_count(self):
        """ Return the count of all documents used in the tree"""
        tmp_sum = sum([child.document_count for child in self.children])
        return len(self.get('document_id', [])) + tmp_sum

    def _get_children_document_id(self):
        res = list(self.get('document_id', []))
        for el in self.children:
            res += el._get_children_document_id()
        return res

class _BirchHierarchy(object):
    def __init__(self, model, metric='jaccard_norm'):
        self.model = model
        self.htree, _n_clusters = self._make_birch_hierarchy(model.root_)
        self._n_clusters = _n_clusters
        self.metric_ = metric


    @staticmethod
    def _make_birch_hierarchy(node, depth=0, cluster_id=0):
        """Construct a cluster hierarchy using a trained Birch model

        Parameters
        ----------
        model : Birch
          a trained Birch clustering

        Returns
        -------
        res : a jwzthreading.Container object
          a hierarchical structure with the resulting clustering
        """

        htree = _HCContainer()
        htree['document_id'] = []
        htree['cluster_id'] =  cluster_id

        document_id_list = htree['document_id']

        for el in node.subclusters_:
            if el.child_ is not None:
                cluster_id += 1
                subtree, cluster_id = _BirchHierarchy._make_birch_hierarchy(el.child_, depth=depth+1, cluster_id=cluster_id)
                htree.add_child(subtree)
            else:
                document_id_list.append(el.id_)
        if depth == 0:
            #make sure we return the correct number of clusters
            cluster_id += 1
        return htree, cluster_id


    def fit(self, X):
        """ Compute all the required parameters """

        for row in self.htree.flatten():
            document_id_lst = row._get_children_document_id()
            row['children_document_id'] = document_id_lst
            inertia, S_sim = centroid_similarity(X, document_id_lst, nn_metric=self.metric_)
            row['document_similarity'] = S_sim
            row['cluster_similarity'] = inertia


def _print_container(ctr, depth=0, debug=0):
    """Print summary of Thread to stdout."""
    if debug:
        message = repr(ctr) + ' ' + repr(ctr.message and ctr.message.subject)
    else:
        message = str(ctr['cluster_id']) + ' N_children: ' \
		      + str(len(ctr.children)) + ' N_docs: ' + str(len(ctr['document_id']))


    print(''.join(['> ' * depth, message]))

    for child in ctr.children:
        _print_container(child, depth + 1, debug)
