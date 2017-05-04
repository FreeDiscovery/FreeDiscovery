# -*- coding: utf-8 -*-

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

    def limit_depth(self, max_depth=None):
        """ Truncate the tree to the provided maximum depth """
        if self.depth >= max_depth:
            self.children = []

        for el in self.children:
            el.limit_depth(max_depth)


def _check_birch_tree_consistency(node):
    """ Check that the _id we added is consistent """
    for el in node.subclusters_:
        if el.n_samples_ != len(el.id_):
            raise ValueError(('For subcluster ',
                             '{}, n_samples={} but len(id_)={}')
                             .format(el, el.n_samples_, el.id_))
        if el.child_ is not None:
            _check_birch_tree_consistency(el.child_)


class _BirchHierarchy(object):
    def __init__(self, model, metric='cosine'):
        self.model = model
        self.htree, _n_clusters = self._make_birch_hierarchy(model.root_)
        if len(self.htree._get_children_document_id()) != self.model.n_samples_:
            raise ValueError(("Building hierarchy failed: ",
                              "root node contains ",
                              "{} documents, while the total document number "
                              "is {}")
                             .format(
                             len(self.htree._get_children_document_id()),
                             self.model.n_samples_))
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
        htree['cluster_id'] = cluster_id

        document_id_list = htree['document_id']

        for el in node.subclusters_:
            if el.child_ is not None:
                cluster_id += 1
                subtree, cluster_id = _BirchHierarchy._make_birch_hierarchy(
                         el.child_, depth=depth+1, cluster_id=cluster_id)
                htree.add_child(subtree)
            else:
                document_id_list += el.id_
        if depth == 0:
            # make sure we return the correct number of clusters
            cluster_id += 1
        return htree, cluster_id

    def fit(self, X):
        """ Compute all the required parameters """

        for row in self.htree.flatten():
            document_id_lst = row._get_children_document_id()
            row['children_document_id'] = document_id_lst
            row['cluster_size'] = len(document_id_lst)
            inertia, S_sim = centroid_similarity(X, document_id_lst,
                                                 nn_metric=self.metric_)
            row['document_similarity'] = S_sim
            row['cluster_similarity'] = inertia


def _print_container(ctr, depth=0, debug=0):
    """Print summary of Thread to stdout."""
    if debug:
        message = repr(ctr) + ' ' + repr(ctr.message and ctr.message.subject)
    else:
        message = str(ctr['cluster_id']) + ' N_children: ' \
                       + str(len(ctr.children)) + ' N_docs: '\
                       + str(len(ctr['document_id']))

    print(''.join(['> ' * depth, message]))

    for child in ctr.children:
        _print_container(child, depth + 1, debug)
