# Authors: Roman Yurchak
#
# License: BSD 3 clause

from freediscovery.externals.jwzthreading import Container


class BirchSubcluster(Container):
    """A container class for BIRCH cluster hierarchy

    This is a dict like container, that links to other subclusters in the
    hierarchy with the following attributes,

     * `parent` : :class:`BirchSubcluster`, the parent
       container
     * `children` : ``list`` of :class:`BirchSubcluster`,
       contains the children subclusters

    Note
    ----
    This class descends from
    :class:`freediscovery.externals.jwzthreading.Container`
    originally used to represent e-mail threads obtained with the JWZ
    algorithm in
    `jwzthreading <https://github.com/FreeDiscovery/jwzthreading>`_,
    though it is general enough to represent other hierarchical
    stuctures (here BIRCH clustering).

    In FreeDiscovery this class is primarly used for documents. As a
    result the variables/methods containing the term "document"
    have the same meaning as "sample" in the general scikit-learn context.
    """

    @property
    def document_count(self):
        """Count of all documents in the children subclusters"""
        tmp_sum = sum([child.document_count for child in self.children])
        return len(self.get('document_id', [])) + tmp_sum

    def _get_children_document_id(self):
        res = list(self.get('document_id', []))
        for el in self.children:
            res += el._get_children_document_id()
        return res

    def limit_depth(self, max_depth=None):
        """ Truncate the tree to the provided maximum depth

        Parameters
        ----------
        max_depth : int
          hierarchy depth to which truncate the tree
        """
        if self.depth >= max_depth:
            self.children = []

        for el in self.children:
            el.limit_depth(max_depth)


def _check_birch_tree_consistency(node):
    """ Check that the _id we added is consistent """
    for el in node.subclusters_:
        if el.samples_id_ is None:
            raise ValueError('Birch was fitted without storing samples. '
                             'Please re-initalize Birch with '
                             'compute_sample_indices=True !')
        if el.n_samples_ != len(el.samples_id_):
            raise ValueError(('For subcluster ',
                             '{}, n_samples={} but len(id_)={}')
                             .format(el, el.n_samples_, el.samples_id_))
        if el.child_ is not None:
            _check_birch_tree_consistency(el.child_)


def _birch_hierarchy_constructor(node, depth=0, cluster_id=0,
                                 container=BirchSubcluster):
    """Wrap BIRCH cluster hierarchy with a container class

    Parameters
    ----------
    node : _CFNode
      a node in the birch hierarchy tree
    depth : int
      cluster depth
    cluster_id : int
      cluster id
    container : freediscovery.cluster.BirchSubcluster, default=BirchSubcluster
      a subclass of :class:`~freediscovery.cluster.BirchSubcluster`
      that will be used to wrap each BIRCH subcluster

    Returns
    -------
    res : BirchSubcluster
      a hierarchical structure with the resulting clustering
    n_subclusters : int
      the total number of subclusters
    """

    htree = container()
    htree['document_id'] = document_id_list = []
    htree['cluster_id'] = cluster_id

    for el in node.subclusters_:
        if el.child_ is not None:
            cluster_id += 1
            subtree, cluster_id = _birch_hierarchy_constructor(
                     el.child_, depth=depth+1, cluster_id=cluster_id)
            htree.add_child(subtree)
        else:
            document_id_list += el.samples_id_
    if depth == 0:
        # make sure we return the correct number of clusters
        cluster_id += 1
    return htree, cluster_id


def birch_hierarchy_wrapper(birch, container=BirchSubcluster, validate=True,
                            compute_document_id=True):
    """Wrap BIRCH cluster hierarchy with a container class

    This class as inpu

    Parameters
    ----------
    birch : freediscovery.cluster.Birch
      a trained Birch clustering
    container : freediscovery.cluster.BirchSubcluster, default=BirchSubcluster
      a subclass of :class:`~freediscovery.cluster.BirchSubcluster`
      that will be used to wrap each BIRCH subcluster
    validate : bool, default=True
      check the consistency of the constructed hierarchical tree
      (this may carry some overhead)
    compute_document_id : bool, default=True
      compute the document/sample ids belonging to each subcluster

    Returns
    -------
    htree : BirchSubcluster
      a container containing the data from the BIRCH cluster hierarchy
    n_subclusters : int
      the total number of subclusters

    Note
    ----
    in FreeDiscovery BIRCH if primairly used to cluster documents. As a
    result the variables/methods containing the term "document"
    have the same meaning as "sample" in the general scikit-learn context.
    """
    if validate:
        _check_birch_tree_consistency(birch.root_)

    htree, n_subclusters = _birch_hierarchy_constructor(birch.root_,
                                                        container=container)
    if validate:
        if len(htree._get_children_document_id()) != birch.n_samples_:
            raise ValueError(("Building hierarchy failed: root node contains ",
                              "{} documents, while the total document number "
                              "is {}")
                             .format(len(htree._get_children_document_id()),
                                     birch.n_samples_))
    if compute_document_id:
        for row in htree.flatten():
            document_id_lst = row._get_children_document_id()
            row['children_document_id'] = document_id_lst
            row['cluster_size'] = len(document_id_lst)
    return htree, n_subclusters


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
