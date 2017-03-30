# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .base import (select_top_words, ClusterLabels,
                   _ClusteringWrapper)
from .utils import centroid_similarity
from .optimal_sampling import compute_optimal_sampling

from .dendrogram import _DendrogramChildren

