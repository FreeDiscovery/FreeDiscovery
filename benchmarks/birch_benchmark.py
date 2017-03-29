# -*- coding: utf-8 -*-

from __future__ import print_function

from time import time

import numpy as np

from sklearn.preprocessing import normalize
from freediscovery.externals.birch import Birch
from freediscovery.cluster.birch import _BirchHierarchy, _print_container

from memory_profiler import memory_usage


n_samples = 100000
n_lsi = 150

np.random.seed(999999)

X = np.random.randn(n_samples, n_lsi)
normalize(X, norm='l2', copy=False)

print('Dataset with {} LSI dim'.format(n_lsi))

for N in np.linspace(0, n_samples, 5).astype(int)[1:]:


    t0 = time()
    mod = Birch(n_clusters=None, threshold=0.8, compute_labels=False)

    X_sl = X[:N, :]

    mod.fit(X_sl)
    t1 = time()
    bhmod = _BirchHierarchy(mod)
    bhmod.fit(X_sl)
    htree = bhmod.htree
    t2 = time()
    mem_usage = np.max(memory_usage())
    del mod
    del htree

    print('Data set {:9} docs, {:5.3f} GB RAM,  t_total= {:.2f} s, t_fit = {:.2f}s, t_hc = {:.2f}s'.format(
                        N, 1e-3*(mem_usage), t2-t0, t1-t0, t2 - t1))

