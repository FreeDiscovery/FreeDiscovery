# Frequently Asked Questions



## 1. Categorization (ML or LSI)
### 1.a Feature extraction

The option `use_hashing=True` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

### 1.b Machine learning categorization methods

The cross validated logistic regression (with `cv=1`) is the advised method for use in production. 

The `xgboost-0.4` library cannot currently be easily installed on Windows, and the corresponding categorization method are disabled on this OS.

### 1.c LSI categorization methods

The recommended value for the `n_components` (dimensions of the SVD decompositions) is in the [100, 200] range.

## 2. Clustering 

### 2.a Feature extraction

The option `use_hashing=False` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

### 2.b Hierarchical clustering

The Ward Hierarchical clustering is generally slower that K-mean, however the run time can be reduced by decreasing the following parameters,

   - `lsi_components`: the number of dimensions used for the Latent Semantic Indexing decomposition (e.g. from 150 to 50)
   - `n_neighbors`:  the number of neighbors used to construct the connectivity (e.g. from 10 to 5)
