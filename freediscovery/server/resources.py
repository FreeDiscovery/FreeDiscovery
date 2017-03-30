# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from glob import glob
from textwrap import dedent

from flask import request
from webargs import fields as wfields
from flask_apispec import (marshal_with, use_kwargs as use_args,
                           MethodResource as Resource)
from flask_apispec.annotations import doc
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, \
                            adjusted_rand_score, adjusted_mutual_info_score,\
                            v_measure_score, average_precision_score
import warnings
from sklearn.metrics.base import UndefinedMetricWarning

from ..text import FeatureVectorizer
from ..parsers import EmailParser
from ..ingestion import _check_mutual_index
from ..lsi import _LSIWrapper
from ..categorization import _CategorizerWrapper
from ..utils import _docstring_description
from ..cluster import _ClusteringWrapper, centroid_similarity, compute_optimal_sampling
from ..search import _SearchWrapper
from ..metrics import (categorization_score,
                       ratio_duplicates_score, f1_same_duplicates_score,
                       mean_duplicates_count_score, _scale_cosine_similarity)
from ..dupdet import _DuplicateDetectionWrapper
from ..email_threading import _EmailThreadingWrapper
from ..datasets import load_dataset
from ..exceptions import WrongParameter
from ..stop_words import _StopWordsWrapper
from .validators import _is_in_range

from .schemas import (IDSchema, FeaturesParsSchema,
                      FeaturesSchema,
                      DocumentIndexNestedSchema,
                      EmailParserSchema, EmailParserElementIndexSchema,
                      ExampleDatasetSchema,
                      LsiParsSchema, LsiPostSchema,
                      ClassificationScoresSchema, _CategorizationInputSchema,
                      CategorizationParsSchema, CategorizationPostSchema,
                      CategorizationPredictSchema, ClusteringSchema,
                      _CategorizationIndex, _CategorizationPredictSchemaElement,
                      ErrorSchema,
                      MetricsCategorizationSchema, MetricsClusteringSchema,
                      MetricsDupDetectionSchema,
                      EmailThreadingSchema, EmailThreadingParsSchema,
                      SearchResponseSchema, DocumentIndexSchema,
                      EmptySchema,
                      CustomStopWordsSchema, CustomStopWordsLoadSchema
                      )

EPSILON = 1e-3 # small numeric value



# ============================================================================ #
#                         Datasets download                                    #
# ============================================================================ #

class ExampleDatasetApi(Resource):

    @use_args({'n_categories': wfields.Int(missing=2)})
    @doc(description=_docstring_description(dedent(load_dataset.__doc__)))
    @marshal_with(ExampleDatasetSchema())
    def get(self, name, **args):
        n_categories = args['n_categories']
        n_categories = min(3, n_categories)
        n_categories = max(1, n_categories)

        categories = None
        if "20newsgroups" in name:
            if n_categories == 3:
                categories = ['comp.graphics', 'rec.sport.baseball', 'sci.space']
            elif n_categories == 2:
                categories = ['comp.graphics', 'rec.sport.baseball']
            elif n_categories == 1:
                categories = ['comp.graphics']


        md, training_set, test_set = load_dataset(name, self._cache_dir, verbose=True,
                                                  verify_checksum=False,
                                                  categories=categories)

        return {'metadata': md, 'training_set': training_set, 'dataset': test_set}


# Definine the response formatting schemas
id_schema = IDSchema()
features_schema = FeaturesSchema()
error_schema = ErrorSchema()

# ============================================================================ #
#                      Feature extraction                                      #
# ============================================================================ #

class FeaturesApi(Resource):

    @doc(description='View parameters used for the feature extraction')
    @marshal_with(FeaturesSchema(many=True))
    def get(self):
        fe = FeatureVectorizer(self._cache_dir)
        return fe.list_datasets()

    @doc(description=dedent("""
            Initialize the feature extraction on a document collection.

            **Parameters**
             - `data_dir`: [optional] relative path to the directory with the input files. Either `data_dir` or `dataset_definition` must be provided.
             - `dataset_definition`: [optional] a list of dictionaries `[{'file_path': <str>, 'document_id': <int>, 'rendition_id': <int>}, ...]` where `document_id` and `rendition_id` are optional. Either `data_dir` or `dataset_definition` must be provided.
             - `n_features`: [optional] number of features (overlapping character/word n-grams that are hashed).  n_features refers to the number of buckets in the hash.  The larger the number, the fewer collisions.   (default: 1100000)
             - `analyzer`: 'word', 'char', 'char_wb' Whether the feature should be made of word or character n-grams.  Option ‘char_wb’ creates character n-grams only from text inside word boundaries.  ( default: 'word')
             - `ngram_range` : tuple (min_n, max_n), default=(1, 1) The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.

             - `stop_words`: "english" or "None" Remove stop words from the resulting tokens. Only applies for the "word" analyzer.  If "english", a built-in stop word list for English is used. ( default: "None")
             - `n_jobs`: The maximum number of concurrently running jobs (default: 1)
             - `norm`: The normalization to use after the feature weighting ('None', 'l1', 'l2') (default: 'l2')
             - `chuck_size`: The number of documents simultaneously processed by a running job (default: 5000)
             - `binary`: If set to 1, all non zero counts are set to 1. (default: False)
             - `use_idf`: Enable inverse-document-frequency reweighting (default: False).
             - `sublinear_tf`: Apply sublinear tf scaling, i.e. replace tf with log(1 + tf) (default: False).
             - `use_hashing`: Enable hashing. This option must be set to True for classification and set to False for clustering. (default: True)
             - `min_df`: When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is ignored when hashing is used.
             - `max_df`: When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold. This value is ignored when hashing is used.
            """))
    @use_args(FeaturesParsSchema(strict=True))
    @marshal_with(FeaturesSchema())
    def post(self, **args):
        args['use_idf'] = args['use_idf'] > 0
        if args['norm'] == 'None':
            args['norm'] = None
        if args['use_hashing']:
            for key in ['min_df', 'max_df']:
                if key in args:
                    del args[key] # the above parameters are ignored with caching
        for key in ['min_df', 'max_df']:
            if key in args and args[key] > 1. + EPSILON: # + eps
                args[key] = int(args[key])

        fe = FeatureVectorizer(self._cache_dir)
        dsid = fe.preprocess(**args)
        pars = fe.get_params()
        return {'id': dsid, 'filenames': pars['filenames']}


class FeaturesApiElement(Resource):
    @doc(description="Load extracted features (and obtain the processing status)")
    def get(self, dsid):
        sc = FeaturesSchema()
        fe = FeatureVectorizer(self._cache_dir, dsid=dsid)
        out = fe.get_params()
        is_processing = os.path.exists(os.path.join(fe.cache_dir, dsid, 'processing'))
        is_finished   = os.path.exists(os.path.join(fe.cache_dir, dsid, 'processing_finished'))
        if is_processing and not is_finished:
            n_chunks = len(glob(os.path.join(fe.cache_dir, dsid, 'features-*[0-9]')))
            out['n_samples_processed'] = min(n_chunks*out['chunk_size'], out['n_samples'])
            return sc.dump(out).data, 202
        elif not is_processing and is_finished:
            out['n_samples_processed'] = out['n_samples']
            return sc.dump(out).data, 200
        else:
            return error_schema.dump({"message": "Processing failed, see server logs!"}).data, 520

    @doc(description="Run feature extraction on a dataset")
    @marshal_with(IDSchema())
    def post(self, dsid):
        fe = FeatureVectorizer(self._cache_dir, dsid=dsid)
        dsid, _ = fe.transform()
        return {'id': dsid}

    @doc(description='Delete a processed dataset')
    @marshal_with(EmptySchema())
    def delete(self, dsid):
        fe = FeatureVectorizer(self._cache_dir, dsid=dsid)
        fe.delete()
        return {}

class FeaturesApiElementMappingNested(Resource):
    @doc(description='Compute correspondence between id fields for documents. '
           'At least one of the fields used for indexing must be provided,'
           'and all the rest will be computed (if available)')
    @use_args(DocumentIndexNestedSchema(strict=True))
    @marshal_with(DocumentIndexNestedSchema())
    def post(self, dsid, **args):
        fe = FeatureVectorizer(self._cache_dir, dsid=dsid)
        query = pd.DataFrame(args['data'])
        res = fe.db.search(query)
        res_repr = fe.db.render_dict(res, return_file_path=True)
        return {'data': res_repr}

# ============================================================================ #
#                   Email parser                                      #
# ============================================================================ #

class EmailParserApi(Resource):

    @doc(description='List processed datasets')
    def get(self):
        fe = EmailParser(self._cache_dir)
        return fe.list_datasets()

    @doc(description=dedent("""
           Load a dataset and parse emails

           Initialize the feature extraction on a document collection.

           **Parameters**
            - `data_dir`: [required] relative path to the directory with the input files
          """))
    @use_args({'data_dir': wfields.Str(required=True)})
    @marshal_with(EmailParserSchema())
    def post(self, **args):
        fe = EmailParser(self._cache_dir)
        dsid = fe.transform(**args)
        pars = fe.get_params()
        return {'id': dsid, 'filenames': pars['filenames']}


class EmailParserApiElement(Resource):
    @doc(description='Load parsed emails')
    def get(self, dsid):
        fe = EmailParser(self._cache_dir, dsid=dsid)
        out = fe.get_params()
        return out

    @marshal_with(EmptySchema())
    def delete(self, dsid):
        fe = EmailParser(self._cache_dir, dsid=dsid)
        fe.delete()
        return {}


class EmailParserApiElementIndex(Resource):
    @doc(description=dedent("""
           Query document index for a list of filenames

           **Parameters**
            - `filenames`: [required] list of filenames

          """))
    @use_args({'filenames': wfields.List(wfields.Str(), required=True)})
    @marshal_with(EmailParserElementIndexSchema())
    def post(self, dsid, **args):
        fe = EmailParser(self._cache_dir, dsid=dsid)
        idx = fe.search(args['filenames'])
        return {'index': list(idx)}

# ============================================================================ #
#                  LSI decomposition
# ============================================================================ #

_lsi_api_get_args  = {'parent_id': wfields.Str(required=True) }
_lsi_api_post_args = {'parent_id': wfields.Str(required=True),
                      'n_components': wfields.Int(missing=150) }
class LsiApi(Resource):

    @doc(description='List existing LSI models')
    @use_args(_lsi_api_get_args)
    @marshal_with(LsiParsSchema(many=True))
    def get(self, **args):
        parent_id = args['parent_id']
        lsi = _LSIWrapper(cache_dir=self._cache_dir, parent_id=parent_id)
        return lsi.list_models()

    @doc(description=dedent("""
           Build a Latent Semantic Indexing (LSI) model

           Recommended data ingestion options also include, `use_idf=1, sublinear_tf=0, binary=0`.

           The recommended value for the `n_components` (dimensions of the SVD decompositions) is
           in the [100, 200] range.

           **Parameters**
             - `n_components`: Desired dimensionality of the output data. Must be strictly less than the number of features.
             - `parent_id`: parent dataset identified by `dataset_id`
          """))
    @use_args(_lsi_api_post_args)
    @marshal_with(LsiPostSchema())
    def post(self, **args):
        parent_id = args['parent_id']
        del args['parent_id']
        lsi = _LSIWrapper(cache_dir=self._cache_dir, parent_id=parent_id)
        _, explained_variance = lsi.fit_transform(**args)
        return {'id': lsi.mid, 'explained_variance': explained_variance}


class LsiApiElement(Resource):

    @doc(description='Show Latent Semantic Indexing (LSI) model parameters')
    @marshal_with(LsiParsSchema())
    def get(self, mid):
        cat = _LSIWrapper(self._cache_dir, mid=mid)

        pars = cat._load_pars()
        pars['parent_id'] = pars['parent_id']
        return pars

    @doc(description='Delete a Latent Semantic Indexing (LSI) model')
    @marshal_with(EmptySchema())
    def delete(self, mid):
        cat = _LSIWrapper(self._cache_dir, mid=mid)
        cat.delete()
        return {}

# ============================================================================ #
#                  Categorization (ML)
# ============================================================================ #



class ModelsApi(Resource):
    @doc(description='List existing categorization models')
    @marshal_with(CategorizationParsSchema(many=True))
    def get(self, parent_id):
        cat = _CategorizerWrapper(parent_id, self._cache_dir)

        return cat.list_models()

    @doc(description=dedent("""
           Build the categorization ML model

           The option `use_hashing=True` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

           **Parameters**
            - `parent_id`: `dataset_id` or `lsi_id`
            - `data`: a list of dict which have a `category` field and one or several fields that can be used for indexing, such as `document_id` and optionally `rendition_id`.
            - `method`: classification algorithm to use (default: LogisticRegression),
              * "LogisticRegression": [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
              * "LinearSVC": [Linear SVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html),
              * "NearestNeighbor": nearest neighbor classifier (requires LSI)
              * "NearestCentroid": nearest centroid classifier (requires LSI)
              * "xgboost": [Gradient Boosting](https://xgboost.readthedocs.io/en/latest/model.html)
                   (*Warning:* for the moment xgboost is not istalled for a direct install on Windows)
            - `cv`: binary, if true optimal parameters of the ML model are determined by cross-validation over 5 stratified K-folds (default False).
            - `training_scores`: binary, compute the efficiency scores on the training dataset. This would make computations much slower for NearestNeighbors (default False). 
          """))
    @use_args(_CategorizationInputSchema())
    @marshal_with(CategorizationPostSchema())
    def post(self, **args):
        training_scores = args['training_scores']
        parent_id = args['parent_id']
        cat = _CategorizerWrapper(self._cache_dir, parent_id=parent_id)

        query = pd.DataFrame(args['data'])
        res_q = cat.fe.db.search(query, drop=False)
        del args['data']

        args['index'] = res_q.internal_id.values
        args['y'] = res_q.category.values

        if args['cv']:
            cv = 'fast'
        else:
            cv = None
        for key in ['parent_id', 'cv', 'training_scores']:
            del args[key]
        _, Y_train = cat.fit(cv=cv, **args)
        idx_train = args['index']
        res = {'id': cat.mid, 'training_scores': {}}
        if training_scores:
            Y_res, md = cat.predict()
            idx_res = np.arange(cat.fe.n_samples_, dtype='int')
            res['training_scores'] = categorization_score(idx_train, Y_train,
                                       idx_res, np.argmax(Y_res, axis=1))
        return res


class ModelsApiElement(Resource):
    @doc(description='Load categorization model parameters')
    @marshal_with(CategorizationParsSchema())
    def get(self, mid):
        cat = _CategorizerWrapper(self._cache_dir, mid=mid)
        pars = cat.get_params()
        return pars

    @doc(description='Delete the categorization model')
    @marshal_with(EmptySchema())
    def delete(self, mid):
        cat = _CategorizerWrapper(self._cache_dir, mid=mid)
        cat.delete()
        return {}


class ModelsApiPredict(Resource):

    @doc(description=dedent("""
            Predict document categorization with a previously trained model

            **Parameters**
             - `max_result_categories` : the maximum number of categories in the results
             - `sort_by` : if provided and not None, the field used for sorting results. Valid values are [None, 'score']
             - `sort_order`: the sort order (if applicable), one of ['ascending', 'descending']
             - `max_results` : return only the first `max_results` documents. If `max_results <= 0` all documents are returned.
             - `ml_output` : type of the output in ['decision_function', 'probability'], only affects ML methods.
             - `nn_metric` : The similarity returned by nearest neighbor classifier in ['cosine', 'jaccard', 'cosine_norm', 'jaccard_norm'].
             - `min_score` : filter out results below a similarity threashold
             - `subset`: apply prediction to a document subset. Must be one of ['all', 'train', 'test']. Default: 'test'.
            """))
    @use_args({'max_result_categories': wfields.Int(missing=1),
               'sort_by': wfields.Str(missing='score'),
               'sort_order': wfields.Str(missing='descending',
                                         validate=_is_in_range(['descending', 'ascending'])),
               'max_results': wfields.Int(),
               'ml_output': wfields.Str(missing='probability'),
               'nn_metric': wfields.Str(missing='jaccard_norm'),
               'min_score': wfields.Number(missing=-1),
               'subset': wfields.Str(missing='test',
                                     validate=_is_in_range(['all', 'train', 'test']))
               })
    @marshal_with(CategorizationPredictSchema())
    def get(self, mid, **args):

        sort_by = args.pop('sort_by')
        sort_reverse = args.pop('sort_order') == 'descending'
        max_result_categories  = args.pop('max_result_categories')
        min_score = args.pop("min_score")
        max_results = args.pop("max_results", 0)
        subset = args.pop("subset")

        cat = _CategorizerWrapper(self._cache_dir, mid=mid)
        y_res, nn_res = cat.predict(**args)
        train_indices = cat._pars['index']
        res = _CategorizerWrapper.to_dict(y_res, nn_res, cat.le.classes_,
                                          cat.fe.db.data,
                                          sort_by=sort_by,
                                          sort_reverse=sort_reverse,
                                          max_result_categories=max_result_categories,
                                          min_score=min_score,
                                          subset=subset,
                                          train_indices=train_indices)
        if max_results > 0:
            res['data'] = res['data'][:max_results]
        return res



# ============================================================================ #
#                              Clustering
# ============================================================================ #

_k_mean_clustering_api_post_args = {
        'parent_id': wfields.Str(required=True),
        'n_clusters': wfields.Int(missing=150),
        }


class KmeanClusteringApi(Resource):

    @doc(description=dedent("""
           Compute K-mean clustering

           The option `use_hashing=False` must be set for the feature extraction. Recommended options for feature extraction include, `use_idf=1, sublinear_tf=0, binary=0`.

           **Parameters**
            - `parent_id`: `dataset_id` or `lsi_id`
            - `n_clusters`: the number of clusters
           """))
    @use_args(_k_mean_clustering_api_post_args)
    @marshal_with(IDSchema())
    def post(self, **args):

        cl = _ClusteringWrapper(cache_dir=self._cache_dir, parent_id=args['parent_id'])

        del args['parent_id']

        labels = cl.k_means(**args)  # TODO unused variable. Remove?
        return {'id': cl.mid}




class BirchClusteringApi(Resource):

    @doc(description=dedent("""
           Compute birch clustering

           The option `use_hashing=False` must be set for the feature extraction. Recommended options for data ingestion also include, `use_idf=1, sublinear_tf=0, binary=0`.

           **Parameters**
            - `parent_id`: `dataset_id` or `lsi_id`
            - `n_clusters`: the number of clusters or -1 to use hierarchical clustering (default: -1)
            - `min_similarity`: The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold. Otherwise a new subcluster is started. See [sklearn.cluster.Birch](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html). Increasing this value would increase the hierarchical tree depth (and the number of clusters).
            - `branching_factor`: Maximum number of CF subclusters in each node. If a new samples enters such that the number of subclusters exceed the branching_factor then the node has to be split. The corresponding parent also has to be split and if the number of subclusters in the parent is greater than the branching factor, then it has to be split recursively. Decreasing this value would increase the number of clusters.
            - `nn_metric` : The similarity returned by nearest neighbor classifier in ['cosine', 'jaccard', 'cosine_norm', 'jaccard_norm'].
           """))
    @use_args( {
            'parent_id': wfields.Str(required=True),
            'n_clusters': wfields.Int(missing=-1),
            'branching_factor': wfields.Int(missing=20),
            'min_similarity': wfields.Number(missing=0.75), # this corresponds approximately to threashold = 0.5
            'nn_metric': wfields.Str(missing='jaccard_norm')
            }
            )
    @marshal_with(IDSchema())
    def post(self, **args):
        from math import sqrt
        

        S_cos = _scale_cosine_similarity(args.pop('min_similarity'),
                                         metric=args.pop('nn_metric'),
                                         inverse=True)
        # cosine sim to euclidean distance
        threshold = sqrt(2 *(1 - S_cos))

        cl = _ClusteringWrapper(cache_dir=self._cache_dir,
                                parent_id=args.pop('parent_id'))

        if args.get('n_clusters') <= 0:
            args['n_clusters'] = None

        cl.birch(threshold=threshold, **args)
        return {'id': cl.mid}


_wardhc_clustering_api_post_args = {
        'parent_id': wfields.Str(required=True),
        'n_clusters': wfields.Int(missing=150),
        'n_neighbors': wfields.Int(missing=5),
        }


class WardHCClusteringApi(Resource):

    @doc(description=dedent("""
           Compute Ward Hierarchical Clustering.

           The option `use_hashing=False` must be set for the feature extraction. Recommended options for data ingestion also include, `use_idf=1, sublinear_tf=0, binary=0`.

           The Ward Hierarchical clustering is generally slower that K-mean, and does not scale well with the dataset size. The run time can be reduced by decreasing the following parameters,

            - `lsi_components`: the number of dimensions used for the Latent Semantic Indexing decomposition (e.g. from 150 to 50)
            - `n_neighbors`:  the number of neighbors used to construct the connectivity (e.g. from 10 to 5)

           **Parameters**
            - `parent_id`: `dataset_id` or `lsi_id`
            - `n_clusters`: the number of clusters
            - `n_neighbors` Number of neighbors for each sample, used to compute the connectivity matrix (see [AgglomerativeClustering](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) and [kneighbors_graph](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html)

           """))
    @use_args(_wardhc_clustering_api_post_args)
    @marshal_with(IDSchema())
    def post(self, **args):

        cl = _ClusteringWrapper(cache_dir=self._cache_dir, parent_id=args['parent_id'])

        del args['parent_id']

        cl.ward_hc(**args)
        return {'id': cl.mid}



class DBSCANClusteringApi(Resource):

    @doc(description=dedent("""
           Compute clustering (DBSCAN)

           The option `use_hashing=False` must be set for the feature extraction. Recommended options for the data ingestion also include, `use_idf=1, sublinear_tf=0, binary=0`.

           **Parameters**
             - `parent_id`: `dataset_id` or `lsi_id`
             - `min_similarity`: The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold. Otherwise a new subcluster is started. See [sklearn.cluster.Birch](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html)
             - `nn_metric` : The similarity returned by nearest neighbor classifier in ['cosine', 'jaccard', 'cosine_norm', 'jaccard_norm'].
             - `min_samples`: (optional) int The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
            """))
    @use_args({ 'parent_id': wfields.Str(required=True),
                'min_samples': wfields.Int(missing=10),
                'min_similarity': wfields.Number(missing=0.75), # this corresponds approximately to threashold = 0.5
                'nn_metric': wfields.Str(missing='jaccard_norm')
                })
    @marshal_with(IDSchema())
    def post(self, **args):
        from math import sqrt
        S_cos = _scale_cosine_similarity(args.pop('min_similarity'),
                                         metric=args.pop('nn_metric'),
                                         inverse=True)
        # cosine sim to euclidean distance
        eps = sqrt(2 *(1 - S_cos))

        cl = _ClusteringWrapper(cache_dir=self._cache_dir, parent_id=args.pop('parent_id'))

        cl.dbscan(eps=eps, **args)
        return {'id': cl.mid}


class ClusteringApiElement(Resource):

    @doc(description=dedent("""
           Compute cluster labels

           **Parameters**
            - `n_top_words`: keep only most relevant `n_top_words` words
            - `nn_metric` : The similarity metric in ['cosine', 'jaccard', 'cosine_norm', 'jaccard_norm'].
            - `return_optimal_sampling` : Instead of cluster results, the optimal sampling results will be returned (with no cluster labels). This option is only valid with Birch algorithm. Note that optimal sampling cannot return more samples than the subclusters in the birch clustering results (default: false)
            - `sampling_min_similarity` : Similarity threashold used by smart sampling. Decreasing this value would result in more sampled documents. Default: 1.0 (i.e. use the full cluster hierarichy).
            - `sampling_min_coverage` : Minimal coverage requirement in [0, 1] range. Increasing this value would result in a larger number of samples. (default: 0.9)
            """))
    @use_args({ 'n_top_words': wfields.Int(missing=5),
               'nn_metric': wfields.Str(missing='jaccard_norm'),
               'return_optimal_sampling': wfields.Bool(missing=False),
               'sampling_min_similarity': wfields.Number(missing=1.0),
               'sampling_min_coverage': wfields.Number(missing=0.9),
          })
    @marshal_with(ClusteringSchema())
    def get(self, method, mid, **args):
        nn_metric = args.pop('nn_metric')
        return_optimal_sampling = args.pop('return_optimal_sampling')
        sampling_min_coverage = args.pop('sampling_min_coverage')
        sampling_min_similarity = args.pop('sampling_min_similarity')

        cl = _ClusteringWrapper(cache_dir=self._cache_dir, mid=mid)
        km = cl._load_model()

        if return_optimal_sampling and not cl._pars['is_hierarchical']:
            raise WrongParameter('Model {} does not support optimal sampling, please use Birch clustering'.format(type(km).__name__))

        cl._fit_X = cl.pipeline.data

        htree = cl._get_htree(cl._fit_X, metric=nn_metric)


        if type(km).__name__ == 'Birch' and cl._pars['is_hierarchical']:
            # Hierarchical clustering

            db = cl.fe.db.data

            if return_optimal_sampling:
                # cut the hierarchical tree to match the smart sampling
                flat_tree = compute_optimal_sampling(htree,
                                                     sampling_min_similarity,
                                                     sampling_min_coverage)
            else:
                # we don't use optimal sampling
                flat_tree = htree.flatten()

                terms = cl.compute_labels(cluster_indices=[row['children_document_id'] for row in flat_tree],
                                          **args)
                for label, row in zip(terms, flat_tree):
                    row['cluster_label'] = label

            res = []
            doc_keys = [key for key in db.columns \
                            if key in ['document_id', 'rendering_id']]
            db = db[doc_keys]

            for idx, row in enumerate(flat_tree):
                irow = {'cluster_similarity': row['cluster_similarity'],
                        'cluster_size': row['cluster_size'],
                        'cluster_id': idx}
                if not return_optimal_sampling:
                    irow['cluster_label'] = ' '.join(row['cluster_label'])
                    irow['children'] = [el['cluster_id'] for el in row.children]
                    irow['cluster_depth'] = row.depth

                db_sl = db.iloc[row['children_document_id']].copy()
                db_sl['similarity'] = row['cluster_similarity']
                tmp = []
                for index, row_tmp in db_sl.iterrows():
                    row_dict = row_tmp.to_dict()
                    tmp.append(row_dict)
                irow['documents'] = tmp

                res.append(irow)

        else:
            # Non hierarchical clustering

            

            if args['n_top_words'] > 0:
                terms = cl.compute_labels(**args)
            else:
                terms = None

            y = cl._merge_response(km.labels_)
            res = []
            valid_keys = ['document_id', 'rendering_id', 'similarity']
            for name, group in y.groupby('cluster_id'):
                name = int(name)

                S_sim_mean, S_sim = centroid_similarity(cl._fit_X, group.index.values, nn_metric)
                group = group.assign(similarity=S_sim)

                row_docs = []
                for idx, row in group.iterrows():
                    row_docs.append({key: val for key, val in row.to_dict().items() if key in valid_keys})
                row['documents'] = row_docs
                irow = {'documents': row_docs, 'cluster_id': int(name),
                            'cluster_similarity': S_sim_mean}
                irow['cluster_size'] = len(row_docs)
                if terms is not None:
                    irow['cluster_label'] = ' '.join(terms[name])
                res.append(irow)


        return {'data': res}


    @doc(description='Delete a clustering model')
    @marshal_with(EmptySchema())
    def delete(self, method, mid):  # TODO unused parameter 'method'
        cl = _ClusteringWrapper(cache_dir=self._cache_dir, mid=mid)
        cl.delete()
        return {}

# ============================================================================ #
#                              Duplicate detection
# ============================================================================ #

_dup_detection_api_post_args = {
        "parent_id": wfields.Str(required=True),
        "method": wfields.Str(required=False, missing='simhash')
        }


class DupDetectionApi(Resource):

    @doc(description=dedent("""
           Compute near duplicates

           **Parameters**
            - `parent_id`: `dataset_id` or `lsi_id`
            - `method`: str, default='simhash' Method used for duplicate detection. One of "simhash", "i-match"
          """))
    @use_args(_dup_detection_api_post_args)
    @marshal_with(IDSchema())
    def post(self, **args):

        model = _DuplicateDetectionWrapper(cache_dir=self._cache_dir,
                                           parent_id=args['parent_id'])

        del args['parent_id']


        model.fit(args['method'])

        return {'id': model.mid}

class DupDetectionApiElement(Resource):

    @doc(description=dedent("""
           Query duplicates

           **Parameters**
            - `distance` : int, default=2 Maximum number of differnet bits in the simhash (Simhash method only)
            - `n_rand_lexicons` : int, default=1 number of random lexicons used for duplicate detection (I-Match method only)
            - `rand_lexicon_ratio` : float, default=0.7 ratio of the vocabulary used in random lexicons (I-Match method only)
            - `nn_metric` : The similarity returned by nearest neighbor classifier in ['cosine', 'jaccard', 'cosine_norm', 'jaccard_norm'].
          """))
    @use_args({'distance': wfields.Int(),
               'n_rand_lexicons': wfields.Int(),
               'rand_lexicon_ratio': wfields.Number(),
               'nn_metric': wfields.Str(missing='jaccard_norm')
               })
    @marshal_with(ClusteringSchema())
    def get(self, mid, **args):

        nn_metric = args.pop('nn_metric')

        model = _DuplicateDetectionWrapper(cache_dir=self._cache_dir, mid=mid)
        cluster_id = model.query(**args)
        model._fit_X = model.pipeline.data # load the data
        y = model._merge_response(cluster_id)
        res = []
        valid_keys = ['document_id', 'rendering_id', 'similarity']
        for name, group in y.groupby('cluster_id'):
            if group.shape[0] <= 1:
                continue

            S_sim_mean, S_sim = centroid_similarity(model._fit_X, group.index.values, nn_metric)
            group = group.assign(similarity=S_sim)

            row_docs = []
            for idx, row in group.iterrows():
                row_docs.append({key: val for key, val in row.to_dict().items() if key in valid_keys})
            row['documents'] = row_docs
            res.append({'documents': row_docs, 'cluster_id': name,
                        'cluster_similarity': S_sim_mean})

        return {'data': res}


    @marshal_with(EmptySchema())
    def delete(self, mid):

        model = _DuplicateDetectionWrapper(cache_dir=self._cache_dir, mid=mid)
        model.delete()
        return {}



# ============================================================================ #
#                             Metrics                                          #
# ============================================================================ #

class MetricsCategorizationApiElement(Resource):
    @doc(description=dedent("""
          Compute categorization metrics to assess the quality
          of categorization.

          In the case of binary categrorization, category labels are sorted alphabetically
          and the second one is expected to be the positive one.

          **Parameters**
            - y_true: [required] ground truth categorization data
            - y_pred: [required] predicted categorization results
            - metrics: [required] list of str. Metrics to compute, any combination of "precision", "recall", "f1", "roc_auc"
          """))
    @use_args({'y_true': wfields.Nested(_CategorizationIndex, many=True, required=True),
               'y_pred': wfields.Nested(_CategorizationPredictSchemaElement, many=True, required=True),
               'metrics': wfields.List(wfields.Str())})
    @marshal_with(MetricsCategorizationSchema())
    def post(self, **args):
        from sklearn.preprocessing import LabelEncoder
        from ..metrics import recall_at_k_score
        output_metrics = {}
        y_true = pd.DataFrame(args['y_true'])

        y_pred_b = []
        for row in args['y_pred']:
            nrow = {'document_id': row['document_id'],
                    'category': row['scores'][0]['category'],
                    'score': row['scores'][0]['score']}
            y_pred_b.append(nrow)

        y_pred_b = pd.DataFrame(y_pred_b)
        index_cols = _check_mutual_index(y_true.columns, y_pred_b.columns)

        y_true = y_true.set_index(index_cols, verify_integrity=True)
        y_pred_b = y_pred_b.set_index(index_cols, verify_integrity=True)

        le = LabelEncoder()
        # this also sorts label by arithmetic order
        y_true['category_id'] = le.fit_transform(y_true.category.values)
        y_pred_b['category_id'] = le.transform(y_pred_b.category.values)

        y = y_true[['category_id']].merge(y_pred_b[['category_id', 'score']],
                                          how='inner',
                                          left_index=True,
                                          right_index=True,
                                          suffixes=('_true', '_pred'))
        if 'metrics' in args:
            metrics = args['metrics']
        else:
            metrics = ['precision', 'recall', 'roc_auc',
                       'f1', 'average_precision', 'recall_at_k']

        _binary_metrics = ['precision', 'recall', 'f1']

        cy_true = y.category_id_true.values
        cy_pred = y.category_id_pred.values

        n_classes = len(le.classes_)

        if n_classes == 2:
            cy_pred_score = y.score.values
            cy_pred_score[cy_pred == 0] *= -1

        # wrapping metrics calculations, as for example F1 score can frequently print warnings
        # "F-score is ill defined and being set to 0.0 due to no predicted samples"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            for func in [precision_score,
                         recall_score,
                         f1_score,
                         roc_auc_score,
                         average_precision_score,
                         recall_at_k_score]:
                name = func.__name__.replace('_score', '')
                opts = {}
                if name in ['roc_auc', 'average_precision', 'recall_at_k'] and n_classes == 2:
                    y_targ = cy_pred_score
                    if name == 'recall_at_k':
                        opts = {'k': 0.2}
                else:
                    y_targ = cy_pred

                if name in _binary_metrics and n_classes != 2:
                    opts['average'] = 'micro'
                if name in metrics:
                    if n_classes == 2 or name in _binary_metrics:
                        output_metrics[name] = func(cy_true, y_targ, **opts)
                    else:
                        output_metrics[name] = np.nan
            if "recall_at_k" in output_metrics:
                output_metrics['recall_at_20p'] = output_metrics.pop('recall_at_k')

        return output_metrics


_metrics_clustering_api_get_args  = {
    'labels_true': wfields.List(wfields.Int(), required=True),
    'labels_pred': wfields.List(wfields.Int(), required=True),
    'metrics': wfields.List(wfields.Str(), required=True)
}

class MetricsClusteringApiElement(Resource):
    @doc(description=dedent("""
          Compute clustering metrics to assess the quality
          of categorization, comparing the groud truth labels with the predicted ones.

          **Parameters**
            - labels_true: [required] list of int. Ground truth clustering labels
            - labels_pred: [required] list of int. Predicted clustering labels
            - metrics: [required] list of str. Metrics to compute, any combination of "adjusted_rand", "adjusted_mutual_info", "v_measure"
          """))
    @use_args(_metrics_clustering_api_get_args)
    @marshal_with(MetricsClusteringSchema())
    def post(self, **args):
        output_metrics = dict()
        labels_true = args['labels_true']
        labels_pred = args['labels_pred']
        metrics = args['metrics']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            if 'adjusted_rand' in metrics:
                output_metrics['adjusted_rand'] = adjusted_rand_score(labels_true, labels_pred)
            if 'adjusted_mutual_info' in metrics:
                output_metrics['adjusted_mutual_info'] = adjusted_mutual_info_score(labels_true, labels_pred)
            if 'v_measure' in metrics:
                output_metrics['v_measure'] = v_measure_score(labels_true, labels_pred)
        return output_metrics


class MetricsDupDetectionApiElement(Resource):
    @doc(description=dedent("""
          Compute duplicate detection metrics to assess the quality
          of categorization, comparing the groud truth labels with the predicted ones.

          **Parameters**
            - labels_true: [required] list of int. Ground truth clustering labels
            - labels_pred: [required] list of int. Predicted clustering labels
            - metrics: [required] list of str. Metrics to compute, any combination of "ratio_duplicates", "f1_same_duplicates", "mean_duplicates_count"
          """))
    @use_args(_metrics_clustering_api_get_args)  # Arguments are the same as for clustering
    @marshal_with(MetricsDupDetectionSchema())
    def post(self, **args):
        output_metrics = dict()
        labels_true = args['labels_true']
        labels_pred = args['labels_pred']
        metrics = args['metrics']

        # Methods 'ratio_duplicates_score' and 'f1_same_duplicates_score' in ..metrics.py
        # accept Numpy array objects, not standard Python lists
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            if 'ratio_duplicates' in metrics:
                output_metrics['ratio_duplicates'] = \
                   ratio_duplicates_score(np.array(labels_true), np.array(labels_pred))
            if 'f1_same_duplicates' in metrics:
                output_metrics['f1_same_duplicates'] = \
                   f1_same_duplicates_score(np.array(labels_true), np.array(labels_pred))
            if 'mean_duplicates_count' in metrics:
                output_metrics['mean_duplicates_count'] = \
                    mean_duplicates_count_score(labels_true, labels_pred)
        return output_metrics

# ============================================================================ #
#                              Email threading
# ============================================================================ #


class EmailThreadingApi(Resource):

    @doc(description='Compute email threading')
    @use_args({ "parent_id": wfields.Str(required=True)})
    @marshal_with(EmailThreadingSchema())
    def post(self, **args):

        model = _EmailThreadingWrapper(cache_dir=self._cache_dir, parent_id=args['parent_id'])

        tree =  model.thread()

        return {'data': [el.to_dict(include=['subject']) for el in tree],
                'id': model.mid}

class EmailThreadingApiElement(Resource):

    @doc(description='Get email threading parameters')
    @marshal_with(EmailThreadingParsSchema())
    def get(self, mid):

        model = _EmailThreadingWrapper(cache_dir=self._cache_dir, mid=mid)

        return model.get_params()

    @doc(description='Delete a processed dataset')
    @marshal_with(EmptySchema())
    def delete(self, mid):

        model = _EmailThreadingWrapper(cache_dir=self._cache_dir, mid=mid)
        model.delete()
        return {}


# ============================================================================ #
#                              (Semantic) search
# ============================================================================ #


class SearchApi(Resource):
    @doc(description=dedent("""
            Perform document search (if `parent_id` is a `dataset_id`) or a semantic search (if `parent_id` is a `lsi_id`).

            Parameters
            ----------
            - `parent_id` : the id of the previous processing step (either `dataset_id` or `lsi_id`)
            - `query` : the seach query. Either `query` or `query_document_id` must be provided.
            - `query_document_id` : the id of the document used as the search query. Either `query` or `query_document_id` must be provided.
            - `nn_metric` : The similarity returned by nearest neighbor classifier in ['cosine', 'jaccard', 'cosine_norm', 'jaccard_norm'].
            - `min_score` : filter out results below a similarity threashold
            - `max_results` : return only the first `max_results` documents. If `max_results <= 0` all documents are returned.
            - `sort_by` : if provided and not None, the field used for sorting results. Valid values are [None, 'score']
            - `sort_order`: the sort order (if applicable), one of ['ascending', 'descending']
            """))
    @use_args({ "parent_id": wfields.Str(required=True),
                "query": wfields.Str(),
                "query_document_id": wfields.Int(),
                'nn_metric': wfields.Str(missing='jaccard_norm'),
                'min_score': wfields.Number(missing=-1),
                'max_results': wfields.Int(),
                'sort_by': wfields.Str(missing='score'),
                'sort_order': wfields.Str(missing='descending',
                                          validate=_is_in_range(['descending', 'ascending'])),
                })
    @marshal_with(SearchResponseSchema())
    def post(self, **args):
        parent_id = args['parent_id']
        model = _SearchWrapper(cache_dir=self._cache_dir, parent_id=parent_id)

        if 'query' in args and 'query_document_id' not in args:
            query = args['query']
            scores = model.search(query, metric=args['nn_metric'])
        elif 'query' not in args and 'query_document_id' in args:
            query = pd.DataFrame([{'document_id': args['query_document_id']}])
            res_q = model.fe.db.search(query, drop=False)

            scores = model.search(None, internal_id=res_q.internal_id.values[0],
                                  metric=args['nn_metric'])
        else:
            raise WrongParameter("One of the 'query', 'query_document_id' must be provided")

        scores_pd = pd.DataFrame({'score': scores,
                                  'internal_id': np.arange(model.fe.n_samples_, dtype='int')})

        res = model.fe.db.render_dict(scores_pd)
        res = [row for row in res if row['score'] > args['min_score']]
        sort_by = args['sort_by']
        if sort_by:
            if sort_by not in res[0]:
                raise WrongParameter('sort_by={} not in []'.format(sort_by, list(res[0].keys())))
            sort_reverse = args['sort_order'] == 'descending'
            res = sorted(res, key=lambda row: row['score'], reverse=sort_reverse)
        if 'max_results' in args and args['max_results'] > 0:
            res = res[:args['max_results']]

        return {'data': res}


# ============================================================================ #
#                            Custom Stop Words
# ============================================================================ #


class CustomStopWordsApi(Resource):
    @doc(description="Store a list of custom stop words")
    @use_args({"name" : wfields.Str(required=True),
               "stop_words" : wfields.List(wfields.Str(), required=True)})
    @marshal_with(CustomStopWordsSchema())
    def post(self, **args):
        name = args['name']
        stop_words = args['stop_words']
        model = _StopWordsWrapper(cache_dir=self._cache_dir)
        model.save(name = name, stop_words = stop_words)
        return {'name': name}


class CustomStopWordsLoadApi(Resource):

    @doc(description="Load a stored list of stop words")
    @marshal_with(CustomStopWordsLoadSchema())
    def get(self, name):
        return {'name': name, 'stop_words': _StopWordsWrapper(cache_dir=self._cache_dir).load(name)}

    @doc(description='Delete a stored custom stop words')
    @marshal_with(EmptySchema())
    def delete(self, name):
        _StopWordsWrapper(cache_dir=self._cache_dir).delete(name)
        return {}
