# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from glob import glob
from textwrap import dedent

#from flask_restful import Resource
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
try:  # sklearn v0.17
    from sklearn.exceptions import UndefinedMetricWarning
except:  # sklearn v0.18
    from sklearn.metrics.base import UndefinedMetricWarning

from ..text import FeatureVectorizer
from ..parsers import EmailParser
from ..lsi import _LSIWrapper
from ..categorization import _CategorizerWrapper
from ..io import parse_ground_truth_file
from ..utils import categorization_score, _docstring_description
from ..cluster import _ClusteringWrapper
from ..search import _SearchWrapper
from ..metrics import ratio_duplicates_score, f1_same_duplicates_score, mean_duplicates_count_score
from ..dupdet import _DuplicateDetectionWrapper
from ..email_threading import _EmailThreadingWrapper
from ..datasets import load_dataset
from ..exceptions import WrongParameter
from .schemas import (IDSchema, FeaturesParsSchema,
                      FeaturesSchema, DocumentIndexListSchema,
                      DocumentIndexNestedSchema,
                      EmailParserSchema, EmailParserElementIndexSchema,
                      DatasetSchema,
                      LsiParsSchema, LsiPostSchema,
                      ClassificationScoresSchema, _CategorizationInputSchema,
                      CategorizationParsSchema, CategorizationPostSchema,
                      CategorizationPredictSchema, ClusteringSchema,
                      ErrorSchema, DuplicateDetectionSchema,
                      MetricsCategorizationSchema, MetricsClusteringSchema,
                      MetricsDupDetectionSchema,
                      EmailThreadingSchema, EmailThreadingParsSchema,
                      ErrorSchema, DuplicateDetectionSchema,
                      SearchResponseSchema, DocumentIndexSchema,
                      EmptySchema
                      )

EPSILON = 1e-3 # small numeric value



# ============================================================================ #
#                         Datasets download                                    #
# ============================================================================ # 

class DatasetsApiElement(Resource):

    @doc(description=_docstring_description(dedent(load_dataset.__doc__)))
    @use_args({'return_file_path': wfields.Boolean()})
    @marshal_with(DatasetSchema())
    def get(self, name, **args):
        res = load_dataset(name, self._cache_dir, verbose=True,
                load_ground_truth=True, verify_checksum=False, **args)
        return res


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
             - `n_features`: [optional] number of features (overlapping character/word n-grams that are hashed).
                             n_features refers to the number of buckets in the hash.  The larger the number, the fewer collisions.   (default: 1100000)
             - `analyzer`: 'word', 'char', 'char_wb'
                           Whether the feature should be made of word or character n-grams.
                           Option ‘char_wb’ creates character n-grams only from text inside word boundaries.  ( default: 'word')
             - `ngram_range` : tuple (min_n, max_n), default=(1, 1)
                           The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.

             - `stop_words`: "english" or "None"
                             Remove stop words from the resulting tokens. Only applies for the "word" analyzer.
                             If "english", a built-in stop word list for English is used. ( default: "None")
             - `n_jobs`: The maximum number of concurrently running jobs (default: 1)
             - `norm`: The normalization to use after the feature weighting ('None', 'l1', 'l2') (default: 'None')
             - `chuck_size`: The number of documents simultaneously processed by a running job (default: 5000)
             - `binary`: If set to 1, all non zero counts are set to 1. (default: True)
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

        if 'dataset_defintion' in args:
            print(args['dataset_definition'])

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

class _DocumentIndexListSchemaInput(DocumentIndexListSchema):
    return_file_path = wfields.Boolean()

class FeaturesApiElementMappingFlat(Resource):
    @doc(description='Compute correspondence between id fields (flat). '
           'At least one of the fields used for indexing must be provided,'
           'and all the rest will be computed (if available)')
    @use_args(_DocumentIndexListSchemaInput(strict=True))
    @marshal_with(DocumentIndexListSchema())
    def get(self, dsid, return_file_path=False, **args):
        fe = FeatureVectorizer(self._cache_dir, dsid=dsid)
        query = pd.DataFrame(args)
        res = fe.db.search(query)
        res_repr = fe.db.render_list(res, return_file_path=return_file_path)
        return res_repr

class _DocumentIndexNestedSchemaInput(DocumentIndexNestedSchema):
    return_file_path = wfields.Boolean()

class FeaturesApiElementMappingNested(Resource):
    @doc(description='Compute correspondence between id fields (nested). '
           'At least one of the fields used for indexing must be provided,'
           'and all the rest will be computed (if available)')
    @use_args(_DocumentIndexNestedSchemaInput(strict=True))
    @marshal_with(DocumentIndexNestedSchema())
    def get(self, dsid, return_file_path=False, **args):
        fe = FeatureVectorizer(self._cache_dir, dsid=dsid)
        query = pd.DataFrame(args['data'])
        res = fe.db.search(query)
        res_repr = fe.db.render_dict(res, return_file_path=return_file_path)
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
    def get(self, dsid, **args):
        fe = EmailParser(self._cache_dir, dsid=dsid)
        idx = fe.search(args['filenames'])
        return {'index': list(idx)}

# ============================================================================ # 
#                  LSI decomposition
# ============================================================================ # 

_lsi_api_get_args  = {'parent_id': wfields.Str(required=True) }
_lsi_api_post_args = {'parent_id': wfields.Str(required=True),
                      'n_components': wfields.Int(default=150) }
class LsiApi(Resource):

    @use_args(_lsi_api_get_args)
    @marshal_with(LsiParsSchema(many=True))
    def get(self, **args):
        parent_id = args['parent_id']
        lsi = _LSIWrapper(cache_dir=self._cache_dir, parent_id=parent_id)
        return lsi.list_models()

    @doc(description=dedent("""
           Build a Latent Semantic Indexing (LSI) model

           The option `use_hashing=True` must be set for the feature extraction.
           Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

           The recommended value for the `n_components` (dimensions of the SVD decompositions) is
           in the [100, 200] range.

           **Parameters**
             - `n_components`: Desired dimensionality of the output data. Must be strictly less than the number of features.
             - `parent_id`: `dataset_id`
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
    @marshal_with(CategorizationParsSchema(many=True))
    def get(self, parent_id):
        cat = _CategorizerWrapper(parent_id, self._cache_dir)

        return cat.list_models()

    @doc(description=dedent("""
           Build the categorization ML model

           The option `use_hashing=True` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

           **Parameters**
            - `parent_id`: `dataset_id` or `lsi_id`
            - `index`: (optional) internal document ids of the training set (can also be provided in `index_nested`)
            - `y`: (optional) target binary class relative to index (can also be provided in `index_nested`)
            - `index_nested`: a list of dict which have a `y` field and one or several fields that can be used for indexing, such as `internal_id`, `document_id`, `file_path`, `rendition_id`.
            - `method`: classification algorithm to use (default: LogisticRegression),
              * "LogisticRegression": [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
              * "LinearSVC": [Linear SVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html),
              * "NearestNeighbor": nearest neighbor classifier (requires LSI)
              * "NearestCentroid": nearest centroid classifier (requires LSI)
              * "xgboost": [Gradient Boosting](https://xgboost.readthedocs.io/en/latest/model.html)
                   (*Warning:* for the moment xgboost is not istalled for a direct install on Windows)
            - `cv`: binary, if true optimal parameters of the ML model are determined by cross-validation over 5 stratified K-folds (default True).
            - `training_scores`: binary, compute the efficiency scores on the training dataset (default True).
          """))
    @use_args(_CategorizationInputSchema())
    @marshal_with(CategorizationPostSchema())
    def post(self, **args):
        training_scores = args['training_scores']
        parent_id = args['parent_id']
        cat = _CategorizerWrapper(self._cache_dir, parent_id=parent_id)

        if 'y' in args and 'index' in args:
            pass
        elif 'index_nested' in args:
            query = pd.DataFrame(args['index_nested'])
            res_q = cat.fe.db.search(query, drop=False)
            del args['index_nested']
            args['index'] = res_q.internal_id.values
            args['y'] = res_q.y.values
        else:
            raise WrongParameter("Either 'index_nested' or y and index must be provided!")


        if args['cv']:
            cv = 'fast'
        else:
            cv = None
        for key in ['parent_id', 'cv', 'training_scores']:
            del args[key]
        _, Y_train = cat.train(cv=cv, **args)
        idx_train = args['index']
        if training_scores:
            Y_res, md = cat.predict()
            idx_res = np.arange(cat.fe.n_samples_, dtype='int')
            res = categorization_score(idx_train, Y_train, idx_res, Y_res)
        else:
            res = {"recall": -1, "precision": -1 , "f1": -1, 
                'auc_roc': -1, 'average_precision': -1}
        res['id'] = cat.mid
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

    @doc(description='Predict document categorization with a previously trained model')
    @marshal_with(CategorizationPredictSchema())
    def get(self, mid):

        cat = _CategorizerWrapper(self._cache_dir, mid=mid)
        y_res, md = cat.predict()
        md = {key: val.tolist() for key, val in md.items()}
        scores_pd = pd.DataFrame({'score': y_res,
                                  'internal_id': np.arange(cat.fe.n_samples_, dtype='int')})
        if not md:

            view_data = cat.fe.db.render_dict(scores_pd)
        else:
            res = scores_pd.set_index('internal_id')
            db = cat.fe.db.data.set_index('internal_id', drop=False)
            base_keys = [key for key in cat.fe.db.data.columns if key != 'file_path']

            view_data = []
            for index, row in res.iterrows():
                row_dict = row.to_dict()
                db_sel = db.loc[index]
                row_dict.update(db_sel[base_keys].to_dict())
                row_dict['internal_id'] = index
                for ind_key, dist_key, field_name in [('ind_p', 'dist_p', 'nn_positive'),
                                                      ('ind_n', 'dist_n', 'nn_negative')]:

                    if ind_key not in md:
                        # we are in unsupervised mode, there is no negative samples
                        row_dict[field_name] = {}
                    else:
                        nn_ind = md[ind_key][index]
                        db_sel = db.loc[nn_ind]
                        nn_tmp = db_sel[base_keys].to_dict()
                        nn_tmp['distance'] = md[dist_key][index]
                        row_dict[field_name] = nn_tmp
                view_data.append(row_dict)

        return dict(data=view_data)


_models_api_test = {'ground_truth_filename' : wfields.Str(required=True)}


class ModelsApiTest(Resource):

    @doc(description=dedent("""
           Test the categorization model

           **Parameters**
            - `ground_truth_filename`: [required] tab-delimited file name with a unique document ID followed by a 1 for relevant or 0 for non-relevant document

          """))
    @use_args(_models_api_test)
    @marshal_with(ClassificationScoresSchema())
    def post(self, mid, **args):
        cat = _CategorizerWrapper(self._cache_dir, mid=mid)

        y_res, md = cat.predict()
        d_ref = parse_ground_truth_file( args["ground_truth_filename"])
        idx_ref = cat.fe.db._search_filenames(d_ref.file_path.values)
        idx_res = np.arange(cat.fe.n_samples_, dtype='int')
        res = categorization_score(idx_ref,
                                   d_ref.is_relevant.values,
                                   idx_res, y_res)
        return res


# ============================================================================ # 
#                              Clustering
# ============================================================================ # 

_k_mean_clustering_api_post_args = {
        'parent_id': wfields.Str(required=True),
        'n_clusters': wfields.Int(default=150),
        }


class KmeanClusteringApi(Resource):

    @doc(description=dedent("""
           Compute K-mean clustering

           The option `use_hashing=False` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

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


_birch_clustering_api_post_args = {
        'parent_id': wfields.Str(required=True),
        'n_clusters': wfields.Int(default=150),
        'threshold': wfields.Number(),
        }


class BirchClusteringApi(Resource):

    @doc(description=dedent("""
           Compute birch clustering

           The option `use_hashing=False` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

           **Parameters**
            - `parent_id`: `dataset_id` or `lsi_id`
            - `n_clusters`: the number of clusters
            - `threshold`: The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold. Otherwise a new subcluster is started. See [sklearn.cluster.Birch](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html)
           """))
    @use_args(_birch_clustering_api_post_args)
    @marshal_with(IDSchema())
    def post(self, **args):

        cl = _ClusteringWrapper(cache_dir=self._cache_dir, parent_id=args['parent_id'])
        del args['parent_id']
        cl.birch(**args)
        return {'id': cl.mid}


_wardhc_clustering_api_post_args = {
        'parent_id': wfields.Str(required=True),
        'n_clusters': wfields.Int(default=150),
        'n_neighbors': wfields.Int(missing=5),
        }


class WardHCClusteringApi(Resource):

    @doc(description=dedent("""
           Compute Ward Hierarchical Clustering.

           The option `use_hashing=False` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

           The Ward Hierarchical clustering is generally slower that K-mean, however the run time can be reduced by decreasing the following parameters,

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

_dbscan_clustering_api_post_args = {
        'parent_id': wfields.Str(required=True),
        'eps': wfields.Number(missing=0.1),
        'min_samples': wfields.Int(missing=10)
        }


class DBSCANClusteringApi(Resource):

    @doc(description=dedent("""
           Compute clustering (DBSCAN)

           The option `use_hashing=False` must be set for the feature extraction. Recommended options also include, `use_idf=1, sublinear_tf=0, binary=0`.

           **Parameters**
             - `parent_id`: `dataset_id` or `lsi_id`
             - `eps`: (optional) float The maximum distance between two samples for them to be considered as in the same neighborhood.
             - `min_samples`: (optional) int The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
            """))
    @use_args(_dbscan_clustering_api_post_args)
    @marshal_with(IDSchema())
    def post(self, **args):

        cl = _ClusteringWrapper(cache_dir=self._cache_dir, parent_id=args['parent_id'])

        del args['parent_id']

        cl.dbscan(**args)
        return {'id': cl.mid}


_clustering_api_get_args = {
        'n_top_words': wfields.Int(missing=5)
        }


class ClusteringApiElement(Resource):

    @doc(description=dedent("""
           Compute cluster labels

           **Parameters**
            - `n_top_words`: keep only most relevant `n_top_words` words
            - `label_method`: str, default='centroid-frequency' the method used for computing the cluster labels
            """))
    @use_args(_clustering_api_get_args)
    @marshal_with(ClusteringSchema())
    def get(self, method, mid, **args):  # TODO unused parameter 'method'

        cl = _ClusteringWrapper(cache_dir=self._cache_dir, mid=mid)

        km = cl._load_model()
        htree = cl._get_htree(km)
        if 'children' in htree:
            htree['children'] = htree['children'].tolist()
        if args['n_top_words'] > 0:
            terms = cl.compute_labels(**args)
        else:
            terms = []

        pars = cl._load_pars()
        return {'labels': km.labels_.tolist(), 'cluster_terms': terms,
                  'htree': htree, 'pars': pars}


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

_dupdet_api_get_args = {
        'distance': wfields.Int(),
        'n_rand_lexicons': wfields.Int(),
        'rand_lexicon_ratio': wfields.Number()
        }


class DupDetectionApiElement(Resource):

    @doc(description=dedent("""
           Query duplicates

           **Parameters**
            - distance : int, default=2 Maximum number of differnet bits in the simhash (Simhash method only) - n_rand_lexicons : int, default=1
              number of random lexicons used for duplicate detection (I-Match method only)
            - rand_lexicon_ratio: float, default=0.7 ratio of the vocabulary used in random lexicons (I-Match method only)
          """))
    @use_args(_dupdet_api_get_args)
    @marshal_with(DuplicateDetectionSchema())
    def get(self, mid, **args):

        model = _DuplicateDetectionWrapper(cache_dir=self._cache_dir, mid=mid)
        cluster_id = model.query(**args)
        return {'cluster_id': cluster_id}


    @marshal_with(EmptySchema())
    def delete(self, mid):

        model = _DuplicateDetectionWrapper(cache_dir=self._cache_dir, mid=mid)
        model.delete()
        return {}



# ============================================================================ #
#                             Metrics                                          #
# ============================================================================ #
_metrics_categorization_api_get_args  = {
    'y_true': wfields.List(wfields.Int(), required=True),
    'y_pred': wfields.List(wfields.Number(), required=True),
    'metrics': wfields.List(wfields.Str())
}

class MetricsCategorizationApiElement(Resource):
    @doc(description=dedent("""
          Compute categorization metrics to assess the quality
          of categorization, comparing the groud truth labels with the predicted ones.

          **Parameters**
            - y_true: [required] list of int. Ground truth labels
            - y_pred: [required] list of int. Predicted labels
            - metrics: [required] list of str. Metrics to compute, any combination of "precision", "recall", "f1", "roc_auc"
          """))
    @use_args(_metrics_categorization_api_get_args)
    @marshal_with(MetricsCategorizationSchema())
    def get(self, **args):
        output_metrics = {}
        y_true = np.array(args['y_true'], dtype='int')
        y_pred = np.array(args['y_pred'], dtype='float')
        threshold = 0
        y_pred_b = (y_pred > threshold).astype('int')
        if 'metrics' in args:
            metrics = args['metrics']
        else:
            metrics = ['precision', 'recall', 'roc_auc', 'f1', 'average_precision']

        # wrapping metrics calculations, as for example F1 score can frequently print warnings
        # "F-score is ill defined and being set to 0.0 due to no predicted samples"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            for func, y_targ in [(precision_score, y_pred_b),
                                 (recall_score, y_pred_b),
                                 (f1_score, y_pred_b),
                                 (roc_auc_score, y_pred),
                                 (average_precision_score, y_pred)]:
                name = func.__name__.replace('_score', '')
                if name in metrics:
                    output_metrics[name] = func(y_true, y_targ)

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
    def get(self, **args):
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
    def get(self, **args):
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
    @doc(description="Perform document search (if `parent_id` is a `dataset_id`)"
                     " or a semantic search (if `parent_id` is a `lsi_id`).")
    @use_args({ "parent_id": wfields.Str(required=True),
                "query": wfields.Str(required=True)})
    @marshal_with(SearchResponseSchema())
    def get(self, **args):
        parent_id = args['parent_id']
        model = _SearchWrapper(cache_dir=self._cache_dir, parent_id=parent_id)

        query = args['query']
        scores = model.search(query)
        scores_pd = pd.DataFrame({'score': scores,
                                  'internal_id': np.arange(model.fe.n_samples_, dtype='int')})

        res = model.fe.db.render_dict(scores_pd)

        return {'data': res}
