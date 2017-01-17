# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from glob import glob

from flask_restful import abort, Resource
from webargs import fields as wfields
from flask_apispec import marshal_with, use_kwargs as use_args
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, \
                            adjusted_rand_score, adjusted_mutual_info_score, v_measure_score
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
from ..utils import categorization_score
from ..cluster import _ClusteringWrapper
from ..search import _SearchWrapper
from ..metrics import ratio_duplicates_score, f1_same_duplicates_score, mean_duplicates_count_score
from ..dupdet import _DuplicateDetectionWrapper
from ..threading import _EmailThreadingWrapper
from .schemas import (IDSchema, FeaturesParsSchema,
                      FeaturesSchema, FeaturesElementIndexSchema,
                      EmailParserSchema, EmailParserElementIndexSchema,
                      DatasetSchema,
                      LsiParsSchema, LsiPostSchema,
                      ClassificationScoresSchema,
                      CategorizationParsSchema, CategorizationPostSchema,
                      CategorizationPredictSchema, ClusteringSchema,
                      ErrorSchema, DuplicateDetectionSchema,
                      MetricsCategorizationSchema, MetricsClusteringSchema,
                      MetricsDupDetectionSchema,
                      EmailThreadingSchema, EmailThreadingParsSchema,
                      ErrorSchema, DuplicateDetectionSchema,
                      SearchResponseSchema
                      )

EPSILON = 1e-3 # small numeric value

# ============================================================================ # 
#                         Datasets download                                    #
# ============================================================================ # 

class DatasetsApiElement(Resource):

    @marshal_with(DatasetSchema())
    def get(self, name):
        from ..datasets import load_dataset
        res = load_dataset(name, self._cache_dir, verbose=True,
                load_ground_truth=True, verify_checksum=False)
        return res


# Definine the response formatting schemas
id_schema = IDSchema()
features_schema = FeaturesSchema()
error_schema = ErrorSchema()

# ============================================================================ # 
#                      Feature extraction                                      #
# ============================================================================ # 

class FeaturesApi(Resource):

    @marshal_with(FeaturesSchema(many=True))
    def get(self):
        fe = FeatureVectorizer(self._cache_dir)
        return fe.list_datasets()

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

    @marshal_with(IDSchema())
    def post(self, dsid):
        fe = FeatureVectorizer(self._cache_dir, dsid=dsid)
        dsid, _ = fe.transform()
        return {'id': dsid}

    def delete(self, dsid):
        fe = FeatureVectorizer(self._cache_dir, dsid=dsid)
        fe.delete()

_features_api_index_get_args = {'filenames': wfields.List(wfields.Str(), required=True)}
class FeaturesApiElementIndex(Resource):
    @use_args(_features_api_index_get_args)
    @marshal_with(FeaturesElementIndexSchema())
    def get(self, dsid, **args):
        fe = FeatureVectorizer(self._cache_dir, dsid=dsid)
        idx = fe.search(args['filenames'])
        return {'index': list(idx)}

# ============================================================================ # 
#                   Email parser                                      #
# ============================================================================ # 

class EmailParserApi(Resource):

    def get(self):
        fe = EmailParser(self._cache_dir)
        return fe.list_datasets()

    @use_args({'data_dir': wfields.Str(required=True)})
    @marshal_with(EmailParserSchema())
    def post(self, **args):
        fe = EmailParser(self._cache_dir)
        dsid = fe.transform(**args)
        pars = fe.get_params()
        return {'id': dsid, 'filenames': pars['filenames']}


class EmailParserApiElement(Resource):
    def get(self, dsid):
        fe = EmailParser(self._cache_dir, dsid=dsid)
        out = fe.get_params()
        return out

    def delete(self, dsid):
        fe = EmailParser(self._cache_dir, dsid=dsid)
        fe.delete()


class EmailParserApiElementIndex(Resource):
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
                      'n_components': wfields.Int(default=100) }
class LsiApi(Resource):

    @use_args(_lsi_api_get_args)
    @marshal_with(LsiParsSchema(many=True))
    def get(self, **args):
        parent_id = args['parent_id']
        lsi = _LSIWrapper(cache_dir=self._cache_dir, parent_id=parent_id)
        return lsi.list_models()

    @use_args(_lsi_api_post_args)
    @marshal_with(LsiPostSchema())
    def post(self, **args):
        parent_id = args['parent_id']
        del args['parent_id']
        lsi = _LSIWrapper(cache_dir=self._cache_dir, parent_id=parent_id)
        _, explained_variance = lsi.fit_transform(**args)
        return {'id': lsi.mid, 'explained_variance': explained_variance}


class LsiApiElement(Resource):

    @marshal_with(LsiParsSchema())
    def get(self, mid):
        cat = _LSIWrapper(self._cache_dir, mid=mid)

        pars = cat._load_pars()
        pars['parent_id'] = pars['parent_id']
        return pars

_lsi_api_element_predict_post_args = {
        # Warning this should be changed to wfields.DelimitedList
        # https://webargs.readthedocs.io/en/latest/api.html#webargs.fields.DelimitedList
        'index': wfields.List(wfields.Int(), required=True),
        'y': wfields.List(wfields.Int(), required=True),
        }


# ============================================================================ # 
#                  Categorization (ML)
# ============================================================================ # 

_models_api_post_args = {
        'parent_id': wfields.Str(required=True),
        # Warning this should be changed to wfields.DelimitedList
        # https://webargs.readthedocs.io/en/latest/api.html#webargs.fields.DelimitedList
        'index': wfields.List(wfields.Int(), required=True),
        'y': wfields.List(wfields.Int(), required=True),
        'method': wfields.Str(required=True),
        'cv': wfields.Boolean(missing=False),
        'training_scores': wfields.Boolean(missing=True)
        }


class ModelsApi(Resource):
    @marshal_with(CategorizationParsSchema(many=True))
    def get(self, parent_id):
        cat = _CategorizerWrapper(parent_id, self._cache_dir)

        return cat.list_models()

    @use_args(_models_api_post_args)
    @marshal_with(CategorizationPostSchema())
    def post(self, **args):
        training_scores = args['training_scores']
        parent_id = args['parent_id']

        if args['cv']:
            cv = 'fast'
        else:
            cv = None
        for key in ['parent_id', 'cv', 'training_scores']:
            del args[key]
        cat = _CategorizerWrapper(self._cache_dir, parent_id=parent_id)
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
    @marshal_with(CategorizationParsSchema())
    def get(self, mid):
        cat = _CategorizerWrapper(self._cache_dir, mid=mid)
        pars = cat.get_params()
        return pars

    def delete(self, mid):
        cat = _CategorizerWrapper(self._cache_dir, mid=mid)
        cat.delete()


class ModelsApiPredict(Resource):

    @marshal_with(CategorizationPredictSchema())
    def get(self, mid):

        cat = _CategorizerWrapper(self._cache_dir, mid=mid)
        y_res, md = cat.predict()
        md = {key: val.tolist() for key, val in md.items()}

        return dict(prediction=y_res.tolist(), **md)


_models_api_test = {'ground_truth_filename' : wfields.Str(required=True)}


class ModelsApiTest(Resource):

    @use_args(_models_api_test)
    @marshal_with(ClassificationScoresSchema())
    def post(self, mid, **args):
        cat = _CategorizerWrapper(self._cache_dir, mid=mid)

        y_res, md = cat.predict()
        d_ref = parse_ground_truth_file( args["ground_truth_filename"])
        idx_ref = cat.fe.search(d_ref.index.values)
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
        'n_clusters': wfields.Int(required=True),
        }


class KmeanClusteringApi(Resource):

    @use_args(_k_mean_clustering_api_post_args)
    @marshal_with(IDSchema())
    def post(self, **args):

        cl = _ClusteringWrapper(cache_dir=self._cache_dir, parent_id=args['parent_id'])

        del args['parent_id']

        labels = cl.k_means(**args)  # TODO unused variable. Remove?
        return {'id': cl.mid}


_birch_clustering_api_post_args = {
        'parent_id': wfields.Str(required=True),
        'n_clusters': wfields.Int(required=True),
        'threshold': wfields.Number(),
        }


class BirchClusteringApi(Resource):

    @use_args(_birch_clustering_api_post_args)
    @marshal_with(IDSchema())
    def post(self, **args):

        cl = _ClusteringWrapper(cache_dir=self._cache_dir, parent_id=args['parent_id'])
        del args['parent_id']
        cl.birch(**args)
        return {'id': cl.mid}


_wardhc_clustering_api_post_args = {
        'parent_id': wfields.Str(required=True),
        'n_clusters': wfields.Int(required=True),
        'n_neighbors': wfields.Int(missing=5),
        }


class WardHCClusteringApi(Resource):

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


    def delete(self, method, mid):  # TODO unused parameter 'method'
        cl = _ClusteringWrapper(cache_dir=self._cache_dir, mid=mid)
        cl.delete()

# ============================================================================ # 
#                              Duplicate detection
# ============================================================================ # 

_dup_detection_api_post_args = {
        "parent_id": wfields.Str(required=True),
        "method": wfields.Str(required=False, missing='simhash')
        }


class DupDetectionApi(Resource):

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

    @use_args(_dupdet_api_get_args)
    @marshal_with(DuplicateDetectionSchema())
    def get(self, mid, **args):

        model = _DuplicateDetectionWrapper(cache_dir=self._cache_dir, mid=mid)
        cluster_id = model.query(**args)
        return {'cluster_id': cluster_id}

    def delete(self, mid):

        model = _DuplicateDetectionWrapper(cache_dir=self._cache_dir, mid=mid)
        model.delete()



# ============================================================================ #
#                             Metrics                                          #
# ============================================================================ #
_metrics_categorization_api_get_args  = {
    'y_true': wfields.List(wfields.Int(), required=True),
    'y_pred': wfields.List(wfields.Int(), required=True),
    'metrics': wfields.List(wfields.Str(), required=True)
}

class MetricsCategorizationApiElement(Resource):
    @use_args(_metrics_categorization_api_get_args)
    @marshal_with(MetricsCategorizationSchema())
    def get(self, **args):
        output_metrics = dict()
        y_true = args['y_true']
        y_pred = args['y_pred']
        metrics = args['metrics']

        # wrapping metrics calculations, as for example F1 score can frequently print warnings
        # "F-score is ill defined and being set to 0.0 due to no predicted samples"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            if 'precision' in metrics:
                output_metrics['precision'] = precision_score(y_true, y_pred)
            if 'recall' in metrics:
                output_metrics['recall'] = recall_score(y_true, y_pred)
            if 'f1' in metrics:
                output_metrics['f1'] = f1_score(y_true, y_pred)
            if 'roc_auc' in metrics:
                output_metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
        return output_metrics


_metrics_clustering_api_get_args  = {
    'labels_true': wfields.List(wfields.Int(), required=True),
    'labels_pred': wfields.List(wfields.Int(), required=True),
    'metrics': wfields.List(wfields.Str(), required=True)
}

class MetricsClusteringApiElement(Resource):
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

    @use_args({ "parent_id": wfields.Str(required=True)})
    @marshal_with(EmailThreadingSchema())
    def post(self, **args):

        model = _EmailThreadingWrapper(cache_dir=self._cache_dir, parent_id=args['parent_id'])

        tree =  model.thread()

        return {'data': [el.to_dict(include=['subject']) for el in tree],
                'id': model.mid}

class EmailThreadingApiElement(Resource):

    @marshal_with(EmailThreadingParsSchema())
    def get(self, mid):

        model = _EmailThreadingWrapper(cache_dir=self._cache_dir, mid=mid)

        return model.get_params()

    def delete(self, mid):

        model = _EmailThreadingWrapper(cache_dir=self._cache_dir, mid=mid)
        model.delete()


# ============================================================================ # 
#                              (Semantic) search
# ============================================================================ # 

class SearchApi(Resource):
    @use_args({ "parent_id": wfields.Str(required=True),
                "query": wfields.Str(required=True)})
    @marshal_with(SearchResponseSchema())
    def get(self, **args):
        parent_id = args['parent_id']
        model = _SearchWrapper(cache_dir=self._cache_dir, parent_id=parent_id)

        query = args['query']
        scores = model.search(query)
        return {'prediction': scores}
