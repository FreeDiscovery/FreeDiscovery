# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from glob import glob

from flask_restful import abort, Resource
from webargs import fields as wfields
#from webargs.flaskparser import use_args
from flask_apispec import marshal_with, use_kwargs as use_args
from marshmallow import Schema, pprint, fields

from ..text import FeatureVectorizer
from ..lsi import LSI
from ..categorization import Categorizer
from ..io import parse_ground_truth_file
from ..utils import classification_score
from ..clustering import Clustering
from ..exceptions import (DatasetNotFound, ModelNotFound, InitException, 
                            NotFound, WrongParameter)
from .schemas import (IDSchema, FeaturesParsSchema, 
                        FeaturesSchema,
                        LsiParsSchema, LsiPostSchema, LsiPredictSchema,
                        ClassificationScoresSchema,
                        CategorizationParsSchema, CategorizationPostSchema,
                        CategorizationPredictSchema, ClusteringSchema,
                        ErrorSchema, DuplicateDetectionSchema
                        )


# Definine the response formatting schemas

id_schema = IDSchema()
features_schema = FeaturesSchema()
error_schema = ErrorSchema()

# ============================================================================ # 
#                      Features extraction                                     #
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



# ============================================================================ # 
#                  Categorization (LSI)
# ============================================================================ # 


_lsi_api_get_args  = {'dataset_id': wfields.Str(required=True) }
_lsi_api_post_args = {'dataset_id': wfields.Str(required=True),
                      'n_components': wfields.Int(default=100) }

class LsiApi(Resource):

    @use_args(_lsi_api_get_args)
    @marshal_with(LsiParsSchema(many=True))
    def get(self, **args):
        dsid = args['dataset_id']

        lsi = LSI(cache_dir=self._cache_dir, dsid=dsid)

        return lsi.list_models()


    @use_args(_lsi_api_post_args)
    @marshal_with(LsiPostSchema())
    def post(self, **args):
        dsid = args['dataset_id']
        del args['dataset_id']

        lsi = LSI(cache_dir=self._cache_dir, dsid=dsid)
        _, explained_variance = lsi.transform(**args)
        return {'id': lsi.mid, 'explained_variance': explained_variance}


class LsiApiElement(Resource):

    @marshal_with(LsiParsSchema())
    def get(self, mid):
        cat = LSI(self._cache_dir, mid=mid)

        pars = cat._load_pars(mid)
        pars['dataset_id'] = pars['dsid']
        return pars

_lsi_api_element_predict_post_args = {
        # Warning this should be changed to wfields.DelimitedList
        # https://webargs.readthedocs.io/en/latest/api.html#webargs.fields.DelimitedList
        'relevant_filenames': wfields.List(wfields.Str(), required=True),
        'non_relevant_filenames': wfields.List(wfields.Str(), required=True),
        }

class LsiApiElementPredict(Resource):
    @use_args(_lsi_api_element_predict_post_args)
    @marshal_with(LsiPredictSchema())
    def post(self, mid, **args):

        lsi = LSI(self._cache_dir, mid=mid)

        _, X_train, Y_train, Y_train_res, X_test, Y_test_res, res  = lsi.predict(
                accumulate='nearest-max', **args) 
        res_scores = classification_score(X_train, Y_train, X_train, Y_train_res)
        res_scores.update({'prediction': Y_test_res.tolist(),
                    'prediction_rel': res['D_d_p'].tolist(),
                    'prediction_nrel': res['D_d_n'].tolist(),
                    'nearest_rel_doc': res['idx_d_p'].tolist(),
                    'nearest_nrel_doc': res['idx_d_n'].tolist(),
                     })
        return res_scores

_lsi_api_element_test_post_args = {
        # Warning this should be changed to wfields.DelimitedList
        # https://webargs.readthedocs.io/en/latest/api.html#webargs.fields.DelimitedList
        'relevant_filenames': wfields.List(wfields.Str(), required=True),
        'non_relevant_filenames': wfields.List(wfields.Str(), required=True),
        'ground_truth_filename': wfields.Str(required=True)
        }

class LsiApiElementTest(Resource):
    @use_args(_lsi_api_element_test_post_args)
    @marshal_with(ClassificationScoresSchema())
    def post(self, mid, **args):

        lsi = LSI(self._cache_dir, mid=mid)

        d_ref = parse_ground_truth_file(os.path.join(lsi.fe.dsid_dir, args["ground_truth_filename"]))
        del args['ground_truth_filename']
        lsi_m, X_train, Y_train, Y_train_res, X_test, Y_test_res, res  = lsi.predict(
                accumulate='nearest-max', **args) 
        res = classification_score(d_ref.index.values,
                                   d_ref.is_relevant.values, X_test, Y_test_res)
        return res


# ============================================================================ # 
#                  Categorization (ML)
# ============================================================================ # 

_models_api_post_args = {
        'dataset_id': wfields.Str(required=True),
        # Warning this should be changed to wfields.DelimitedList
        # https://webargs.readthedocs.io/en/latest/api.html#webargs.fields.DelimitedList
        'relevant_filenames': wfields.List(wfields.Str(), required=True),
        'non_relevant_filenames': wfields.List(wfields.Str(), required=True),
        'method': wfields.Str(required=True),
        'cv': wfields.Boolean(missing=True),
        'training_scores': wfields.Boolean(missing=True)
        }


class ModelsApi(Resource):
    @marshal_with(CategorizationParsSchema(many=True))
    def get(self, dsid):
        cat = Categorizer(dsid, self._cache_dir)

        return cat.list_models()


    @use_args(_models_api_post_args)
    @marshal_with(CategorizationPostSchema())
    def post(self, **args):
        training_scores = args['training_scores']
        dsid = args['dataset_id']

        if args['cv']:
            cv = 'fast'
        else:
            cv = None

        for key in ['dataset_id', 'cv', 'training_scores']:
            del args[key]

        cat = Categorizer(self._cache_dir, dsid=dsid)

        _, X_train, Y_train = cat.train(cv=cv, **args)
        if training_scores:
            Y_res = cat.predict()
            X_res = cat.fe._pars['filenames']
            res = classification_score(X_train, Y_train, X_res, Y_res)
        else:
            res = {"recall": -1, "precision": -1 , "f1": -1, 
                'auc_roc': -1, 'average_precision': -1}
        res['id'] = cat.mid
        return res


class ModelsApiElement(Resource):
    @marshal_with(CategorizationParsSchema())
    def get(self, mid):

        cat = Categorizer(self._cache_dir, mid=mid)
        pars = cat.get_params()
        return pars


    def delete(self, mid):
        cat = Categorizer(self._cache_dir, mid=mid)
        cat.delete()


class ModelsApiPredict(Resource):

    @marshal_with(CategorizationPredictSchema())
    def get(self, mid):

        cat = Categorizer(self._cache_dir, mid=mid)
        y_res = cat.predict()

        return {'prediction': y_res.tolist()}


_models_api_test = {'ground_truth_filename' : wfields.Str(required=True)}

class ModelsApiTest(Resource):

    @use_args(_models_api_test)
    @marshal_with(ClassificationScoresSchema())
    def post(self, mid, **args):
        cat = Categorizer(self._cache_dir, mid=mid)

        y_res = cat.predict()
        d_ref = parse_ground_truth_file(os.path.join(cat.fe.dsid_dir,
                                                     args["ground_truth_filename"]))
        res = classification_score(d_ref.index.values,
                                   d_ref.is_relevant.values,
                                   cat.fe._pars['filenames'], y_res)
        return res



# ============================================================================ # 
#                              Clustering
# ============================================================================ # 

_k_mean_clustering_api_post_args = {
        'dataset_id': wfields.Str(required=True),
        'n_clusters': wfields.Int(required=True),
        'lsi_components': wfields.Int(missing=-1),
        }

class KmeanClusteringApi(Resource):

    @use_args(_k_mean_clustering_api_post_args)
    @marshal_with(IDSchema())
    def post(self, **args):

        if args['lsi_components'] < 0:
            args['lsi_components'] = None

        cl = Clustering(cache_dir=self._cache_dir, dsid=args['dataset_id'])

        del args['dataset_id']

        labels = cl.k_means(**args)
        return {'id': cl.mid}


_birch_clustering_api_post_args = {
        'dataset_id': wfields.Str(required=True),
        'n_clusters': wfields.Int(required=True),
        'lsi_components': wfields.Int(missing=-1),
        'threshold': wfields.Number(),
        }

class BirchClusteringApi(Resource):

    @use_args(_birch_clustering_api_post_args)
    @marshal_with(IDSchema())
    def post(self, **args):

        if args['lsi_components'] < 0:
            args['lsi_components'] = None

        cl = Clustering(cache_dir=self._cache_dir, dsid=args['dataset_id'])

        del args['dataset_id']

        cl.birch(**args)
        return {'id': cl.mid}



_wardhc_clustering_api_post_args = {
        'dataset_id': wfields.Str(required=True),
        'n_clusters': wfields.Int(required=True),
        'lsi_components': wfields.Int(missing=-1),
        'n_neighbors': wfields.Int(missing=5),
        }

class WardHCClusteringApi(Resource):

    @use_args(_wardhc_clustering_api_post_args)
    @marshal_with(IDSchema())
    def post(self, **args):

        if args['lsi_components'] < 0:
            args['lsi_components'] = None

        cl = Clustering(cache_dir=self._cache_dir, dsid=args['dataset_id'])

        del args['dataset_id']

        cl.ward_hc(**args)
        return {'id': cl.mid}


_clustering_api_get_args = {
        'n_top_words': wfields.Int(missing=5)
        }

class ClusteringApiElement(Resource):

    @use_args(_clustering_api_get_args)
    @marshal_with(ClusteringSchema())
    def get(self, method, mid, **args):

        cl = Clustering(cache_dir=self._cache_dir, mid=mid)

        km = cl.load(mid=mid)
        htree = cl._get_htree(km)
        if 'children' in htree:
            htree['children'] = htree['children'].tolist()
        terms = cl.compute_labels(**args)
        pars = cl._load_pars()
        if pars['lsi']:
            pars['lsi'] = True
        return {'labels': km.labels_.tolist(), 'cluster_terms': terms,
                  'htree': htree, 'pars': pars}


    def delete(self, method, mid):
        cl = Clustering(cache_dir=self._cache_dir, mid=mid)
        cl.delete()

# ============================================================================ # 
#                              Duplicate detection
# ============================================================================ # 

_dup_detection_api_post_args = {
        'dataset_id': wfields.Str(required=True),
        "method": wfields.Str(required=False, missing='simhash')
        }

class DupDetectionApi(Resource):

    @use_args(_dup_detection_api_post_args)
    @marshal_with(IDSchema())
    def post(self, **args):
        from ..simhash import DuplicateDetection

        model = DuplicateDetection(cache_dir=self._cache_dir, dsid=args['dataset_id'])

        del args['dataset_id']


        model.fit()

        return {'id': model.mid}

_dupdet_api_get_args = {
        'distance': wfields.Int(missing=2)
        }

class DupDetectionApiElement(Resource):

    @use_args(_dupdet_api_get_args)
    @marshal_with(DuplicateDetectionSchema())
    def get(self, mid, **args):
        from ..simhash import DuplicateDetection

        model = DuplicateDetection(cache_dir=self._cache_dir, mid=mid)
        model.fit()
        simhash, cluster_id, dup_pairs = model.query(**args)
        return {'simhash': simhash, 'cluster_id': cluster_id,
                'dup_pairs': dup_pairs}


    def delete(self, mid):
        from ..simhash import DuplicateDetection

        model = DuplicateDetection(cache_dir=self._cache_dir, mid=mid)
        model.delete()
