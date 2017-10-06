# -*- coding: utf-8 -*-

from textwrap import dedent

from webargs import fields as wfields
from flask_apispec import (marshal_with, use_kwargs as use_args,
                           MethodResource as Resource)
from flask_apispec.annotations import doc
import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score,
                             adjusted_mutual_info_score,
                             precision_score, recall_score, f1_score,
                             v_measure_score)
from sklearn.metrics import (roc_auc_score, average_precision_score)
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from freediscovery.engine.vectorizer import FeatureVectorizer
from freediscovery.engine.ingestion import _check_mutual_index
from freediscovery.engine.lsi import _LSIWrapper
from freediscovery.engine.categorization import _CategorizerWrapper
from freediscovery.utils import _docstring_description, _paginate
from freediscovery.cluster import centroid_similarity, compute_optimal_sampling
from freediscovery.engine.cluster import _ClusteringWrapper
from freediscovery.engine.search import _SearchWrapper
from freediscovery.metrics import (categorization_score,
                       ratio_duplicates_score, f1_same_duplicates_score,
                       mean_duplicates_count_score, _scale_cosine_similarity)
from freediscovery.engine.near_duplicates import _DuplicateDetectionWrapper
from freediscovery.engine.email_threading import _EmailThreadingWrapper
from freediscovery.datasets import load_dataset
from freediscovery.exceptions import WrongParameter
from freediscovery.engine.stop_words import _StopWordsWrapper
from .validators import _is_in_range

from .schemas import (IDSchema, FeaturesParsSchema,
                      FeaturesSchema, _DatasetDefinitionShort,
                      DocumentIndexNestedSchema, _DatasetDefinition,
                      ExampleDatasetSchema, DocumentIndexFullSchema,
                      LsiParsSchema, LsiPostSchema,
                      _CategorizationInputSchema,
                      CategorizationParsSchema, CategorizationPostSchema,
                      CategorizationPredictSchema, ClusteringSchema,
                      _CategorizationIndex, ErrorSchema,
                      _CategorizationPredictSchemaElement,
                      MetricsCategorizationSchema, MetricsClusteringSchema,
                      MetricsDupDetectionSchema,
                      EmailThreadingSchema, EmailThreadingParsSchema,
                      SearchResponseSchema,
                      EmptySchema,
                      CustomStopWordsSchema, CustomStopWordsLoadSchema
                      )

EPSILON = 1e-3  # small numeric value

# =========================================================================== #
#                        Datasets download                                    #
# =========================================================================== #


class ExampleDatasetApi(Resource):

    @use_args({'n_categories': wfields.Int(missing=2)})
    @doc(description=_docstring_description(dedent(load_dataset.__doc__)))
    @marshal_with(ExampleDatasetSchema())
    def get(self, name, **args):
        n_categories = args['n_categories']
        n_categories = min(3, n_categories)
        n_categories = max(1, n_categories)

        categories = None
        if "20_newsgroups_" in name:
            if n_categories == 3:
                categories = ['comp.graphics', 'rec.sport.baseball',
                              'sci.space']
            elif n_categories == 2:
                categories = ['comp.graphics', 'rec.sport.baseball']
            elif n_categories == 1:
                categories = ['comp.graphics']

        md, training_set, test_set = load_dataset(name, self._cache_dir,
                                                  verbose=True,
                                                  verify_checksum=False,
                                                  categories=categories)

        return {'metadata': md, 'training_set': training_set,
                'dataset': test_set}


# Definine the response formatting schemas
id_schema = IDSchema()
features_schema = FeaturesSchema()
error_schema = ErrorSchema()

# ========================================================================== #
#                    Feature extraction                                      #
# ========================================================================== #


class FeaturesApi(Resource):

    @doc(description='View parameters used for the feature extraction')
    @marshal_with(FeaturesSchema(many=True))
    def get(self):
        fe = FeatureVectorizer(self._cache_dir)
        return fe.list_datasets()

    @doc(description=dedent("""
            Initialize the feature extraction on a document collection.

            **Parameters**
             - `n_features`: [optional] number of features (overlapping character/word n-grams that are hashed).  n_features refers to the number of buckets in the hash.  The larger the number, the fewer collisions.   (default: 1100000)
             - `analyzer`: 'word', 'char', 'char_wb' Whether the feature should be made of word or character n-grams.  Option ‘char_wb’ creates character n-grams only from text inside word boundaries.  ( default: 'word')
             - `ngram_range` : tuple (min_n, max_n), default=(1, 1) The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.

             - `stop_words`: "english" or "None" Remove stop words from the resulting tokens. Only applies for the "word" analyzer.  If "english", a built-in stop word list for English is used. ( default: "english")
             - `n_jobs`: The maximum number of concurrently running jobs (default: 1)
             - `chuck_size`: The number of documents simultaneously processed by a running job (default: 5000)
             - `weighting`: the SMART notation for document term weighting and normalization.  In the form [nlabL][ntp][ncb] , see https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System
             - `norm_alpha`: the alpha value used for pivoted normalization

             - `use_hashing`: Enable hashing. This option must be set to True for classification and set to False for clustering. (default: True)
             - `min_df`: When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is ignored when hashing is used.
             - `max_df`: When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold. This value is ignored when hashing is used.
             - `parse_email_headers`: when documents are emails, attempt to parse the information contained in the header (default: False)
             - `preprocess`: a list of pre-processing steps to apply before vectorization. A subset of ['emails_ignore_header'], default: [].
             - `id`: (optional) custom dataset id. Can only contain letters, numbers, "_" or "-". It must also be between 2 and 50 characters long.
             - `overwrite`: if a custom dataset id was provided, and it already exists, overwrite it. Default: false
            """))
    @use_args(FeaturesParsSchema(strict=True, exclude=('data_dir')))
    @marshal_with(IDSchema())
    def post(self, **args):
        if args['use_hashing']:
            for key in ['min_df', 'max_df']:
                if key in args:
                    # the above parameters are ignored with caching
                    del args[key]
        for key in ['min_df', 'max_df']:
            if key in args and args[key] > 1. + EPSILON:  # + eps
                args[key] = int(args[key])
        dsid = args.pop('id', None)
        overwrite = args.pop('overwrite', None)

        if dsid is None:
            mode = 'w'
        elif not overwrite:
            mode = 'w'
        else:
            mode = 'fw'

        fe = FeatureVectorizer(self._cache_dir, mode=mode, dsid=dsid)
        dsid = fe.setup(**args)
        return {'id': dsid}


class FeaturesApiElement(Resource):
    @doc(description="Load extracted features "
                     "(and obtain the processing status)")
    def get(self, dsid):
        sc = FeaturesSchema()
        fe = FeatureVectorizer(self._cache_dir, dsid=dsid)
        out = fe.pars_.copy()
        is_processing = (fe.cache_dir / dsid / 'processing').exists()
        is_finished = (fe.cache_dir / dsid / 'processing_finished').exists()
        if is_processing and not is_finished:
            n_chunks = len((fe.cache_dir / dsid).glob('features-*[0-9]'))
            out['n_samples_processed'] = min(n_chunks*out['chunk_size'],
                                             out['n_samples'])
            return sc.dump(out).data, 202
        elif not is_processing and is_finished:
            out['n_samples_processed'] = out['n_samples']
            out['filenames'] = fe.filenames_
            return sc.dump(out).data, 200
        else:
            return error_schema.dump({"message":
                                     "Processing failed, see server logs!"
                                      }).data, 520

    @doc(description=dedent("""
         Run feature extraction on a dataset,

         **Parameters**
          - `data_dir`: [optional] relative path to the directory with the input files. Either `data_dir` or `dataset_definition` must be provided.
          - `dataset_definition`: [optional] a list of dictionaries `[{'file_path': <str>, 'content': <str>, 'document_id': <int>, 'rendition_id': <int>}, ...]` where `document_id` and `rendition_id` are optional, while either `file_path` or `content` field must be provided. 
          - `vectorize`: [optional] this option can be used to ingest the dataset_definition in batches (optionally with document content), then make one final call to vectorize all sent documents (bool, default: True)
          - `document_id_generator`: [optional] if the `document_id` is not provided, this specifies how it is generated. If `indexed_file_path` the `document_id` is given by the index of the sorted `file_path`, otherwise if `infer_file_path` the `document_id` is inferred from the `file_path` strings, removing all non digit characters. In this second case, the `file_path` must contain a unique numeric ID (default: `indexed_file_path`)
         """))
    @use_args({"data_dir":  wfields.Str(),
               "dataset_definition": wfields.Nested(_DatasetDefinition, many=True),
               "vectorize": wfields.Bool(missing=True),
               "document_id_generator": wfields.Str(missing="indexed_file_path")})
    @marshal_with(IDSchema())
    def post(self, dsid, **args):
        fe = FeatureVectorizer(self._cache_dir, dsid=dsid, mode='r')
        fe.ingest(**args)
        if fe.pars_['parse_email_headers']:
            fe.parse_email_headers()
        return {'id': fe.dsid}

    @doc(description='Delete a processed dataset')
    @marshal_with(EmptySchema())
    def delete(self, dsid):
        fe = FeatureVectorizer(self._cache_dir, dsid=dsid)
        fe.delete()
        return {}


class FeaturesApiElementMappingNested(Resource):
    @doc(description=dedent("""
         Compute correspondence between id fields for documents.
         At least one of the fields used for indexing must be provided,
         and all the rest will be computed (if available).
         If the data parameter is not provided, return all the correspondence table

         **Parameters**
          - `data`: the ids of documents used as the query
          - `return_file_path`: whether the results should include the file path
         """))
    @use_args({'data': wfields.Nested(DocumentIndexFullSchema, many=True),
               'return_file_path': wfields.Bool(missing=True)})
    @marshal_with(DocumentIndexNestedSchema())
    def post(self, dsid, **args):
        fe = FeatureVectorizer(self._cache_dir, dsid=dsid)
        if 'data' in args and args['data']:
            query = pd.DataFrame(args['data'])
            fe.db_.filenames_ = fe.filenames_
            res = fe.db_.search(query, return_file_path=args['return_file_path'])
        else:
            res = None
        if args['return_file_path']:
            fe.db_.filenames_ = fe.filenames_
        res_repr = fe.db_.render_dict(res, return_file_path=args['return_file_path'])
        return {'data': res_repr}


class FeaturesApiAppend(Resource):
    @doc(description=dedent("""
         Add new documents to an existing processed dataset.
         This will also automatically update the LSI model if any
         is present. Raw documents on disk are not affected.

         This operation cannot be undone.

         Warning: all categorization, clustering, duplicate detection and
         email threading models associated with this dataset will be removed and
         need to be re-trained.

         **Parameters**
          - `data_dir`: [optional] relative path to the directory with the input files. Either `data_dir` or `dataset_definition` must be provided.
          - `dataset_definition`: [optional] a list of dictionaries `[{'file_path': <str>, 'document_id': <int>, 'rendition_id': <int>}, ...]` where  `rendition_id` are optional, while either `file_path` or `content` field must be provided. 
          """))
    @use_args({'data_dir': wfields.Str(),
               'dataset_definition': wfields.Nested(_DatasetDefinition, many=True,
                                                    required=True)})
    @marshal_with(EmptySchema())
    def post(self, dsid, **args):
        fe = FeatureVectorizer(self._cache_dir, dsid=dsid)
        fe.append(**args)
        return {}


class FeaturesApiRemove(Resource):
    @doc(description=dedent("""Remove documents from an existing processed dataset.
         This will also automatically update the LSI model if any
         is present.  Raw documents on disk are not affected.

         This operation cannot be undone.

         Warning: all categorization, clustering, duplicate detection and
         email threading models associated with this dataset will be removed and
         need to be re-trained.

         **Parameters**
          - `dataset_definition`: [optional] a list of dictionaries `[{'file_path': <str>, 'document_id': <int>, 'rendition_id': <int>}, ...]` where  `rendition_id` are optional.
          """))
    @use_args({'dataset_definition': wfields.Nested(_DatasetDefinitionShort,
                                                    many=True, required=True)})
    @marshal_with(EmptySchema())
    def post(self, dsid, **args):
        fe = FeatureVectorizer(self._cache_dir, dsid=dsid, mode='r')
        fe.remove(args['dataset_definition'])
        return {}


# =========================================================================== #
#                 LSI decomposition
# =========================================================================== #


class LsiApi(Resource):

    @doc(description='List existing LSI models')
    @use_args({'parent_id': wfields.Str(required=True)})
    @marshal_with(LsiParsSchema(many=True))
    def get(self, **args):
        parent_id = args['parent_id']
        lsi = _LSIWrapper(cache_dir=self._cache_dir, parent_id=parent_id,
                          random_state=self._random_seed)
        return lsi.list_models()

    @doc(description=dedent("""
           Build a Latent Semantic Indexing (LSI) model

           Recommended data ingestion options also include, `use_idf=1, sublinear_tf=0, binary=0`.

           The recommended value for the `n_components` (dimensions of the SVD decompositions) is
           in the [100, 200] range.

           **Parameters**
             - `n_components`: Desired dimensionality of the output data. Must be strictly less than the number of features.
             - `parent_id`: parent dataset identified by `dataset_id`
             - `alpha`: floor on the number of components used with small datasets
             - `id`: (optional) custom model id. Can only contain letters, numbers, "_" or "-". It must also be between 2 and 50 characters long.
             - `overwrite`: if a custom model id was provided, and it already exists, overwrite it. Default: false

          """))
    @use_args({'parent_id': wfields.Str(required=True),
               'n_components': wfields.Int(missing=150),
               'alpha': wfields.Number(missing=0.33),
               'id': wfields.Str(),
               'overwrite': wfields.Boolean(missing=False)})
    @marshal_with(LsiPostSchema())
    def post(self, **args):
        parent_id = args.pop('parent_id')

        mid = args.pop('id', None)
        overwrite = args.pop('overwrite', None)
        if mid is None:
            mode = 'w'
        elif not overwrite:
            mode = 'w'
        else:
            mode = 'fw'

        lsi = _LSIWrapper(cache_dir=self._cache_dir, parent_id=parent_id,
                          random_state=self._random_seed, mid=mid, mode=mode)
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

# =========================================================================== #
#                 Categorization (ML)
# =========================================================================== #


class ModelsApi(Resource):
    @doc(description='List existing categorization models')
    @marshal_with(CategorizationParsSchema(many=True))
    def get(self, parent_id):
        cat = _CategorizerWrapper(parent_id, self._cache_dir)

        return cat.list_models()

    @doc(description=dedent("""
           Build the categorization ML model

           The option `use_hashing=True` must be set for the feature extraction. Recommended options also include, `weighting="ntc"`.

           **Parameters**
            - `parent_id`: `dataset_id` or `lsi_id`
            - `data`: a list of dict which have a `category` field and one or several fields that can be used for indexing, such as `document_id` and optionally `rendition_id`.
            - `method`: classification algorithm to use (default: LogisticRegression),
              * "LogisticRegression": [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
              * "LinearSVC": [Linear SVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html),
              * "NearestNeighbor": nearest neighbor classifier (requires LSI)
            - `cv`: binary, if true optimal parameters of the ML model are determined by cross-validation over 5 stratified K-folds (default False).
            - `training_scores`: binary, compute the efficiency scores on the training dataset. This would make computations much slower for NearestNeighbors (default False). 
          """))
    @use_args(_CategorizationInputSchema())
    @marshal_with(CategorizationPostSchema())
    def post(self, **args):
        training_scores = args['training_scores']
        parent_id = args['parent_id']
        cat = _CategorizerWrapper(self._cache_dir, parent_id=parent_id,
                                  random_state=self._random_seed)

        query = pd.DataFrame(args['data'])
        res_q = cat.fe.db_.search(query, drop=False)
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
             - `sort_by` : if provided and not None, the field used for sorting results. Valid values are [None, 'score'] or any of the ingested category names.
             - `sort_order`: the sort order (if applicable), one of ['ascending', 'descending']
             - `max_results` : return only the first `max_results` documents. If `max_results <= 0` all documents are returned.
             - `ml_output` : type of the output in ['decision_function', 'probability'], only affects ML methods.
             - `metric` : The similarity returned by nearest neighbor classifier in ['cosine', 'jaccard', 'cosine-positive'].
             - `min_score` : filter out results below a similarity threashold
             - `subset`: apply prediction to a document subset. Must be one of ['all', 'train', 'test']. Default: 'test'.
             - `subset_document_id`: apply prediction to a subset of document_id. 
             - `batch_id`: retrieve a given subset of scores (-1 to retrieve all). Default: 0
             - `batch_size`: the number of document scores retrieved per batch. Default: 10000
            """))
    @use_args({'max_result_categories': wfields.Int(missing=1),
               'sort_by': wfields.Str(missing='score'),
               'sort_order': wfields.Str(missing='descending',
                                         validate=_is_in_range(['descending',
                                                                'ascending'])),
               'max_results': wfields.Int(),
               'ml_output': wfields.Str(missing='probability'),
               'metric': wfields.Str(missing='cosine'),
               'min_score': wfields.Number(missing=-1),
               'subset': wfields.Str(missing='test',
                                     validate=_is_in_range(['all', 'train',
                                                            'test'])),
               'subset_document_id': wfields.List(wfields.Int()),
               'batch_id': wfields.Int(missing=0),
               'batch_size': wfields.Int(missing=10000),
               })
    @marshal_with(CategorizationPredictSchema())
    def get(self, mid, **args):

        valid_sort = ['score']

        sort_by = args.pop('sort_by')
        sort_ascending = args.pop('sort_order') == 'ascending'
        max_result_categories = args.pop('max_result_categories')
        min_score = args.pop("min_score")
        max_results = args.pop("max_results", 0)
        subset = args.pop("subset")
        subset_document_id = args.pop('subset_document_id', None)
        batch_size = args.pop("batch_size")
        batch_id = args.pop("batch_id")

        cat = _CategorizerWrapper(self._cache_dir, mid=mid)
        y_res, nn_res = cat.predict(**args)
        train_indices = cat._pars['index']

        labels = list(cat.le.classes_)
        Y_pred = y_res

        if max_result_categories <= 0:
            raise WrongParameter(('the max_result_categories={} '
                                  'must be strictly positive')
                                 .format(max_result_categories))
        if sort_by not in valid_sort + labels:
            raise WrongParameter(("sort_by={} not value. Must be "
                                  "one of {}")
                                 .format(sort_by, valid_sort + labels))

        id_mapping = cat.fe.db_.data
        # have to cast to object as otherwise
        # we get serializing np.int64 issues...
        base_keys = [key for key in id_mapping.columns
                     if key in ['internal_id', 'document_id', 'rendition_id']]
        id_mapping = id_mapping[base_keys].set_index('internal_id', drop=True).astype('object')
        # create dataframe out of results
        Y_pred = id_mapping.copy()
        for idx, el in enumerate(labels):
            Y_pred[el] = y_res[:, idx]
        if nn_res is not None:
            nn_range = np.arange(nn_res.shape[0])
            NN_map = []
            for idx, el in enumerate(labels):
                NN_map_el = pd.DataFrame({'internal_id': nn_res[:, idx]},
                                         index=nn_range)
                NN_map_el = NN_map_el.join(id_mapping, on='internal_id',
                                           how='left')
                NN_map.append(NN_map_el)
        else:
            NN_map = None

        # optionally filter out test or training set
        if subset_document_id is not None:
            Y_pred = Y_pred.reset_index().set_index('document_id', drop=False)
            subset_document_id = np.array(subset_document_id)
            subset_mask = np.in1d(Y_pred.index, subset_document_id)
            Y_pred = Y_pred.iloc[subset_mask].set_index('internal_id')
        elif subset in ['train', 'test']:
            _mask = np.in1d(Y_pred.index.values, train_indices)
            if subset == 'test':
                _mask = ~_mask
            Y_pred = Y_pred.loc[_mask, :]

        # sort output
        if sort_by in labels:
            Y_pred = Y_pred.sort_values(sort_by, ascending=sort_ascending)

        Y_pred_max = Y_pred[labels].max(axis=1)
        if sort_by in valid_sort:
            if sort_by == 'score':
                Y_pred_max = Y_pred_max.sort_values(ascending=sort_ascending)

        # filter out low scores
        if min_score is not None:
            Y_pred_max = Y_pred_max[Y_pred_max > min_score]

        Y_pred = Y_pred.loc[Y_pred_max.index.values, :]

        # return only first N results
        if max_results > 0:
            Y_pred = Y_pred.iloc[:max_results]

        Y_pred, pagination = _paginate(Y_pred, batch_id, batch_size)

        # render NN mapping
        if NN_map is not None:
            for idx, NN_map_el in enumerate(NN_map):
                NN_map[idx] = NN_map_el.loc[Y_pred.index, :].to_dict(orient='records')

        res = Y_pred.to_dict(orient='records')

        # convert scores to a more usable format
        for idx, row in enumerate(res):
            scores = []
            for label_id, key in enumerate(labels):
                scores_el = {'category': key,
                             'score': row.pop(key)}
                if NN_map is not None:
                    scores_el.update(NN_map[label_id][idx])
                scores.append(scores_el)
            row['scores'] = scores

        def sort_func(scores):
            return -scores['score']

        # sort categories and return only the first max_result_categories
        for idx, row in enumerate(res):
            row['scores'] = sorted(row['scores'], key=sort_func)[:max_result_categories]

        return {'data': res, 'pagination': pagination}

# =========================================================================== #
#                             Clustering
# =========================================================================== #


class KmeanClusteringApi(Resource):

    @doc(description=dedent("""
           Compute K-mean clustering

           The option `use_hashing=False` must be set for the feature extraction. Recommended options for feature extraction include, `weighting="ntc"`.

           **Parameters**
            - `parent_id`: `dataset_id` or `lsi_id`
            - `n_clusters`: the number of clusters
            - `metric` : The similarity returned by nearest neighbor classifier in ['cosine', 'jaccard', 'cosine-positive'].
           """))
    @use_args({'parent_id': wfields.Str(required=True),
               'n_clusters': wfields.Int(missing=150),
               'metric': wfields.Str(missing='cosine')})
    @marshal_with(IDSchema())
    def post(self, **args):
        metric = args.pop('metric')

        cl = _ClusteringWrapper(cache_dir=self._cache_dir,
                                parent_id=args['parent_id'],
                                metric=metric)

        del args['parent_id']

        cl.k_means(**args)
        return {'id': cl.mid}


class BirchClusteringApi(Resource):

    @doc(description=dedent("""
           Compute birch clustering

           The option `use_hashing=False` must be set for the feature extraction. Recommended options for data ingestion also include, `ntc`.

           **Parameters**
            - `parent_id`: `dataset_id` or `lsi_id`
            - `n_clusters`: the number of clusters or -1 to use hierarchical clustering (default: -1)
            - `min_similarity`: The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold. Otherwise a new subcluster is started. See [sklearn.cluster.Birch](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html). Increasing this value would increase the hierarchical tree depth (and the number of clusters).
            - `branching_factor`: Maximum number of CF subclusters in each node. If a new samples enters such that the number of subclusters exceed the branching_factor then the node has to be split. The corresponding parent also has to be split and if the number of subclusters in the parent is greater than the branching factor, then it has to be split recursively. Decreasing this value would increase the number of clusters.
            - `max_tree_depth` : Maximum hierarchy depth (only applicable when `n_clusters=-1`)
            - `metric` : The similarity returned by nearest neighbor classifier in ['cosine', 'jaccard', 'cosine-positive'].
           """))
    @use_args({
            'parent_id': wfields.Str(required=True),
            'n_clusters': wfields.Int(missing=-1),
            'branching_factor': wfields.Int(missing=20),
            'max_tree_depth': wfields.Int(),
            # this corresponds approximately to threashold = 0.5
            'min_similarity': wfields.Number(missing=0.5),
            'metric': wfields.Str(missing='cosine')
            }
            )
    @marshal_with(IDSchema())
    def post(self, **args):
        from math import sqrt

        metric = args.pop('metric')
        S_cos = _scale_cosine_similarity(args.pop('min_similarity'),
                                         metric=metric,
                                         inverse=True)
        # cosine sim to euclidean distance
        threshold = sqrt(2 * (1 - S_cos))

        cl = _ClusteringWrapper(cache_dir=self._cache_dir,
                                parent_id=args.pop('parent_id'),
                                metric=metric)

        if args.get('n_clusters') <= 0:
            args['n_clusters'] = None

        cl.birch(threshold=threshold, **args)
        return {'id': cl.mid}


class DBSCANClusteringApi(Resource):

    @doc(description=dedent("""
           Compute clustering (DBSCAN)

           The option `use_hashing=False` must be set for the feature extraction. Recommended options for the data ingestion also include, `weighting="ntc"`.

           **Parameters**
             - `parent_id`: `dataset_id` or `lsi_id`
             - `min_similarity`: The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold. Otherwise a new subcluster is started. See [sklearn.cluster.Birch](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html)
             - `metric` : The similarity returned by nearest neighbor classifier in ['cosine', 'jaccard', 'cosine-positive'].
             - `min_samples`: (optional) int The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
            """))
    @use_args({'parent_id': wfields.Str(required=True),
               'min_samples': wfields.Int(missing=10),
               # this corresponds approximately to threashold = 0.5
               'min_similarity': wfields.Number(missing=0.5),
               'metric': wfields.Str(missing='cosine')})
    @marshal_with(IDSchema())
    def post(self, **args):
        from math import sqrt
        metric = args.pop('metric')
        S_cos = _scale_cosine_similarity(args.pop('min_similarity'),
                                         metric=metric,
                                         inverse=True)
        # cosine sim to euclidean distance
        eps = sqrt(2 * (1 - S_cos))

        cl = _ClusteringWrapper(cache_dir=self._cache_dir,
                                parent_id=args.pop('parent_id'),
                                metric=metric)

        cl.dbscan(eps=eps, **args)
        return {'id': cl.mid}


class ClusteringApiElement(Resource):

    @doc(description=dedent("""
           Compute cluster labels

           **Parameters**
            - `n_top_words`: keep only most relevant `n_top_words` words
            - `return_optimal_sampling` : Instead of cluster results, the optimal sampling results will be returned (with no cluster labels). This option is only valid with Birch algorithm. Note that optimal sampling cannot return more samples than the subclusters in the birch clustering results (default: false)
            - `sampling_min_similarity` : Similarity threashold used by smart sampling. Decreasing this value would result in more sampled documents. Default: 1.0 (i.e. use the full cluster hierarichy).
            - `sampling_min_coverage` : Minimal coverage requirement in [0, 1] range. Increasing this value would result in a larger number of samples. (default: 0.9)
            """))
    @use_args({'n_top_words': wfields.Int(missing=5),
               'return_optimal_sampling': wfields.Bool(missing=False),
               'sampling_min_similarity': wfields.Number(missing=1.0),
               'sampling_min_coverage': wfields.Number(missing=0.9)})
    @marshal_with(ClusteringSchema())
    def get(self, method, mid, **args):
        return_optimal_sampling = args.pop('return_optimal_sampling')
        sampling_min_coverage = args.pop('sampling_min_coverage')
        sampling_min_similarity = args.pop('sampling_min_similarity')

        cl = _ClusteringWrapper(cache_dir=self._cache_dir, mid=mid)
        km = cl._load_model()

        if return_optimal_sampling and not cl._pars['is_hierarchical']:
            raise WrongParameter(('Model {} does not support optimal sampling,'
                                  'please use hierarchical Birch clustering '
                                  '(with n_clusters=-1)')
                                 .format(type(km).__name__))

        cl._fit_X = cl.pipeline.data

        if type(km).__name__ in ['_BirchDummy', 'Birch'] and cl._pars['is_hierarchical']:
            # Hierarchical clustering

            htree = cl._load_htree()

            db = cl.fe.db_.data

            if return_optimal_sampling:
                # cut the hierarchical tree to match the smart sampling
                flat_tree = compute_optimal_sampling(htree,
                                                     sampling_min_similarity,
                                                     sampling_min_coverage)
            else:
                # we don't use optimal sampling
                if cl._pars['max_tree_depth'] is not None:
                    max_tree_depth = cl._pars['max_tree_depth']
                    htree.limit_depth(max_tree_depth)
                flat_tree = htree.flatten()

                terms = cl.compute_labels(
                            cluster_indices=[row['document_id_accumulated']
                                             for row in flat_tree],
                            **args)
                for label, row in zip(terms, flat_tree):
                    row['cluster_label'] = label

            res = []
            doc_keys = [key for key in db.columns
                        if key in ['document_id', 'rendering_id']]
            db = db[doc_keys]

            for row in flat_tree:
                irow = {'cluster_similarity': row['cluster_similarity'],
                        'cluster_size': row['cluster_size'],
                        'cluster_id': row['cluster_id']}
                if not return_optimal_sampling:
                    irow['cluster_label'] = ' '.join(row['cluster_label'])
                    irow['children'] = [el['cluster_id']
                                        for el in row.children]
                    irow['cluster_depth'] = row.current_depth

                db_sl = db.iloc[row['document_id_accumulated']].copy()
                db_sl['similarity'] = row['document_similarity']
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

                S_sim_mean, S_sim = centroid_similarity(cl._fit_X,
                                                        group.index.values,
                                                        cl._pars['metric'])
                group = group.assign(similarity=S_sim)

                row_docs = []
                for idx, row in group.iterrows():
                    row_docs.append({key: val
                                     for key, val in row.to_dict().items()
                                     if key in valid_keys})
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

# =========================================================================== #
#                             Duplicate detection
# =========================================================================== #


class DupDetectionApi(Resource):

    @doc(description=dedent("""
           Compute near duplicates

           **Parameters**
            - `parent_id`: `dataset_id` or `lsi_id`
            - `method`: str, default='simhash' Method used for duplicate detection. One of "simhash", "i-match"
          """))
    @use_args({"parent_id": wfields.Str(required=True),
               "method": wfields.Str(required=False, missing='simhash')})
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
            - `metric` : The similarity returned by nearest neighbor classifier in ['cosine', 'jaccard', 'cosine-positive'].
          """))
    @use_args({'distance': wfields.Int(),
               'n_rand_lexicons': wfields.Int(),
               'rand_lexicon_ratio': wfields.Number(),
               'metric': wfields.Str(missing='cosine')
               })
    @marshal_with(ClusteringSchema())
    def get(self, mid, **args):

        metric = args.pop('metric')

        model = _DuplicateDetectionWrapper(cache_dir=self._cache_dir, mid=mid)
        cluster_id = model.query(**args)
        model._fit_X = model.pipeline.data  # load the data
        y = model._merge_response(cluster_id)
        res = []
        valid_keys = ['document_id', 'rendering_id', 'similarity']
        for name, group in y.groupby('cluster_id'):
            if group.shape[0] <= 1:
                continue

            S_sim_mean, S_sim = centroid_similarity(model._fit_X,
                                                    group.index.values,
                                                    metric)
            group = group.assign(similarity=S_sim)

            row_docs = []
            for idx, row in group.iterrows():
                row_docs.append({key: val for key, val in row.to_dict().items()
                                 if key in valid_keys})
            row['documents'] = row_docs
            res.append({'documents': row_docs, 'cluster_id': name,
                        'cluster_similarity': S_sim_mean})

        return {'data': res}

    @marshal_with(EmptySchema())
    def delete(self, mid):

        model = _DuplicateDetectionWrapper(cache_dir=self._cache_dir, mid=mid)
        model.delete()
        return {}

# =========================================================================== #
#                            Metrics                                          #
# =========================================================================== #


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
    @use_args({'y_true': wfields.Nested(_CategorizationIndex,
                                        many=True, required=True),
               'y_pred': wfields.Nested(_CategorizationPredictSchemaElement,
                                        many=True, required=True),
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

        # wrapping metrics calculations, as for example F1 score
        # can frequently print warnings "F-score is ill defined
        # and being set to 0.0 due to no predicted samples"
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
                if name in ['roc_auc', 'average_precision', 'recall_at_k'] \
                        and n_classes == 2:
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


_metrics_clustering_api_get_args = {
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
    # Arguments are the same as for clustering
    @use_args(_metrics_clustering_api_get_args)
    @marshal_with(MetricsDupDetectionSchema())
    def post(self, **args):
        output_metrics = dict()
        labels_true = args['labels_true']
        labels_pred = args['labels_pred']
        metrics = args['metrics']

        # Methods 'ratio_duplicates_score' and 'f1_same_duplicates_score'
        # in ..metrics.py
        # accept Numpy array objects, not standard Python lists
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            if 'ratio_duplicates' in metrics:
                output_metrics['ratio_duplicates'] = \
                   ratio_duplicates_score(np.array(labels_true),
                                          np.array(labels_pred))
            if 'f1_same_duplicates' in metrics:
                output_metrics['f1_same_duplicates'] = \
                   f1_same_duplicates_score(np.array(labels_true),
                                            np.array(labels_pred))
            if 'mean_duplicates_count' in metrics:
                output_metrics['mean_duplicates_count'] = \
                    mean_duplicates_count_score(labels_true, labels_pred)
        return output_metrics

# =========================================================================== #
#                             Email threading
# =========================================================================== #


class EmailThreadingApi(Resource):

    @doc(description='Compute email threading')
    @use_args({"parent_id": wfields.Str(required=True)})
    @marshal_with(EmailThreadingSchema())
    def post(self, **args):

        model = _EmailThreadingWrapper(cache_dir=self._cache_dir,
                                       parent_id=args['parent_id'])

        tree = model.thread()

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


# =========================================================================== #
#                             (Semantic) search
# =========================================================================== #


class SearchApi(Resource):
    @doc(description=dedent("""
            Perform document search (if `parent_id` is a `dataset_id`) or a semantic search (if `parent_id` is a `lsi_id`).

            Parameters
            ----------
            - `parent_id` : the id of the previous processing step (either `dataset_id` or `lsi_id`)
            - `query` : the seach query. Either `query` or `query_document_id` must be provided.
            - `query_document_id` : the id of the document used as the search query. Either `query` or `query_document_id` must be provided.
            - `metric` : The similarity returned by nearest neighbor classifier in ['cosine', 'jaccard', 'cosine-positive'].
            - `min_score` : filter out results below a similarity threashold
            - `max_results` : return only the first `max_results` documents. If `max_results <= 0` all documents are returned.
            - `sort_by` : if provided and not None, the field used for sorting results. Valid values are [None, 'score']
            - `sort_order`: the sort order (if applicable), one of ['ascending', 'descending']
            - `batch_id`: retrieve a given subset of scores (-1 to retrieve all). Default: 0
            - `batch_size`: the number of document scores retrieved per batch. Default: 10000
             - `subset_document_id`: apply prediction to a subset of document_id. 
            """))
    @use_args({"parent_id": wfields.Str(required=True),
               "query": wfields.Str(),
               "query_document_id": wfields.Int(),
               'metric': wfields.Str(missing='cosine'),
               'min_score': wfields.Number(missing=-1),
               'max_results': wfields.Int(),
               'sort_by': wfields.Str(missing='score'),
               'sort_order': wfields.Str(missing='descending',
                                         validate=_is_in_range(['descending',
                                                                'ascending'])),
               'batch_id': wfields.Int(missing=0),
               'batch_size': wfields.Int(missing=10000),
               'subset_document_id': wfields.List(wfields.Int()),
               })
    @marshal_with(SearchResponseSchema())
    def post(self, **args):
        parent_id = args['parent_id']
        subset_document_id = args.pop('subset_document_id', None)
        model = _SearchWrapper(cache_dir=self._cache_dir, parent_id=parent_id)

        if 'query' in args and 'query_document_id' not in args:
            query = args['query']
            scores = model.search(query, metric=args['metric'])
        elif 'query' not in args and 'query_document_id' in args:
            query = pd.DataFrame([{'document_id': args['query_document_id']}])
            res_q = model.fe.db_.search(query, drop=False)

            scores = model.search(None,
                                  internal_id=res_q.internal_id.values[0],
                                  metric=args['metric'])
        else:
            raise WrongParameter("One of the 'query', "
                                 "'query_document_id' must be provided")

        scores_pd = pd.DataFrame({'score': scores,
                                  'internal_id': np.arange(
                                       model.fe.n_samples_, dtype='int')})
        scores_pd = scores_pd[scores_pd.score > args['min_score']]

        if 'query' not in args and 'query_document_id' in args:
            # remove the query document
            scores_pd = scores_pd[scores_pd.internal_id != res_q.internal_id.values[0]]

        if subset_document_id is not None:
            db = model.fe.db_.data[['document_id', 'internal_id']].set_index('internal_id')
            scores_pd = scores_pd.set_index('internal_id', drop=False)
            scores_pd = scores_pd.join(db, how='left')
            scores_pd = scores_pd.set_index('document_id', drop=False)
            subset_document_id = np.array(subset_document_id)
            subset_mask = np.in1d(scores_pd.index, subset_document_id)
            scores_pd = scores_pd.iloc[subset_mask].set_index('internal_id', drop=False)

        sort_by = args['sort_by']
        if sort_by:
            if sort_by not in scores_pd.columns:
                raise WrongParameter(
                    'sort_by={} not in {}'.format(sort_by,
                                                  list(scores_pd.columns)))
            is_ascending = args['sort_order'] == 'ascending'
            scores_pd.sort_values(by=sort_by, inplace=True,
                                  ascending=is_ascending)

        if 'max_results' in args and args['max_results'] > 0:
            scores_pd = scores_pd.iloc[:args['max_results'], :]

        # make the pagination happen
        scores_batch, pagination = _paginate(scores_pd, args['batch_id'],
                                             args['batch_size'])

        res = model.fe.db_.render_dict(scores_batch)

        return {'data': res, 'pagination': pagination}


# =========================================================================== #
#                            Custom Stop Words
# =========================================================================== #


class CustomStopWordsApi(Resource):
    @doc(description="Store a list of custom stop words")
    @use_args({"name": wfields.Str(required=True),
               "stop_words": wfields.List(wfields.Str(), required=True)})
    @marshal_with(CustomStopWordsSchema())
    def post(self, **args):
        name = args['name']
        stop_words = args['stop_words']
        model = _StopWordsWrapper(cache_dir=self._cache_dir)
        model.save(name=name, stop_words=stop_words)
        return {'name': name}


class CustomStopWordsLoadApi(Resource):

    @doc(description="Load a stored list of stop words")
    @marshal_with(CustomStopWordsLoadSchema())
    def get(self, name):
        return {'name': name,
                'stop_words': _StopWordsWrapper(
                    cache_dir=self._cache_dir).load(name)}

    @doc(description='Delete a stored custom stop words')
    @marshal_with(EmptySchema())
    def delete(self, name):
        _StopWordsWrapper(cache_dir=self._cache_dir).delete(name)
        return {}
