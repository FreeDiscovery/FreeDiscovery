# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from marshmallow import Schema, fields


# This file defines marshmallow schemas for REST API output formatting
# Occasionally the same class can be used for input validation
# As webargs (used for input validation) is just a wrapper around marshmallow

class DatasetSchema(Schema):
    base_dir = fields.Str(required=True)
    data_dir = fields.Str(required=True)
    ground_truth_file = fields.Str()
    seed_filenames = fields.List(fields.Str())
    seed_y = fields.List(fields.Int())

class IDSchema(Schema):
    id = fields.Str(required=True)


class FeaturesParsSchema(Schema):
    data_dir = fields.Str(required=True)
    dir_pattern = fields.Str()
    n_features = fields.Int(missing=100000)
    analyzer = fields.Str(missing='word')
    stop_words = fields.Str()
    n_jobs = fields.Int(missing=1)
    chunk_size = fields.Int()
    ngram_range = fields.List(fields.Int(), missing=[1,1])
    use_idf = fields.Boolean(missing=False)
    sublinear_tf = fields.Boolean(missing=False)
    binary = fields.Boolean(missing=True)
    use_hashing = fields.Boolean(missing=True)
    norm = fields.Str(missing='None')
    n_samples = fields.Int(dump_only=True)
    n_samples_processed = fields.Int(dump_only=True)
    min_df = fields.Number(required=False)
    max_df = fields.Number(required=False)

    class Meta:
        strict = True


class FeaturesSchema(FeaturesParsSchema):
    id = fields.Str(required=True)
    filenames = fields.List(fields.Str())


class FeaturesElementIndexSchema(Schema):
    index = fields.List(fields.Int(), required=True)


class EmailParserSchema(FeaturesParsSchema):
    id = fields.Str(required=True)
    filenames = fields.List(fields.Str())

class EmailParserElementIndexSchema(Schema):
    index = fields.List(fields.Int(), required=True)

# TODO to delete after successful implementation of metrics
class ClassificationScoresSchema(Schema):
    precision = fields.Number(required=True)
    recall = fields.Number(required=True)
    f1 = fields.Number(required=True)
    roc_auc = fields.Number(required=True)
    average_precision = fields.Number(required=True)


class LsiParsSchema(Schema):
    parent_id = fields.Str(required=True)
    n_components = fields.Int(required=True)


class LsiPostSchema(IDSchema):
    explained_variance = fields.Number(required=True)


class CategorizationPostSchema(ClassificationScoresSchema):
    id = fields.Str(required=True)


class CategorizationPredictSchema(Schema):
    prediction = fields.List(fields.Number, required=True)
    dist_p = fields.List(fields.Number())
    dist_n = fields.List(fields.Number())
    ind_p = fields.List(fields.Int())
    ind_n = fields.List(fields.Int())


class CategorizationParsSchema(Schema):
    method = fields.Str(required=True)
    index = fields.List(fields.Int(), required=True)
    y = fields.List(fields.Int(), required=True)
    options = fields.Str(required=True)


class _HTreeSchema(Schema):
    n_leaves = fields.Int(required=True)
    n_components = fields.Int(required=True)
    children = fields.List(fields.List(fields.Int), required=True)


class ClusteringSchema(Schema):
    labels = fields.List(fields.Int(), required=True)
    cluster_terms = fields.List(fields.Str(), required=True)
    htree = fields.Nested(_HTreeSchema()) 
    pars = fields.Str(required=True)

class DuplicateDetectionSchema(Schema):
    simhash = fields.List(fields.Int(), required=True)
    cluster_id = fields.List(fields.Int(), required=True)
    dup_pairs = fields.List(fields.List(fields.Int()), required=True)



class EmailThreadingParsSchema(Schema):
    group_by_subject = fields.Boolean(required=True)

class ErrorSchema(Schema):
    message = fields.Str(required=True)

class TreeSchema(Schema):
    id = fields.Int(required=True)
    parent = fields.Int(allow_none=True, required=True)
    subject = fields.Str()
    children = fields.Nested('self', many=True, required=True)


class EmailThreadingSchema(Schema):
    id = fields.Str(required=True)
    data = fields.Nested(TreeSchema, many=True)



class MetricsCategorizationSchema(Schema):
    precision = fields.Number()
    recall = fields.Number()
    f1 = fields.Number()
    roc_auc = fields.Number()


class MetricsClusteringSchema(Schema):
    adjusted_rand = fields.Number()
    adjusted_mutual_info = fields.Number()
    v_measure = fields.Number()


class MetricsDupDetectionSchema(Schema):
    ratio_duplicates = fields.Number()
    f1_same_duplicates = fields.Number()
    mean_duplicates_count = fields.Number()
