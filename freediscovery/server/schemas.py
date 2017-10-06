# -*- coding: utf-8 -*-

from marshmallow import Schema as BaseSchema, fields


# This file defines marshmallow schemas for REST API output formatting
# Occasionally the same class can be used for input validation
# As webargs (used for input validation) is just a wrapper around marshmallow

class Schema(BaseSchema):

    class Meta:
        strict = True
        ordered = True


class DocumentIndexSchema(Schema):
    document_id = fields.Int()
    render_id = fields.Int()


class DocumentIndexFullSchema(DocumentIndexSchema):
    file_path = fields.Str()
    internal_id = fields.Int()


class _ExampleDatasetElementSchema(DocumentIndexFullSchema):
    category = fields.Str()


class _ExampleDatasetMetadataSchema(Schema):
    data_dir = fields.Str(required=True)
    name = fields.Str(required=True)


class ExampleDatasetSchema(Schema):
    metadata = fields.Nested(_ExampleDatasetMetadataSchema, required=True)
    training_set = fields.Nested(_ExampleDatasetElementSchema, many=True)
    dataset = fields.Nested(_ExampleDatasetElementSchema, many=True,
                            required=True)


class EmptySchema(Schema):
    pass


class IDSchema(Schema):
    id = fields.Str(required=True)


class DocumentIndexNestedSchema(Schema):
    data = fields.Nested(DocumentIndexFullSchema, many=True)


class _DatasetDefinition(Schema):
    document_id = fields.Int()
    rendition_id = fields.Int()
    file_path = fields.Str()
    content = fields.Str()


class _DatasetDefinitionShort(Schema):
    document_id = fields.Int()
    rendition_id = fields.Int()
    file_path = fields.Str()


class FeaturesParsSchema(Schema):
    data_dir = fields.Str()
    n_features = fields.Int(missing=100001)
    analyzer = fields.Str(missing='word')
    stop_words = fields.Str(missing='english')
    n_jobs = fields.Int(missing=1)
    chunk_size = fields.Int()
    ngram_range = fields.List(fields.Int(), missing=[1, 1])
    weighting = fields.Str(missing='nnc')
    norm_alpha = fields.Number(missing=0.75)
    use_hashing = fields.Boolean(missing=False)
    n_samples = fields.Int(dump_only=True)
    n_samples_processed = fields.Int(dump_only=True)
    min_df = fields.Number(required=False)
    max_df = fields.Number(required=False)
    parse_email_headers = fields.Boolean(missing=False)
    preprocess = fields.List(fields.Str(), missing=[])
    id = fields.Str()
    overwrite = fields.Boolean(missing=False)


class FeaturesSchema(FeaturesParsSchema):
    id = fields.Str(required=True)
    filenames = fields.List(fields.Str())


class ClassificationScoresSchema(Schema):
    precision = fields.Number(required=True)
    recall = fields.Number(required=True)
    f1 = fields.Number(required=True)
    roc_auc = fields.Number(required=True)
    average_precision = fields.Number(required=True)
    recall_at_20p = fields.Number()


class LsiParsSchema(Schema):
    parent_id = fields.Str(required=True)
    n_components = fields.Int(required=True)


class LsiPostSchema(IDSchema):
    explained_variance = fields.Number(required=True)


class _ResponsePaginaton(Schema):
    total_response_count = fields.Int()
    current_response_count = fields.Int()
    batch_id = fields.Int()
    batch_id_last = fields.Int()


class CategorizationPostSchema(Schema):
    id = fields.Str(required=True)
    training_scores = fields.Nested(ClassificationScoresSchema())


class _CategorizationIndex(DocumentIndexSchema):
    category = fields.Str(required=True)


class _CategorizationInputSchema(Schema):
    parent_id = fields.Str(required=True)
    # Warning this should be changed to wfields.DelimitedList
    # https://webargs.readthedocs.io/en/latest/api.html#webargs.fields.DelimitedList
    data = fields.Nested(_CategorizationIndex, many=True)
    method = fields.Str(default='LinearSVC')
    cv = fields.Boolean(missing=False)
    training_scores = fields.Boolean(missing=False)


class _CategorizationScoreSchemaElement(DocumentIndexSchema):
    category = fields.Str(required=True)
    score = fields.Number(required=True)


class _CategorizationPredictSchemaElement(DocumentIndexSchema):
    scores = fields.Nested(_CategorizationScoreSchemaElement, many=True,
                           required=True)


class CategorizationPredictSchema(Schema):
    data = fields.Nested(_CategorizationPredictSchemaElement, many=True,
                         required=True)
    pagination = fields.Nested(_ResponsePaginaton, required=True)


class CategorizationParsSchema(Schema):
    method = fields.Str(required=True)
    options = fields.Str(required=True)


class _ClusteringElementSubSchema(DocumentIndexSchema):
    similarity = fields.Number()


class _ClusteringElementSchema(Schema):
    cluster_id = fields.Int(required=True)
    cluster_similarity = fields.Number()
    cluster_label = fields.Str()
    cluster_depth = fields.Int()
    cluster_size = fields.Int()
    children = fields.List(fields.Int())
    documents = fields.Nested(_ClusteringElementSubSchema, many=True,
                              required=True)


class ClusteringSchema(Schema):
    data = fields.Nested(_ClusteringElementSchema, many=True, required=True)


class EmailThreadingParsSchema(Schema):
    group_by_subject = fields.Boolean(required=True)


class ErrorSchema(Schema):
    message = fields.Str(required=True)


class TreeSchema(Schema):
    id = fields.Int(required=True)
    parent = fields.Int(allow_none=True, required=True)
    subject = fields.Str()
    children = fields.Nested('self', many=True)


class EmailThreadingSchema(Schema):
    id = fields.Str(required=True)
    data = fields.Nested(TreeSchema, many=True)


class MetricsCategorizationSchema(Schema):
    precision = fields.Number()
    recall = fields.Number()
    f1 = fields.Number()
    roc_auc = fields.Number()
    average_precision = fields.Number()
    recall_at_20p = fields.Number()


class MetricsClusteringSchema(Schema):
    adjusted_rand = fields.Number()
    adjusted_mutual_info = fields.Number()
    v_measure = fields.Number()


class MetricsDupDetectionSchema(Schema):
    ratio_duplicates = fields.Number()
    f1_same_duplicates = fields.Number()
    mean_duplicates_count = fields.Number()


class _SearchResponseSchemaElement(DocumentIndexSchema):
    score = fields.Number(required=True)


class SearchResponseSchema(Schema):
    data = fields.Nested(_SearchResponseSchemaElement,
                         many=True, required=True)
    pagination = fields.Nested(_ResponsePaginaton, required=True)


class CustomStopWordsSchema(Schema):
    name = fields.Str(required=True)
    stop_words = fields.List(fields.Str(), required=True)


class CustomStopWordsLoadSchema(Schema):
    name = fields.Str(required=True)
    stop_words = fields.List(fields.Str(), required=True)
