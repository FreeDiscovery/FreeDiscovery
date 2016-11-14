# List processed datasets 
 * **URL**: `/api/v0/feature-extraction/`
 * **Method**: `GET`,  **URL Params**: None
 * **Success Response**: `HTTP 200`
          
        [
          {"id": <str>, "data_dir": <str>, "n_samples": <int>,
           "n_features": <int>, "n_jobs": <int>, "chunk_size": <int>,
           "analyzer": <str>, "stop_words": <str>,
           "ngram_range": <list[int]> }, 
            ...
        ]
