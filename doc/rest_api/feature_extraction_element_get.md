# Load extracted features (and obtain the processing status).  

 * **URL**: `/api/v0/feature-extraction/<dataset-id>`
 * **Method**: `GET`,  **URL Params**: None

 * **Success Response**: `HTTP 202` (processing in progress) or `HTTP 200` (processing done) 
           
         {"id": <str>, "data_dir": <str>, "n_samples": <int>,
          "n_features": <int>, "analyzer": <str>, "stop_words": <str>,
          "ngram_range": <list[int]>, "n_jobs": <int>, "chunk_size": <int>,
          "n_samples_processed": <int>,
          "filenames": [] or <list[str>>}

 * **Error Response (processing failed)**: `HTTP 520`
 
        {"message": "Processing failed, see server logs!"}
