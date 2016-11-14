# Query duplicates

 * **URL**: `/api/v0/duplicate-detection/<model-id>` 
 * **Method**: `GET` **URL Params**: None
 * **Data Params**: 
    - distance : int, default=2
              Maximum number of differnet bits in the simhash (Simhash method only)
    - n_rand_lexicons : int, default=1
              number of random lexicons used for duplicate detection (I-Match method only)
    - rand_lexicon_ratio: float, default=0.7
              ratio of the vocabulary used in random lexicons (I-Match method only)


 * **Success Response**: `HTTP 200`
    
        {"simhash": <list[int]>, "cluster_id": <list[int]>, "dup_pairs": <list[list[int]]> }
