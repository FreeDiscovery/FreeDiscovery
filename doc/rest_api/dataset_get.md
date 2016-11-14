
# Load benchmark dataset

Currently the following datasets based on TREC 2009 legal collection
are supported:
   - treclegal09_2k_subset  :   2 400 documents,   2 MB
   - treclegal09_20k_subset :  20 000 documents,  30 MB
   - treclegal09_37k_subset :  37 000 documents,  55 MB
   - treclegal09            : 700 000 documents, 1.2 GB
The ground truth files for categorization are adapted from TAR Toolkit.

If you encounter any issues for downloads with this function,
you can also manually download and extract the required dataset to `cache_dir` (the
download url is `http://r0h.eu/d/<name>.tar.gz`), then re-run this function to get
the required metadata.


 * **URL**: `/api/v0/dataset/<dataset-name>` 
 * **Method**: `GET` **URL Params**: None
 * **Data Params**: None

 * **Success Response**: `HTTP 200`
    
        {"data_dir": <str>, "base_dir": <str>,
         "seed_non_relevant_files": <list[str]>, "seed_relevant_files": <list[str]>,
         "ground_truth_file": <str>}
