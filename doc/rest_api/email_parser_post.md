# Load a dataset and parse emails

Initialize the feature extraction on a document collection.

 * **URL**: `/api/v0/email-parser/`
 * **Method**: `POST`,                **URL Params**: None
 * **Data Params**: 
    - `data_dir`: [required] relative path to the directory with the input files 

 * **Success Response**: `HTTP 200`

        {"id": <str>, "filenames": <list[str]>  }
