"""
Categorization Example [REST API]
---------------------------------

An example to illustrate binary categorizaiton with FreeDiscovery
"""

from __future__ import print_function

from time import time, sleep
from pathlib import Path
import os.path
from multiprocessing import Process
import requests
import pandas as pd

pd.options.display.float_format = '{:,.3f}'.format
pd.options.display.expand_frame_repr = False

dataset_name = "treclegal09_37k_subset"     # see list of available datasets

BASE_URL = "http://localhost:5001/api/v0"  # FreeDiscovery server URL

ingestion_method = 'file_path'
#ingestion_method = 'content'

if __name__ == '__main__':

    print(" 0. Load the example dataset")
    url = BASE_URL + '/example-dataset/{}'.format(dataset_name)
    print(" GET", url)
    input_ds = requests.get(url).json()

    data_dir = input_ds['metadata']['data_dir']
    dataset_definition = []
    for row in input_ds['dataset']:
        dd_row = {'document_id': row['document_id']}
        if ingestion_method == 'file_path':
            dd_row['file_path'] = os.path.join(data_dir, row['file_path'])
            dataset_definition.append(dd_row)
        elif ingestion_method == 'content':
            with Path(data_dir, row['file_path']).open('rt') as fh:
                try:
                    dd_row['content'] = fh.read()
                    dataset_definition.append(dd_row)
                except UnicodeDecodeError:
                    print(Path(data_dir, row['file_path']), ' failed!')
        else:
            raise ValueError
    # create a custom dataset definition for ingestion

    # 1. Feature extraction

    print("\n1.a Load dataset and initalize feature extraction")
    url = BASE_URL + '/feature-extraction'
    print(" POST", url)
    res = requests.post(url, json={'max_df': 0.6, 'preprocess': ['emails_ignore_header']}).json()

    dsid = res['id']
    print("   => received {}".format(list(res.keys())))
    print("   => dsid = {}".format(dsid))

    print("\n1.b Start feature extraction (in the background)")

    # Make this call in a background process (there should be a better way of doing it)
    url = BASE_URL+'/feature-extraction/{}'.format(dsid)
    print(" POST", url)
    t0 = time()
    for i in range(0, len(dataset_definition), 8000):
        res = requests.post(url, json={'dataset_definition': dataset_definition[i:i + 8000],
                                       'vectorize': False})

    print('     ... files ingested in {:.1f} s'.format((time() - t0)))
    t1 = time()
    res = requests.post(url, json={'vectorize': True})
    print('     ... files vectorized in {:.1f} s'.format((time() - t0)))

    print("\n1.d. check the parameters of the extracted features")
    url = BASE_URL + '/feature-extraction/{}'.format(dsid)
    print(' GET', url)
    res = requests.get(url).json()

    print('\n'.join(['     - {}: {}'.format(key, val)
          for key, val in res.items() if "filenames" not in key]))

    # 3. Document categorization

    print("\n3.a. Train the categorization model")
    print("   {} positive, {} negative files".format(
          pd.DataFrame(input_ds['training_set'])
            .groupby('category').count()['document_id'], 0))

    for method, use_lsi in [('LogisticRegression', False)]:

        print('='*80, '\n', ' '*10,
              method, " + LSI" if use_lsi else ' ', '\n', '='*80)
        parent_id = dsid

        url = BASE_URL + '/categorization/'
        print(" POST", url)
        print(' Training...')

        res = requests.post(url,
                            json={'parent_id': parent_id,
                                  'data': input_ds['training_set'],
                                  'method': method,  # one of "LinearSVC", "LogisticRegression", 'xgboost'
                                  'training_scores': True
                                  }).json()

        mid = res['id']
        print("     => model id = {}".format(mid))
        print('    => Training scores: MAP = {average_precision:.3f}, ROC-AUC = {roc_auc:.3f}, recall @20%: {recall_at_20p:.3f} '.format(**res['training_scores']))

        print("\n3.b. Check the parameters used in the categorization model")
        url = BASE_URL + '/categorization/{}'.format(mid)
        print(" GET", url)
        res = requests.get(url).json()

        print('\n'.join(['     - {}: {}'.format(key, val)
              for key, val in res.items() if key not in ['index', 'category']]))

        print("\n3.c Categorize the complete dataset with this model")
        url = BASE_URL + '/categorization/{}/predict'.format(mid)
        print(" GET", url)
        res = requests.get(url, json={'subset': 'test'}).json()

        data = []
        for row in res['data']:
            nrow = {'document_id': row['document_id'],
                    'category': row['scores'][0]['category'],
                    'score': row['scores'][0]['score']}
            if method == 'NearestNeighbor':
                nrow['nearest_document_id'] = row['scores'][0]['document_id']
            data.append(nrow)

        df = pd.DataFrame(data).set_index('document_id')
        print(df.iloc[:10])

    # 4. Cleaning
    print("\n5.a Delete the extracted features (and LSI decomposition)")
    url = BASE_URL + '/feature-extraction/{}'.format(dsid)
    print(" DELETE", url)
    requests.delete(url)
