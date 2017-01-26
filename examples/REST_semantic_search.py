"""
Semantic Search Example [REST API]
----------------------------------

An example of Semantic Search
"""

from __future__ import print_function

import requests
import pandas as pd

pd.options.display.float_format = '{:,.3f}'.format
pd.options.display.expand_frame_repr = False

dataset_name = "treclegal09_2k_subset"     # see list of available datasets

BASE_URL = "http://localhost:5001/api/v0"  # FreeDiscovery server URL

if __name__ == '__main__':

    print(" 0. Load the test dataset")
    url = BASE_URL + '/datasets/{}'.format(dataset_name)
    print(" GET", url)
    res = requests.get(url, json={'return_file_path': True}).json()

    # To use a custom dataset, simply specify the following variables
    seed_document_id = res['seed_document_id']
    seed_y = res['seed_y']
    ground_truth_y = res['ground_truth_y']

    # create a custom dataset definition for ingestion
    dataset_definition = []
    for document_id, fname in zip(res['document_id'], res['file_path']):
        dataset_definition.append({'document_id': document_id,
                                  'rendering_id': 0,
                                  'file_path': fname})

    # 1. Feature extraction

    print("\n1.a Load dataset and initalize feature extraction")
    url = BASE_URL + '/feature-extraction'
    print(" POST", url)
    res = requests.post(url, json={'dataset_definition': dataset_definition,
                                   'use_hashing': True}).json()

    dsid = res['id']
    print("   => received {}".format(list(res.keys())))
    print("   => dsid = {}".format(dsid))

    print("\n1.b Start feature extraction")

    url = BASE_URL+'/feature-extraction/{}'.format(dsid)
    print(" POST", url)
    requests.post(url)

    # 3. Document categorization with LSI (used for Nearest Neighbors method)

    print("\n2. Calculate LSI")

    url = BASE_URL + '/lsi/'
    print("POST", url)

    n_components = 100
    res = requests.post(url,
                        json={'n_components': n_components,
                              'parent_id': dsid
                              }).json()

    lsi_id = res['id']
    print('  => LSI model id = {}'.format(lsi_id))
    print('  => SVD decomposition with {} dimensions explaining {:.2f} % variabilty of the data'.format(
                            n_components, res['explained_variance']*100))


    # 3. Semantic search

    print("\n3.a. Perform the semantic search")


    query = """There are some conflicts with the draft date, so we will probably need to
                have it on a different date."""

    url = BASE_URL + '/search/'
    print(" POST", url)

    res = requests.get(url,
                        json={'parent_id': lsi_id,
                              'query': query
                              }).json()

    data = res['data']

    df = pd.DataFrame(data).set_index('internal_id')
    print(df)

    print(df.score.max())


    # 4. Cleaning
    print("\n4. Delete the extracted features")
    url = BASE_URL + '/feature-extraction/{}'.format(dsid)
    print(" DELETE", url)
    requests.delete(url)
