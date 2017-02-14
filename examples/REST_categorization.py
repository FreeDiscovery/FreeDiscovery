"""
Categorization Example [REST API]
---------------------------------

An example to illustrate binary categorizaiton with FreeDiscovery
"""

from __future__ import print_function

from time import time, sleep
from multiprocessing import Process
import requests
import pandas as pd

pd.options.display.float_format = '{:,.3f}'.format
pd.options.display.expand_frame_repr = False

dataset_name = "treclegal09_2k_subset"     # see list of available datasets

BASE_URL = "http://localhost:5001/api/v0"  # FreeDiscovery server URL

if __name__ == '__main__':

    print(" 0. Load the example dataset")
    url = BASE_URL + '/example-dataset/{}'.format(dataset_name)
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

    print("\n1.b Start feature extraction (in the background)")

    # Make this call in a background process (there should be a better way of doing it)
    url = BASE_URL+'/feature-extraction/{}'.format(dsid)
    print(" POST", url)
    p = Process(target=requests.post, args=(url,))
    p.start()
    sleep(5.0) # wait a bit for the processing to start

    print('\n1.c Monitor feature extraction progress')
    url = BASE_URL+'/feature-extraction/{}'.format(dsid)
    print(" GET", url)

    t0 = time()
    while True:
        res = requests.get(url)
        if res.status_code == 520:
            p.terminate()
            raise ValueError('Processing did not start')
        elif res.status_code == 200:
            break # processing finished
        data = res.json()
        print('     ... {}k/{}k files processed in {:.1f} min'.format(
                    data['n_samples_processed']//1000, data['n_samples']//1000, (time() - t0)/60.))
        sleep(15.0)

    p.terminate()  # just in case, should not be necessary


    print("\n1.d. check the parameters of the extracted features")
    url = BASE_URL + '/feature-extraction/{}'.format(dsid)
    print(' GET', url)
    res = requests.get(url).json()

    print('\n'.join(['     - {}: {}'.format(key, val) for key, val in res.items() \
                                                      if "filenames" not in key]))
    # this step is not necessary anymore
    #method = BASE_URL + "/feature-extraction/{}/id-mapping/flat".format(dsid)
    #res = requests.get(method, data={'document_id': seed_document_id})
    #seed_internal_id = res.json()['internal_id']


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


    # 3. Document categorization

    print("\n3.a. Train the categorization model")
    print("   {} relevant, {} non-relevant files".format(seed_y.count(1), seed_y.count(0)))

    seed_index_nested = [{'document_id': internal_id, 'category': y} \
                                for internal_id, y in zip(seed_document_id, seed_y)]

    for method, use_lsi in [('LinearSVC', False),
                            ('NearestNeighbor', True)]:

        print('='*80, '\n', ' '*10,
              method, " + LSI" if use_lsi else ' ', '\n', '='*80)
        if use_lsi:
            # Categorization with the previously created LSI model
            parent_id = lsi_id
        else:
            # Categorization with original text features
            parent_id = dsid

        url = BASE_URL + '/categorization/'
        print(" POST", url)
        print(' Training...')

        res = requests.post(url,
                            json={'parent_id': parent_id,
                                  'data': seed_index_nested,
                                  'method': method,  # one of "LinearSVC", "LogisticRegression", 'xgboost'
                                  }).json()

        mid = res['id']
        print("     => model id = {}".format(mid))
        print('    => Training scores: MAP = {average_precision:.3f}, ROC-AUC = {roc_auc:.3f}, F1= {f1:.3f}'.format(**res))

        print("\n3.b. Check the parameters used in the categorization model")
        url = BASE_URL + '/categorization/{}'.format(mid)
        print(" GET", url)
        res = requests.get(url).json()

        print('\n'.join(['     - {}: {}'.format(key, val) for key, val in res.items() \
                                                          if key not in ['index', 'category']]))

        print("\n3.c Categorize the complete dataset with this model")
        url = BASE_URL + '/categorization/{}/predict'.format(mid)
        print(" GET", url)
        res = requests.get(url).json()

        if method == "NearestNeighbor":
            data = res['data']
        else:
            data = res['data']

        df = pd.DataFrame(data).set_index('internal_id')
        if method == "NearestNeighbor":
            df = df[['document_id', 'nn_negative__distance', 'nn_negative__document_id',
                  'nn_positive__distance', 'nn_positive__document_id', 'score']]

        print(df)

        #print("\n3.d Compute the categorization scores")
        #url = BASE_URL + '/metrics/categorization'
        #print(" GET", url)
        #res = requests.post(url, json={'y_true': ground_truth_y,
        #                              'y_pred': df.score.values.tolist(),
        #                             } ).json()


        #print('    => Test scores: MAP = {average_precision:.3f}, ROC-AUC = {roc_auc:.3f}'.format(**res))

    # 4. Cleaning
    print("\n5.a Delete the extracted features (and LSI decomposition)")
    url = BASE_URL + '/feature-extraction/{}'.format(dsid)
    print(" DELETE", url)
    requests.delete(url)
