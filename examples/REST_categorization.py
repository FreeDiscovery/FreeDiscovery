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

    print(" 0. Load the test dataset")
    url = BASE_URL + '/datasets/{}'.format(dataset_name)
    print(" GET", url)
    res = requests.get(url).json()

    # To use a custom dataset, simply specify the following variables
    data_dir = res['data_dir']
    seed_filenames = res['seed_filenames']
    seed_y = res['seed_y']
    ground_truth_file = res['ground_truth_file']  # (optional)


    # 1. Feature extraction

    print("\n1.a Load dataset and initalize feature extraction")
    url = BASE_URL + '/feature-extraction'
    print(" POST", url)
    fe_opts = {'data_dir': data_dir,
               'stop_words': 'english', 'chunk_size': 2000, 'n_jobs': -1,
               'use_idf': 1, 'sublinear_tf': 0, 'binary': 0, 'n_features': 50001,
               'analyzer': 'word', 'ngram_range': (1, 1), "norm": "l2"
              }
    res = requests.post(url, json=fe_opts).json()

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

    method = BASE_URL + "/feature-extraction/{}/index".format(dsid)
    res = requests.get(method, data={'filenames': seed_filenames})
    seed_index = res.json()['index']

    # 2. Document categorization with ML algorithms

    print("\n2.a. Train the ML categorization model")
    print("   {} relevant, {} non-relevant files".format(seed_y.count(1), seed_y.count(0)))
    url = BASE_URL + '/categorization/'
    print(" POST", url)
    print(' Training...')

    res = requests.post(url,
                        json={'index': seed_index,
                              'y': seed_y,
                              'dataset_id': dsid,
                              'method': 'LinearSVC',  # one of "LinearSVC", "LogisticRegression", 'xgboost'
                              'cv': 0                          # Cross Validation
                              }).json()

    mid = res['id']
    print("     => model id = {}".format(mid))
    print('    => Training scores: MAP = {average_precision:.3f}, ROC-AUC = {roc_auc:.3f}'.format(**res))

    print("\n2.b. Check the parameters used in the categorization model")
    url = BASE_URL + '/categorization/{}'.format(mid)
    print(" GET", url)
    res = requests.get(url).json()

    print('\n'.join(['     - {}: {}'.format(key, val) for key, val in res.items() \
                                                      if key not in ['index', 'y']]))

    print("\n2.c Categorize the complete dataset with this model")
    url = BASE_URL + '/categorization/{}/predict'.format(mid)
    print(" GET", url)
    res = requests.get(url).json()
    prediction = res['prediction']

    print("    => Predicting {} relevant and {} non relevant documents".format(
        len(list(filter(lambda x: x>0, prediction))),
        len(list(filter(lambda x: x<0, prediction)))))

    print("\n2.d Test categorization accuracy")
    print("         using {}".format(ground_truth_file))  
    url = BASE_URL + '/categorization/{}/test'.format(mid)
    print("POST", url)
    res = requests.post(url, json={'ground_truth_filename': ground_truth_file}).json()

    print('    => Test scores: MAP = {average_precision:.3f}, ROC-AUC = {roc_auc:.3f}'.format(**res))


    # 3. Document categorization with LSI

    print("\n3.a. Calculate LSI")

    url = BASE_URL + '/lsi/'
    print("POST", url)

    n_components = 100
    res = requests.post(url,
                        json={'n_components': n_components,
                              'dataset_id': dsid
                              }).json()

    lid = res['id']
    print('  => LSI model id = {}'.format(lid))
    print('  => SVD decomposition with {} dimensions explaining {:.2f} % variabilty of the data'.format(
                            n_components, res['explained_variance']*100))
    print("\n3.b. Predict categorization with LSI")

    url = BASE_URL + '/lsi/{}/predict'.format(lid)
    print("POST", url)
    res = requests.post(url,
                        json={'index': seed_index,
                              'y': seed_y
                              }).json()
    prediction = res['prediction']

    print('    => Training scores: MAP = {average_precision:.3f}, ROC-AUC = {roc_auc:.3f}'.format(**res['scores']))
    df = pd.DataFrame({key: res[key] for key in res if key not in ['id', 'scores']})


    print("\n3.c. Test categorization with LSI")
    url = BASE_URL + '/lsi/{}/test'.format(lid)
    print(" POST", url)

    res = requests.post(url,
                        json={'index': seed_index,
                              'y': seed_y,
                              'ground_truth_filename': ground_truth_file
                              }).json()
    print(res)
    print('    => Test scores: MAP = {average_precision:.3f}, ROC-AUC = {roc_auc:.3f}'.format(**res))

    print('\n', df)

    # 4. Cleaning
    print("\n4.a Delete the extracted features")
    url = BASE_URL + '/feature-extraction/{}'.format(dsid)
    print(" DELETE", url)
    requests.delete(url)
