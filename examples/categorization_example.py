"""
Binary Categorization Example
-------------------------------

This example should be run in a Jupyter Notebook (cf. "Examples" section in FreeDiscovery Documentation for more detailed information)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import time, sleep
import os
from multiprocessing import Process
import requests
import pandas as pd


def _parent_dir(path, n=0):
    path = os.path.abspath(path)
    if n == 0:
        return path
    else:
        return os.path.dirname(_parent_dir(path, n=n-1))


def _print_url(op, url):
    print(' '*1, op, url) 

use_docker = False

if use_docker:
    data_dir = "/freediscovery_shared/tar_fd_benchmark"
else:
    data_dir = "../freediscovery_shared/tar_fd_benchmark"
rel_data_dir = os.path.abspath("../../freediscovery_shared/tar_fd_benchmark") # relative path between this file and the FreeDiscovery source folder

BASE_URL = "http://localhost:5001/api/v0"  # FreeDiscovery server URL


# 1. Feature extraction

print("\n1.a Load dataset and initalize feature extraction")
url = BASE_URL + '/feature-extraction'
_print_url("POST", url)
fe_opts = {'data_dir': os.path.join(data_dir, 'data'),
           'stop_words': 'None', 'chunk_size': 2000, 'n_jobs': -1,
           'use_idf': 1, 'sublinear_tf': 1, 'binary': 0, 'n_features': 50001,
           'analyzer': 'word', 'ngram_range': (1, 1), "norm": "l2"
          }
res = requests.post(url, json=fe_opts)

dsid = res.json()['id']
print("   => received {}".format(list(res.json().keys())))
print("   => dsid = {}".format(dsid))

print("\n1.b Start feature extraction (in the background)")

# Make this call in a background process (there should be a better way of doing it)
url = BASE_URL+'/feature-extraction/{}'.format(dsid)
_print_url("POST", url)
p = Process(target=requests.post, args=(url,))
p.start()
sleep(5.0) # wait a bit for the processing to start

print('\n1.c Monitor feature extraction progress')
url = BASE_URL+'/feature-extraction/{}'.format(dsid)
_print_url("GET", url)

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
_print_url('GET', url)
res = requests.get(url)

data = res.json()
for key, val in data.items():
    if key!='filenames':
           print('     - {}: {}'.format(key, val))


# 2. Load relevant and non relevant seed file list

with open(os.path.join(rel_data_dir,'seed_relevant.txt'), 'rt') as fh:
    relevant_files = [el.strip() for el in fh.readlines()]

with open(os.path.join(rel_data_dir,'seed_non_relevant.txt'), 'rt') as fh:
    non_relevant_files = [el.strip() for el in fh.readlines()]

# Load ground truth file
if use_docker:
    gtfile = os.path.join(data_dir, "ground_truth_file.txt")  
else:
    gtfile = os.path.join(rel_data_dir, "ground_truth_file.txt") 


# 3. Document categorization with ML algorithms

print("\n3.b. Train the ML categorization model")
print("       {} relevant, {} non-relevant files".format(
    len(relevant_files), len(non_relevant_files)))
url = BASE_URL + '/categorization/'
_print_url("POST", url)

res = requests.post(url,
                    json={'relevant_filenames': relevant_files,
                          'non_relevant_filenames': non_relevant_files,
                          'dataset_id': dsid,
                          'method': 'LogisticRegression',  # one of "LinearSVC", "LogisticRegression", 'xgboost'
                          'cv': 0  # use cross validation (recommended)
                          })

data = res.json()
mid = data['id']
print("     => model id = {}".format(mid))
print('    => Training scores: MAP = {average_precision:.2f}, ROC-AUC = {roc_auc:.2f}'.format(**data))

print("\n3.c. Check the parameters used in the categorization model")
url = BASE_URL + '/categorization/{}'.format(mid)
_print_url("GET", url)
res = requests.get(url)

data = res.json()
for key, val in data.items():
    if "filenames" not in key:
        print('     - {}: {}'.format(key, val))

print("\n3.d Categorize the complete dataset with this model")
url = BASE_URL + '/categorization/{}/predict'.format(mid)
_print_url("GET", url)
res = requests.get(url)
prediction = res.json()['prediction']

print("    => Predicting {} relevant and {} non relevant documents".format(
    len(list(filter(lambda x: x>0, prediction))),
    len(list(filter(lambda x: x<0, prediction)))))

print("\n3.e Test categorization accuracy")
print("         using {}".format(gtfile))  
url = BASE_URL + '/categorization/{}/test'.format(mid)
_print_url("POST", url)
res = requests.post(url,
                    json={'ground_truth_filename': gtfile})
               
data2 = res.json()
print('    => Test scores: MAP = {average_precision:.2f}, ROC-AUC = {roc_auc:.2f}'.format(**data2))


# 4. Document categorization with LSI

print("\n4.a. Calculate LSI")

url = BASE_URL + '/lsi/'
_print_url("POST", url)

n_components = 100
res = requests.post(url,
                    json={'n_components': n_components,
                          'dataset_id': dsid
                          })

data = res.json()
lid = data['id']
print('  => LSI model id = {}'.format(lid))
print('  => SVD decomposition with {} dimensions explaining {:.2f} % variabilty of the data'.format(
                        n_components, data['explained_variance']*100))
print("\n4.b. Predict categorization with LSI")

url = BASE_URL + '/lsi/{}/predict'.format(lid)
_print_url("POST", url)
res = requests.post(url,
                    json={'relevant_filenames': relevant_files,
                          'non_relevant_filenames': non_relevant_files
                          })
data = res.json()
print(data.keys())
prediction = data['prediction']

print('    => Training scores: MAP = {average_precision:.2f}, ROC-AUC = {roc_auc:.2f}'.format(**data))


print("\n4.c. Test categorization with LSI")
url = BASE_URL + '/lsi/{}/test'.format(lid)
_print_url("POST", url)

res = requests.post(url,
                    json={'relevant_filenames': relevant_files,
                          'non_relevant_filenames': non_relevant_files,
                          'ground_truth_filename': gtfile
                          })
data2 = res.json()
print('    => Test scores: MAP = {average_precision:.2f}, ROC-AUC = {roc_auc:.2f}'.format(**data2))

pd.DataFrame({key: data[key] for key in data if 'prediction' in key or 'nearest' in key})


print("\n5.a Delete the extracted features")
url = BASE_URL + '/feature-extraction/{}'.format(dsid)
_print_url("DELETE", url)
