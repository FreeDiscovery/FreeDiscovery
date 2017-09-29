"""
Data Ingestion
==============

An example illustrating the data ingestion in FreeDiscovery
"""

from __future__ import print_function

import requests
import pandas as pd
import json
import os.path

pd.options.display.float_format = '{:,.3f}'.format
pd.options.display.expand_frame_repr = False

dataset_name = "treclegal09_2k_subset"     # see list of available datasets

BASE_URL = "http://localhost:5001/api/v0"  # FreeDiscovery server URL

###############################################################################
#
#  0. Load the test dataset

url = BASE_URL + '/example-dataset/{}'.format(dataset_name)
print(" GET", url)
input_ds = requests.get(url).json()

# To use a custom dataset, simply specify the following variables
# create a custom dataset definition for ingestion
data_dir = input_ds['metadata']['data_dir']
dataset_definition = [{'document_id': row['document_id'],
                       'file_path': os.path.join(data_dir, row['file_path'])} \
                               for row in input_ds['dataset']]


###############################################################################
#
# 1.a Load dataset and initalize feature extraction
url = BASE_URL + '/feature-extraction'
print(" POST", url)
res = requests.post(url, json={'use_hashing': True}).json()

dsid = res['id']
print("   => received {}".format(list(res.keys())))
print("   => dsid = {}".format(dsid))

print("\n1.b Start feature extraction")

url = BASE_URL+'/feature-extraction/{}'.format(dsid)
print(" POST", url)
res = requests.post(url, json={'dataset_definition': dataset_definition})


###############################################################################
#
# 2 check the parameters of the extracted features
url = BASE_URL + '/feature-extraction/{}'.format(dsid)
print(' GET', url)
res = requests.get(url).json()

print('\n'.join(['     - {}: {}'.format(key, val)
      for key, val in res.items() if "filenames" not in key]))

###############################################################################
#
# 3. Examine the id mapping

method = BASE_URL + "/feature-extraction/{}/id-mapping".format(dsid)
print('\n GET', method)
data = {'data': [{'internal_id': row['internal_id']} for row in input_ds['dataset'][:3]]}
print('   DATA', json.dumps(data))
res = requests.post(method, json=data).json()

print(' Response:')
print('  ', json.dumps(res, indent=4))

###############################################################################
#
# 4 Delete the extracted features

url = BASE_URL + '/feature-extraction/{}'.format(dsid)
print(" DELETE", url)
requests.delete(url)
