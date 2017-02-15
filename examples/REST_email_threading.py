"""
Email threading Example [REST API]
--------------------------------------

Thread a email collection
"""
from __future__ import print_function

from time import time
import sys
import platform
from itertools import groupby

import numpy as np
import pandas as pd
import requests
import sys
import os

pd.options.display.float_format = '{:,.3f}'.format


if platform.system() == 'Windows' and sys.version_info > (3, 0):
   print('This example currently fails on Windows with PY3 (issue #')
   sys.exit()


dataset_name = "fedora_ml_3k_subset"     # see list of available datasets

BASE_URL = "http://localhost:5001/api/v0"  # FreeDiscovery server URL

print(" 0. Load the test dataset")
url = BASE_URL + '/example-dataset/{}'.format(dataset_name)
print(" GET", url)
res = requests.get(url)
res = res.json()

# To use a custom dataset, simply specify the following variables
data_dir = res['metadata']['data_dir']


print("\n1.a Parse emails")
url = BASE_URL + '/email-parser'
print(" POST", url)
res = requests.post(url, json={'data_dir': data_dir })

dsid = res.json()['id']
print("   => received {}".format(list(res.json().keys())))
print("   => dsid = {}".format(dsid))



print("\n1.b. check the parameters of the emails")
url = BASE_URL + '/email-parser/{}'.format(dsid)
print(' GET', url)
res = requests.get(url)

data = res.json()
print('\n'.join(['     - {}: {}'.format(key, val) for key, val in data.items() \
                                                        if key != 'filenames']))

print("\n2. Email threading")

url = BASE_URL + '/email-threading/'
print(" POST", url)
t0 = time()
res = requests.post(url,
        json={'parent_id': dsid }).json()

mid  = res['id']
print("     => model id = {}".format(mid))


def print_thread(container, depth=0):
    print(''.join(['> ' * depth,  container['subject'],
                   ' (id={})'.format(container['id'])]))

    for child in container['children']:
        print_thread(child, depth + 1)


print("\nThreading examples\n"
      "cf. https://www.redhat.com/archives/rhl-devel-list/2008-October/thread.htlm \n"
      "for ground truth data (mailman has a maximum threading depth of 3, unlike FreeDiscovery)"
       )
for idx in [-1, -2, -3, -4, -5]: # get latest threads
    print(' ')
    print_thread(res['data'][idx])
