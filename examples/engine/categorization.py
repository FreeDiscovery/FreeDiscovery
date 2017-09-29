"""
Categorization
==============

An example to illustrate binary categorizaiton with FreeDiscovery
"""

from __future__ import print_function

import os.path
import requests
import pandas as pd

pd.options.display.float_format = '{:,.3f}'.format
pd.options.display.expand_frame_repr = False

dataset_name = "treclegal09_2k_subset"     # see list of available datasets

BASE_URL = "http://localhost:5001/api/v0"  # FreeDiscovery server URL

#############################################################################
#
# 0. Load the example dataset
# ---------------------------

url = BASE_URL + '/example-dataset/{}'.format(dataset_name)
print(" GET", url)
input_ds = requests.get(url).json()


data_dir = input_ds['metadata']['data_dir']
dataset_definition = [{'document_id': row['document_id'],
                       'file_path': os.path.join(data_dir, row['file_path'])}
                      for row in input_ds['dataset']]

############################################################################
#
# 1. Feature extraction
# ---------------------

print("\n1.a Load dataset and initalize feature extraction")
url = BASE_URL + '/feature-extraction'
print(" POST", url)
res = requests.post(url).json()

dsid = res['id']
print("   => received {}".format(list(res.keys())))
print("   => dsid = {}".format(dsid))

###########################################################################
#
# 1.b Start feature extraction (in the background)

url = BASE_URL+'/feature-extraction/{}'.format(dsid)
print(" POST", url)
res = requests.post(url, json={'dataset_definition': dataset_definition})

#############################################################################
#
# 2. Calculate Latent Semantic Indexing
# -------------------------------------
# (used by Nearest Neighbors method)

url = BASE_URL + '/lsi/'
print("POST", url)

n_components = 100
res = requests.post(url,
                    json={'n_components': n_components,
                          'parent_id': dsid
                          }).json()

lsi_id = res['id']
print('  => LSI model id = {}'.format(lsi_id))
print(('  => SVD decomposition with {} dimensions explaining '
       '{:.2f} % variabilty of the data')
      .format(n_components, res['explained_variance']*100))

#############################################################################
#
# 3. Document categorization
# --------------------------
# 3.a. Train the categorization model

print("   {} positive, {} negative files".format(
      pd.DataFrame(input_ds['training_set'])
        .groupby('category').count()['document_id'], 0))

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
                              'data': input_ds['training_set'],
                              'method': method,
                              'training_scores': True
                              }).json()

    mid = res['id']
    print("     => model id = {}".format(mid))
    print(("    => Training scores: MAP = {average_precision:.3f}, "
           "ROC-AUC = {roc_auc:.3f}, recall @20%: {recall_at_20p:.3f} ")
          .format(**res['training_scores']))

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
    print(df)

    print("\n3.d Compute the categorization scores")
    url = BASE_URL + '/metrics/categorization'
    print(" GET", url)
    res = requests.post(url, json={'y_true': input_ds['dataset'],
                                   'y_pred': res['data']}).json()

    print(("    => Test scores: MAP = {average_precision:.3f}, "
           "ROC-AUC = {roc_auc:.3f}, recall @20%: {recall_at_20p:.3f} ")
          .format(**res))

#############################################################################
#
# 5 Delete the extracted features (and LSI decomposition)
url = BASE_URL + '/feature-extraction/{}'.format(dsid)
requests.delete(url)
