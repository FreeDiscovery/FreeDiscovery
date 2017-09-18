"""
Categorization Interpretation Example
-------------------------------------

A visual interpretation for the binary categorization outcome for a single document
by looking at the relative contribution of individual words
"""
from __future__ import print_function

import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

from freediscovery.categorization import binary_sensitivity_analysis
from freediscovery.interpretation import explain_categorization, _make_cmap


newsgroups = fetch_20newsgroups(subset='train', categories=['sci.space', 'comp.graphics'],
                                remove=('headers', 'footers', 'quotes'))

document_id = 312  # the document id we want to visualize

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(newsgroups.data)

clf = LogisticRegression()
clf.fit(X, newsgroups.target)

repr_proba = 'Predicted: {0}: {{0:.2f}}, {1}: {{1:.2f}}'.format(*newsgroups.target_names)
print(repr_proba.format(*clf.predict_proba(X[document_id])[0]))
print('Actual label :', newsgroups.target_names[newsgroups.target[document_id]])


weights = binary_sensitivity_analysis(clf, vectorizer.vocabulary_, X[document_id])

cmap = _make_cmap(alpha=0.2, filter_ratio=0.15)
html, norm = explain_categorization(weights, newsgroups.data[document_id], cmap)

fig, ax = plt.subplots(1, 1, figsize=(6, 1.2))
plt.subplots_adjust(bottom=0.4, top=0.7)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')

cb1.set_label('{} < ----- > {}'.format(*newsgroups.target_names))
ax.set_title('Relative word weights', fontsize=12)

# visualize the html results in sphinx gallery
tmp_dir = os.path.join('..', '..', 'doc', 'python', 'examples')
print(os.path.abspath(tmp_dir))
if os.path.exists(tmp_dir):
    with open(os.path.join(tmp_dir, 'out.html'), 'wt') as fh:
        fh.write(html)

####################################
# .. raw:: html
#     :file: out.html
