"""
Categorization Interpreation Example [Python API]
-------------------------------------------------

A visual explanation for the outcome for the binary categorization of a single document
"""
import os

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from freediscovery.categorization import explain_binary_categorization


newsgroups = fetch_20newsgroups(subset='train', categories=['sci.space', 'alt.atheism'])

document_id = 3 # the document we want to visualize


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(newsgroups.data)

clf = LogisticRegression()
clf.fit(X, newsgroups.target)

viz = explain_binary_categorization(clf, vectorizer.vocabulary_,
                                    X[document_id])


for key, val in sorted(viz.items(), key=lambda x: x[1]):
    print(key, val)


# some sample HTML, to be replaced with the visualization example
html = '<span style="background-color:red;"> test</span>'

if 'CI' in os.environ:
    # rendering example in sphinx-gallery
    # a hack to render HTML
    with open('../doc/examples/out.html', 'wt') as fh:
        fh.write(html)

    ####################################
    # .. raw:: html
    #     :file: out.html
else:
    # assumes we are in an Jupyter notebook
    from IPython.display import display, HTML
    display(HTML(html))
