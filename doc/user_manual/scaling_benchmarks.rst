Scaling Benchmarks
==================

This page aims to summarize the performance and scaling of the algorithms used in FreeDiscovery.

Benchmarks are computed running the `examples <./engine/examples/index.html>`_ on the TREC 2009 corpus of 700 000 documents (1.5 GB or 7 GB uncompressed). The following benchmarks are given for Intel(R) Xeon(R) CPU E3-1225 V2 @ 3.20GHz (4 CPU cores) server with 16 GB of RAM. The time complexites are experimentally (approximately) estimated for the given parameters.

Document ingestion
------------------

+--------------+---------------------------+-----------+---------------------------------+
| Method       | Parameters                | Time (s)  | Complexity                      |
+==============+===========================+===========+=================================+
| Vectorizer   | - `use_hashing=False`     |  780      | `O(n_samples)`                  |
|              | - `use_idf=False`         |           |                                 |
+--------------+---------------------------+-----------+---------------------------------+


Preprocessing
-------------

+--------------+---------------------------+-----------+---------------------------------+
| Method       | Parameters                | Time (s)  | Complexity                      |
+==============+===========================+===========+=================================+
| LSI          | - `n_components=150`      |  270      | `O(n_samples)`                  |
+--------------+---------------------------+-----------+---------------------------------+

Semantic search
---------------

+-------------------------+---------------------------+-----------+---------------------------------+
| Method                  | Parameters                | Time (s)  | Complexity                      |
+=========================+===========================+===========+=================================+
| Text query              | - `n_components=150`      |  270      | `O(n_samples)`                  |
+-------------------------+---------------------------+-----------+---------------------------------+

Near Duplicates Detection
-------------------------

The `examples/duplicate_detection_example.py <./examples/duplicate_detection_example.html>`_ script with `dataset_name='legal09int'` was used,


+---------+---------------------------+-----------+---------------------------------+
| Method  | Parameters                | Time (s)  | Complexity                      |
+=========+===========================+===========+=================================+
|         | - `eps=0.1`               |           |                                 |
| DBSCAN  | - `n_max_samples=2`       |    3800   | `O(n_samples*log(n_samples))`   |
|         | - `lsi_components=100`    |           |                                 |
+---------+---------------------------+-----------+---------------------------------+
| I-Match | - `n_rand_lexicons=10`    |    680    | `O(n_samples)`                  |
|         | - `rand_lexicon_ratio=0.9`|           |                                 |
+---------+---------------------------+-----------+---------------------------------+
| Simhash | - `distance=1`            |    270    | `O(n_samples)`                  |
+---------+---------------------------+-----------+---------------------------------+

where `n_samples` is the number of documents in the dataset.
