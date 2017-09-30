# -*- coding: utf-8 -*-

import os
import os.path

from sklearn.externals.joblib import dump, load

from freediscovery.engine.pipeline import PipelineFinder


class _StopWordsWrapper(object):
    """A mechanism for adding / managing custom stop words
        Parameters
        ----------
        cache_dir : str
           folder where the model will be saved
        stop_words : list
           a list of strings
    """
    _wrapper_type = "stop_words"

    def __init__(self, cache_dir='/tmp/'):
        """ Initialize a stop words wrapper

        Parameters
        ----------
        cache_dir : str
          the cache directory
        """
        self.cache_dir = PipelineFinder._normalize_cachedir(cache_dir)
        self.model_dir = self.cache_dir / 'stop_words'

        if not self.model_dir.exists():
            self.model_dir.mkdir()

    def save(self, name, stop_words):
        """
        Save a list of stop_words with joblib.save under
             $CACHE_DIR/stop_words/<name>.pkl

        Parameters
        ----------
        name : str
            stop words name / identifier
        stop_words : list
            list of stop words
        """

        self.stop_words = stop_words  # list of stop words

        self.name = self.model_dir / (str(name) + '.pkl')

        dump(self.stop_words, str(self.name))

    def load(self, name):
        """Retrive stop words specified by a name
        """
        self.name = self.model_dir / (str(name) + '.pkl')
        self.stop_words = load(str(self.name))
        return (self.stop_words)

    def delete(self, name):
        """Delete stop words specified by a name
        """
        if (self.model_dir / (str(name) + '.pkl')).exists():
            (self.model_dir / (str(name) + '.pkl')).unlink()

    def __contains__(self, name):
        """ Check if a given stop words set exists """
        return (self.model_dir / (str(name) + '.pkl')).exists()

    def list(self):
        """ Returns a list of exiting stop-words """
        return [os.splitext(el)[0] for el in
                os.listdir(str(self.model_dir))]
