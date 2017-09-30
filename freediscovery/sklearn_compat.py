# Authors: Roman Yurchak
#
# License: BSD 3 clause

import sklearn


def _parse_version(version_string):
    """ adapted from sklearn """
    version = []
    for x in version_string.split('.'):
        try:
            version.append(int(x))
        except ValueError:
            # x may be of the form dev-1ea1592
            version.append(x)
    return tuple(version)


sklearn_version = _parse_version(sklearn.__version__)
