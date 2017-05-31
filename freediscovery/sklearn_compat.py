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

if sklearn_version >= (0, 18):
    from sklearn.model_selection import train_test_split

else:
    # sklearn 0.17
    from sklearn.cross_validation import train_test_split
