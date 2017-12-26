# -*- coding: utf-8 -*-

import os
import sys

from freediscovery.externals.joblib.pool import _get_temp_dir

base_dir = os.path.dirname(os.path.dirname(__file__))

EXTERNAL_DATASETS_PATH = os.environ.get('FREEDISCOVERY_EXTERNAL_TEST_DATASETS')


def run(coverage=False):
    import pytest
    argv = ['-x', base_dir, '-v', '-s']  #, "
    if coverage:
        argv += ["--cov=freediscovery"]
    result = pytest.main(argv)
    status = int(result)
    return status


def run_cli(coverage=False):
    status = run(coverage=coverage)
    print('Exit status: {}'.format(status))
    sys.exit(status)


def check_cache(test_env=True):
    import tempfile

    if test_env:
        subfolder = 'freediscovery-cache-test-{}'.format(os.getpid())
    else:
        subfolder = 'freediscovery-cache'

    cache_dir, _ = _get_temp_dir(subfolder, temp_folder=tempfile.gettempdir())

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    return cache_dir

if __name__ == '__main__':
    run_cli()
