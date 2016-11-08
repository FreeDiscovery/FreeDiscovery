# -*- coding: utf-8 -*-

import os
import sys

base_dir = os.path.dirname(__file__)


def run(coverage=False):
    import pytest
    argv = ['-x', base_dir, '-v']  #, "
    if coverage:
        argv += ["--cov=freediscovery"]
    result = pytest.main(argv)
    status = int(result)
    return status


def run_cli(coverage=False):
    status = run(coverage=coverage)
    print('Exit status: {}'.format(status))
    sys.exit(status)


def check_cache():
    if os.name == 'nt':
        cache_dir = '.\\'
    else:
        cache_dir = "/tmp/"

    if not os.path.exists(cache_dir):
        raise SkipTest
    return cache_dir

if __name__ == '__main__':
    run_cli()


