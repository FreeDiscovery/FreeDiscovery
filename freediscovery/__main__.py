import os
import sys
import time
import argparse

from .server import fd_app
from .cli import _query_yes_no, _TeeLogger
from ._version import __version__

DEFAULT_CACHE_DIR = '../freediscovery_shared/'


def _run(args):
    cache_dir = os.path.normpath(os.path.abspath(args.cache_dir))
    if not os.path.exists(cache_dir):
        _cache_dir_exists = False
        _create_cache_dir = _query_yes_no('Cache directory does not exist. '
                                          'Create {} ?'.format(cache_dir))
        if _create_cache_dir:
            os.makedirs(cache_dir)
        else:
            print('Cache directory not created. Exiting.')
            return
    else:
        _cache_dir_exists = False
        #print('Using an existing cache directory {}'.format(cache_dir))
    log_fname = args.log.replace('${CACHE_DIR}', cache_dir)
    log_fname = os.path.normpath(os.path.abspath(log_fname))

    # redirect stdout / stderr to a file
    sys.stdout = _TeeLogger(log_fname)
    print('='*80)
    print(' '*20, 'FreeDiscovery server')
    print(' '*21, '(version {})'.format(__version__))
    print('='*80)
    print(' * Started on {}'.format(time.strftime("%c")))
    print(' * CACHE_DIR: {}'.format(cache_dir))
    print(' * LOG_FILE: {}'.format(log_fname))



    fd_app(cache_dir).run(debug=False, host='0.0.0.0',
                          processes=1, threaded=True,
                          port=args.port, use_reloader=False)


def _info(args):
    pass


# Allow to propagate formatter_class to subparsers
# https://bugs.python.org/issue21633

class _ArgParser(argparse.ArgumentParser):

    def __init__(self, **kwargs):
        kwargs["formatter_class"] = argparse.ArgumentDefaultsHelpFormatter
        super(_ArgParser, self).__init__(**kwargs)


def main(args=None):
    """The main CLI interface."""

    parser = _ArgParser()
    subparsers = parser.add_subparsers(help='action')
    # parser.add_argument("-v", ..)

    run_parser = subparsers.add_parser("run")
    info_parser = subparsers.add_parser("info")

    # start parser
    run_parser.add_argument('--debug',  type=bool,
                            default=False,
                            help='Start server in debug mode.')
    run_parser.add_argument('-c', '--cache-dir',
                            default=DEFAULT_CACHE_DIR,
                            help='The cache directory in which '
                                 'the trained models will be saved.')
    run_parser.add_argument('--hostname', default='0.0.0.0',
                            help='Server hostname.')
    run_parser.add_argument('-p', '--port', default=5001, type=int)
    run_parser.add_argument('-s', '--server', default='auto',
                            choices=['auto', 'flask', 'gunicorn'],
                            help='The server used to run freediscovery. '
                                 '"flask" is the server built-in in flask '
                                 'suitable for developpement. '
                                 'In production please use gunicorn. '
                                 'When server="auto", gunicorn is used '
                                 'if installed otherwise the "flask" '
                                 'server is used as a fallback.')
    run_parser.add_argument('--log',
                            default='${CACHE_DIR}/freediscovery-backend.log',
                            help='Path to the log file.')
    run_parser.add_argument('-n', default=4, type=int,
                            help='Number of workers to use when starting '
                                 'the freediscovery server.')
    run_parser.set_defaults(func=_run)

    # info parser
    info_parser.set_defaults(func=_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
