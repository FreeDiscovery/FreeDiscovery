import os
import sys
import time
import argparse
import shutil
from subprocess import call
from pathlib import Path

from sklearn.externals import joblib

from .server import fd_app
from freediscovery.engine.cli import (_query_yes_no, _TeeLogger,
                                      _number_of_workers)
from freediscovery.engine.vectorizer import FeatureVectorizer
from freediscovery.engine.pipeline import PipelineFinder
from .datasets import IR_DATASETS, load_dataset
from ._version import __version__

DEFAULT_CACHE_DIR = os.path.join('..', 'freediscovery_shared')


def _parse_cache_dir(cache_dir):
    if cache_dir == DEFAULT_CACHE_DIR:
        cache_dir_new = os.environ.get('FREEDISCOVERY_CACHE_DIR')
        if cache_dir_new is not None:
            cache_dir = cache_dir
    return cache_dir


def _run(args):
    df_config = vars(args).copy()
    df_config.pop('func', None)
    df_config.pop('yes', None)

    cache_dir = _parse_cache_dir(args.cache_dir)
    cache_dir = os.path.normpath(os.path.abspath(cache_dir))
    if not os.path.exists(cache_dir):
        _cache_dir_exists = False
        _create_cache_dir = _query_yes_no('Cache directory does not exist. '
                                          'Create {} ?'.format(cache_dir),
                                          overwrite=args.yes)
        if _create_cache_dir:
            os.makedirs(cache_dir)
        else:
            print('Cache directory not created. Exiting.')
            return
    else:
        _cache_dir_exists = True

    df_config['cache_dir'] = cache_dir
    log_fname = args.log_file.replace('${CACHE_DIR}', cache_dir)
    log_fname = os.path.normpath(os.path.abspath(log_fname))

    df_config['log_file'] = log_fname
    df_config['n_workers'] = df_config.pop('n', None)

    # redirect stdout / stderr to a file
    sys.stdout = _TeeLogger(log_fname)
    print('='*80)
    print(' '*20, 'FreeDiscovery server')
    print(' '*21, '(version {})'.format(__version__))
    print('='*80)
    print(' * Started on {}'.format(time.strftime("%c")))
    print(' * CACHE_DIR: {} [{}]'.format(cache_dir,
                                         'EXISTING' if _cache_dir_exists else 'NEW'))
    print(' * LOG_FILE: {}'.format(log_fname))

    app = fd_app(cache_dir, df_config)
    if args.hostname == '0.0.0.0':
        print(' * WARNING: running the server on hostname 0.0.0.0 '
              '(accessible from any IP address). Please make sure '
              'that this server runs in a protected environement '
              '(e.g. is behind a firewall) or restrict '
              'connections to localhost with --hostname 127.0.0.1 .')

    if args.server in ['gunicorn']:
        try:
            from .server.gunicorn import GunicornApplication
            options = {
                'bind': '%s:%s' % (args.hostname, str(args.port)),
                'workers': args.n,
                'accesslog': '-',  # stdout
                'check_config': True,
                'limit_request_field_size': 0,  # unlimited
                'limit_request_line': 8190,
                'timeout': 18000,  # 5 hours
                'graceful_timeout': 18000,
            }
            parent_pid = os.getpid()
            print(' * Server: gunicorn with {} workers'.format(args.n))
            print(' * Running on http://{}/ (Press CTRL+C to quit)'.format(options['bind']))
            GunicornApplication(app, options).run()
            return
        except SystemExit:
            if os.getpid() == parent_pid:
                print("Stopping FreeDiscovery server.")
            return
        except ImportError:
            #print('Gunicorn not installed')
            pass
        except:
            if os.getpid() == parent_pid:
                print('Exiting.')
            return

    # run the built-in server
    print(' * Server: flask (Werkzeug) threaded')

    df_config['server'] = 'werkzeug'
    app.run(debug=False, host=args.hostname,
            processes=1, threaded=True,
            port=args.port, use_reloader=False)


def _info(args):

    print('FreeDiscovery version: {}'.format(__version__))

    conda_exec = shutil.which('conda')
    pip_exec = shutil.which('pip')

    if conda_exec is not None:
        print("\n# Conda setup\n")
        if args.all:
            call([conda_exec, 'info', '-a'])
            print('\n# List of installed packages\n')
            call([conda_exec, 'list'])
        else:
            call([conda_exec, 'info'])
    elif pip_exec is not None:
        print("\n# Python setup\n")
        call([pip_exec, '-V'])
        print('\n# List of installed packages\n')
        call([pip_exec, 'list'])
    else:
        print('Warning: neither pip nor conda found in $PATH!')


def _list(args):
    cache_dir = _parse_cache_dir(args.cache_dir)
    fe = FeatureVectorizer(cache_dir)
    res = fe.list_datasets()
    res = sorted(res, key=lambda row: row['creation_date'], reverse=True)
    for row in res:
        print(' * Processed dataset {}'.format(row['id']))
        print('    - data_dir: {}'.format(row['data_dir']))
        print('    - creation_date: {}'.format(row['creation_date']))
        for method in ['lsi', 'categorizer', 'cluster', 'dupdet',
                       'threading']:
            dpath = os.path.join(fe.cache_dir, row['id'], method)
            if not os.path.exists(dpath):
                continue
            mid_list = os.listdir(dpath)
            if mid_list:
                print('     # {}'.format(method))
                for mid in mid_list:
                    print('       * {}'.format(mid))
        print(' ')


def _show(args):
    cache_dir = _parse_cache_dir(args.cache_dir)
    p = PipelineFinder.by_id(mid=args.mid, cache_dir=cache_dir)
    print(p)
    print(' * model_id: {}'.format(args.mid))
    print(' * model_type: {}'.format(list(p.keys())[-1]))
    print(' * file_path: {}'.format(p.get_path()))
    try:
        pars = joblib.load(os.path.join(p.get_path(), 'pars'))
        for key, val in pars.items():
            val_str = str(val)
            if len(val_str) > 30 and not isinstance(val, dict):
                continue
            print(' * {}: {}'.format(key, val_str))
    except:
        pass


def _rm(args):
    cache_dir = _parse_cache_dir(args.cache_dir)
    if args.all:
        p = PipelineFinder(cache_dir=cache_dir)
        fpath = p.cache_dir
    elif args.mid:
        mid = args.mid
        p = PipelineFinder.by_id(mid=mid, cache_dir=cache_dir)
        fpath = p.get_path()
    else:
        print('Error: either mid or the -a (--all) flag should be provided. '
              'Exiting.')
        return
    _del_mid = _query_yes_no('Are you sure you want to delete\n'
                             '        {} ?'.format(fpath),
                             default='no',
                             overwrite=args.yes)
    if _del_mid:
        shutil.rmtree(fpath)
        print('Folder {} deleted.'.format(fpath))
    else:
        print('Nothing to be done. Exiting.')

def _download(args):
    cache_dir = Path(args.output).resolve()

    if not cache_dir.exists():
        raise ValueError('Output directory {} does not exist!'
                         .format(cache_dir))

    load_dataset(args.name, cache_dir=cache_dir, verbose=True)


# Allow to propagate formatter_class to subparsers
# https://bugs.python.org/issue21633

class _ArgParser(argparse.ArgumentParser):

    def __init__(self, **kwargs):
        kwargs["formatter_class"] = argparse.ArgumentDefaultsHelpFormatter
        super(_ArgParser, self).__init__(**kwargs)


def main(args=None, return_parser=False):
    """The main CLI interface."""

    parser = _ArgParser()
    subparsers = parser.add_subparsers(help='action')
    # parser.add_argument("-v", ..)

    run_parser = subparsers.add_parser("run",
                     description='The command used to start the server.')
    info_parser = subparsers.add_parser("info",
                     description='Return debug information about '
                                 'the FreeDiscovery install.')
    list_parser = subparsers.add_parser("list",
                     description='List trained models.')
    show_parser = subparsers.add_parser("show", 
                     description='Show detailed information about '
                                 'a trained model.')
    rm_parser = subparsers.add_parser("rm", 
                     description='Remove a trained model specified by its ID.')
    download_parser = subparsers.add_parser("download", 
                     description='Download a document dataset')

    for subparser in [run_parser, list_parser, show_parser,
                      rm_parser]:
        subparser.add_argument('-c', '--cache-dir',
                               default=DEFAULT_CACHE_DIR,
                               help='The cache directory in which '
                                    'the trained models are saved. '
                                    'If this parameter is not specified '
                                    'the value in the environement variable '
                                    'FREEDISCOVERY_CACHE_DIR will be used if '
                                    'it is specified. Otherwise the default '
                                    'value is used.')
    for subparser in [run_parser, rm_parser]:
        subparser.add_argument('-y', '--yes', action='store_true',
                               help='Do not ask for confirmation.')

    # start parser
    run_parser.add_argument('--debug',
                            default=False, action='store_true',
                            help='Start server in debug mode.')
    run_parser.add_argument('--hostname', default='0.0.0.0',
                            help='Server hostname.')
    run_parser.add_argument('-p', '--port', default=5001, type=int,
                            help='Server port.')
    run_parser.add_argument('-s', '--server', default='flask',
                            choices=['flask', 'gunicorn'],
                            help='The server used to run freediscovery. '
                                 '"flask" is the server built-in in flask '
                                 'suitable for developpement. ')
    run_parser.add_argument('--log-file',
                            default='${CACHE_DIR}/freediscovery-backend.log',
                            help='Path to the log file.')
    run_parser.add_argument('-n', default=_number_of_workers(), type=int,
                            help='Number of workers to use when starting '
                                 'the freediscovery server. Only affects '
                                 'the gunicorn server.')
    run_parser.set_defaults(func=_run)

    # info parser
    info_parser.add_argument('-a', '--all', action='store_true',
                             help='Print detailed report.')
    info_parser.set_defaults(func=_info)

    # list parser
    list_parser.set_defaults(func=_list)

    # show parser
    show_parser.add_argument('mid', help='Model id')
    show_parser.set_defaults(func=_show)

    # rm parser
    rm_parser.add_argument('-a', '--all', action='store_true',
                           help='Remove all models.')
    rm_parser.add_argument('mid', nargs='?', help='Model id')
    rm_parser.set_defaults(func=_rm)

    # download parser
    download_parser.add_argument('-o', '--output',
                             default='.',
                             help='Folder where to save the output folder')
    download_parser.add_argument('name',
                             help='The dataset name',
                             choices=IR_DATASETS.keys())
    download_parser.set_defaults(func=_download)

    if return_parser:
        # used to generate sphinx docs
        return parser

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
