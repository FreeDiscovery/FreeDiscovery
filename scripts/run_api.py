# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from freediscovery.server import fd_app

if __name__ == '__main__':
    cache_dir = sys.argv[1]

    fd_app(cache_dir).run(debug=False, host='0.0.0.0',
            processes=1, threaded=True,
            port=5001, use_reloader=False)

