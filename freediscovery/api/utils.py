# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def _is_inside_docker():
    """ An imperfect way of checking that the server is
    running inside a Docker container"""

    return os.path.exists('/freediscovery_shared/')

