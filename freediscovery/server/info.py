
import sys

from flask_apispec import (marshal_with,
                           MethodResource as Resource)
from flask_apispec.annotations import doc
from webargs import fields as wfields

from webargs.core import argmap2schema
from .._version import __version__

# =========================================================================== #
#                        Server info                                          #
# =========================================================================== #


class ServerInfoApi(Resource):

    @doc(description="Return FreeDiscovery server information "
                     " (versions, etc).")
    @marshal_with(argmap2schema(
                  {'version': wfields.Nested({'number': wfields.Str()}),
                   'env': wfields.Nested({'python_version': wfields.Str()}),
                   'config': wfields.Nested({'cache_dir': wfields.Str(),
                                             'debug': wfields.Boolean(),
                                             'hostname': wfields.Str(),
                                             'log_file': wfields.Str(),
                                             'n_workers': wfields.Int(),
                                             'port': wfields.Int(),
                                             'server': wfields.Str()})
                   }))
    def get(self):
        out = {'version': {},
               'env': {}}

        out['version']['number'] = __version__
        out['env']['python_version'] = sys.version
        out['config'] = self._fd_config
        return out
