
import sys

from flask_apispec import (marshal_with,
                           MethodResource as Resource)
from flask_apispec.annotations import doc

from .._version import __version__

# =========================================================================== #
#                        Server info                                          #
# =========================================================================== #


class ServerInfoApi(Resource):

    @doc(description="Return FreeDiscovery server information "
                     " (versions, etc).")
    def get(self):
        out = {'version': {},
               'env': {}}

        out['version']['number'] = __version__
        out['env']['python_version'] = sys.version
        out['config'] = self._fd_config
        return out
