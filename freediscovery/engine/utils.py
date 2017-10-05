import re
from ..exceptions import WrongParameter


def validate_mid(mid):
    """Validate a user provided dataset id"""

    if not re.match('^[a-zA-Z0-9_\-]+$', mid):
        raise WrongParameter(('id={} is not valid. '
                              'It can only contain letters, numbers '
                              'and "-", "_" characters. ')
                             .format(mid))

    if len(mid) < 2 or len(mid) > 50:
        raise WrongParameter(('id={} is not valid. '
                              'It must be between 2 and 50 characters long.')
                             .format(mid))
