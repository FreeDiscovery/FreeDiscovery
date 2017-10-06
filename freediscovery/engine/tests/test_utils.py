import pytest

from freediscovery.engine.utils import validate_mid
from freediscovery.exceptions import WrongParameter


def test_validate_mid():

    validate_mid('test')
    validate_mid('08UIb00')
    with pytest.raises(WrongParameter):
        validate_mid('çd$^a')
    with pytest.raises(WrongParameter):
        validate_mid(' 9023bd')
