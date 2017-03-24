# -*- coding: utf-8 -*-


def _is_in_range(valid_values):
    """
    Validate that an element is within a given value
    """

    def f(x):
        if x not in valid_values:
            raise ValueError('{} not in {}'.format(x, valid_values))
