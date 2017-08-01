#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from numpy.testing import assert_allclose, assert_array_equal

from freediscovery.io import (parse_ground_truth_file,
                              parse_smart_tokens)


def test_parse_ground_truth_file():
    basename = os.path.dirname(__file__)
    filename = os.path.join(basename, "..","data", "ds_001", "ground_truth_file.txt")
    res = parse_ground_truth_file(filename)
    assert_allclose(res.is_relevant.values, [1, 1, 1, 0, 0, 0])


def test_parse_smart_stemmed():
    from textwrap import dedent
    text = dedent("""\
    .I 26187
    .W
    sunday complet tabulat race race race top top andret brazil brazil brazil rahal bobby lola lola lola cosworth cosworth cosworth cosworth ford ford ford ford mph reynard reynard reynard reynard
    reynard reynard merced merced merced christ benz benz benz motor motor fittipald jimmy vass al uns grand prix finish finish countr chass gil fer pensk bryan hert michael adrian gordon goodyear
    lap robby driv kph canad hond hond hond vancouv vancouv de indycar indycar scot fernandez jr

    .I 26188
    .W
    sunday won race race andret andret rahal bobby lola lola ford ford reynard merced christ benz motor fittipald grand prix finish finish michael win vancouv vancouv indycar indycar
    """)

    res = parse_smart_tokens(text)
    assert_array_equal(res.index, [26187, 26188])
    assert len(res.loc[26188].W.split(' ')) == 28
    res_ref = ['sunday', 'won', 'race', 'race', 'andret', 'andret', 'rahal',
               'bobby', 'lola', 'lola', 'ford', 'ford', 'reynard', 'merced',
               'christ', 'benz', 'motor', 'fittipald', 'grand', 'prix',
               'finish', 'finish', 'michael', 'win', 'vancouv', 'vancouv',
               'indycar', 'indycar']
    assert res.loc[26188].W.split(' ') == res_ref
