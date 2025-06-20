# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from wikidata_tools.core import Date, Quantity, enable_level_heterogeneous_comparison
from ke_utils.glob_core import TimeUnit
import numpy as np
import pytest

def test_date():
    np_date = np.datetime64("2020-01-15", "D")
    d = Date(np_date)
    assert d.level == TimeUnit.DAY
    assert d.year == 2020
    assert d.month == 1
    assert d.day == 15

    np_date = np.datetime64("2020-01", "M")
    d = Date(np_date)
    assert d.level == TimeUnit.MONTH
    assert d.year == 2020
    assert d.month == 1
    assert d.day is None

    np_date = np.datetime64("2020", "Y")
    d = Date(np_date)
    assert d.level == TimeUnit.YEAR
    assert d.year == 2020
    assert d.month is None
    assert d.day is None

    d = Date.from_string("2020-01-15")
    assert d.level == TimeUnit.DAY
    assert d.year == 2020
    assert d.month == 1
    assert d.day == 15

    d = Date.from_string("2020-01")
    assert d.level == TimeUnit.MONTH
    assert d.year == 2020
    assert d.month == 1
    assert d.day is None

    d = Date.from_string("2020")
    assert d.level == TimeUnit.YEAR
    assert d.year == 2020
    assert d.month is None
    assert d.day is None


def test_date_equal():
    # Test equal
    d1 = Date.from_string("2020-01-15")
    d2 = Date.from_string("2020-01-15")
    assert d1 == d2
    assert not d1 != d2

    d3 = Date.from_string('2020-01')
    assert d3 != d2 and d3 != d1

    d1, d2 = Date(np.datetime64('2021', 'Y')), Date(np.datetime64('2021-01', 'M'))
    assert d1 != d2


def test_neg_date():
    d1 = Date.from_string("-2020-01-15")
    assert d1.year == -2020


def test_eq_date_quantity():
    d1 = Date.from_string("2020-01-15")
    assert d1 != Quantity(1)
    assert d1 not in [Quantity(1)]


def heteregeneous_dates():
    d1 = Date.from_string("2020")
    d2 = Date.from_string('2020-01')
    yield d1, d2
    d2 = Date.from_string('2020-02')
    yield d1, d2
    d2 = Date.from_string('2020-06')
    yield d1, d2
    d2 = Date.from_string('2020-11')
    yield d1, d2
    d2 = Date.from_string('2020-12')
    yield d1, d2

    d2 = Date.from_string('2020-12-31')
    yield d1, d2
    d2 = Date.from_string('2020-01-31')
    yield d1, d2

@pytest.mark.parametrize(['d1', 'd2'], heteregeneous_dates())
def test_heterogeneous_dates(d1 : Date, d2 : Date):
    with enable_level_heterogeneous_comparison():
        assert not (d1 < d2) and not (d1 > d2) and d1 != d2 and not (d1 <= d2) and not (d1 >= d2)
        assert d1.includes(d2) and not d2.includes(d1)