# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from ke_utils.glob_core import JSONable
from ke_utils.hash import consistent_hash


# def test_version():
#     x = consistent_hash(dict(a=1,b=2))
#     assert x[0] == _VERSION


def test_dict():
    a = consistent_hash(dict(a=1,b=2))
    b = consistent_hash(dict(a=1,b=2))
    c = consistent_hash(dict(a=1,b=2,c=3))
    d = consistent_hash(dict(c=3,a=1,b=2))
    e = consistent_hash(dict(c=3,b=2,a=1))
    assert a == b
    assert all(a != x for x in (c,d,e))
    assert c == d == e

def test_to_full_json():
    class Person(JSONable):
        def __init__(self, name : str, age : int):
            super().__init__()
            self.name = name
            self.age = age
        
        def _to_json(self):
            return {
                'name': self.name,
                "age": self.age
            }
    a = consistent_hash(Person("A", 24))
    b = consistent_hash(Person('B', 26))
    c = consistent_hash(Person('A', 26))
    d = consistent_hash(Person('A', 24))
    
    assert a == d
    assert a != b and a != c
    assert c != b