# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from ke_utils.general import search_str_in_tokens
from pytest import fixture
import pytest

def text_tokens_spans():
    tokens = ['Paris', ' is', ' the', ' the', ' capital', ' of  ', 'France']
    text = "is t"
    spans = [(1,3)]
    yield text, tokens, spans

    text = "Par"
    spans = [(0,1)]
    yield text, tokens, spans

    text = "Paris"
    spans = [(0,1)]
    yield text, tokens, spans

    text = "rance"
    spans = [(6,7)]
    yield text, tokens, spans

    text = " France"
    spans = [(5,7)]
    yield text, tokens, spans

    text = "France"
    spans = [(6,7)]
    yield text, tokens, spans

    text = " is"
    spans = [(1,2),]
    yield text, tokens, spans

    text = " the"
    spans = [(2,3),(3,4)]
    yield text, tokens, spans

    text = " Australia (continent)"
    spans = [(7,12)]
    tokens = ['The', ' location', ' of', ' World', ' War', ' I', ' was', ' Australia', ' (', 'cont', 'inent', ')']
    yield text, tokens, spans


    # 港女版孟慶樹
    text = " 香港女"
    spans = [(0,3)]
    tokens = [' 香', '港', '女', '版', '孟', '慶樹']
    yield text, tokens, spans


@pytest.mark.parametrize("text, tokens, spans", text_tokens_spans())
def test_search_str_in_list_tokens(text, tokens, spans):
    spans_pred = search_str_in_tokens(text, tokens)
    assert spans_pred == spans

