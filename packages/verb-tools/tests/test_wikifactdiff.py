# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from verb_tools.wikifactdiff import WFDVerbVersion, WikiFactDiffVerbalizer
from wikidata_tools.core import Entity, Relation, Triple

def test_load():
    france = Entity('Q142')
    paris = Entity('Q90')
    capital = Relation('P36')
    triple = Triple(france, capital, paris)
    v1 = WikiFactDiffVerbalizer(version=WFDVerbVersion.V1, use_cache=True)
    v1.verbalize(triple)
    v2 = WikiFactDiffVerbalizer(version=WFDVerbVersion.V2, use_cache=True)
    v2.verbalize(triple)