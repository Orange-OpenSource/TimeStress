# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

import pytest

from ke_utils.glob_core import TimeUnit
from wikidata_tools.core import Date, Entity, Quantity, Relation, Triple, TripleComp
from verb_tools.core import EntityFormatter, Relation2TemplatesVerbalizer, Template, Relation2TemplatesVerbalizer, VerbalizeConfig
from verb_tools.options import BlankOut, DateIndicator, DatePrecision, DateStyle, QuantityPrecision


@pytest.fixture
def verbalizer() -> Relation2TemplatesVerbalizer:
    president = Relation('president')
    spouse = Relation('spouse')
    capital = Relation('capital')
    pop = Relation('population')
    dob = Relation('date_of_birth')
    wrong = Relation('wrong')
    template_scores = {
        president : {
            Template.from_string('[OBJ] is the president of [SUB]', president) : 100,
            Template.from_string('The president of [SUB] is [OBJ]', president) : 50,
        },
        spouse : {
            Template.from_string('The spouse of [SUB] is [OBJ]', spouse) : 20,
            Template.from_string('[SUB] is linked to [OBJ] by marriage.', spouse) : 5,
            Template.from_string('[OBJ] is the spouse of [SUB]', spouse) : 10
        },
        capital : {
            Template.from_string('The capital of [SUB] is [OBJ]', capital) : 100,
            Template.from_string('[OBJ] is the capital of [SUB]', capital) : 50
        },
        pop: {
            Template.from_string('The population of [SUB] is [OBJ]', pop) : 1
        },
        dob:{
            Template.from_string('The date of birth of [SUB] is [OBJ]', dob) : 1
        },
        wrong:{
            Template.from_string('Wrong template [SUB] ladies and gentlemen [OBJ]', dob) : 1
        }
    }
    v = Relation2TemplatesVerbalizer(template_scores)
    # v.load_mlconjug3()
    # v.load_spacy_model()
    return v

def test_from_string():
    test = Relation('test')
    template = Template.from_string(' [SUB] is [OBJ]', test)
    assert template.relation == test
    assert template._template == " [SUB] is [OBJ]"
    assert template.text == " [SUB] is [OBJ]"

def test_simple(verbalizer : Relation2TemplatesVerbalizer):
    triple = Triple(Entity('E1', 'E1'), Relation('spouse'), Entity('E2', 'E2'))
    verbs = list(verbalizer.verbalize(triple))
    assert all(v.full for v in verbs)
    assert verbs[0].text == "The spouse of E1 is E2"
    assert verbs[1].text == "E2 is the spouse of E1"
    assert verbs[2].text == "E1 is linked to E2 by marriage."

    # verbs are copies not references
    assert verbs[0] != verbalizer.template_scores[triple.relation][0][0]
    assert verbs[0] is not verbalizer.template_scores[triple.relation][0][0]

    config = VerbalizeConfig(blank_outs=(BlankOut.SUBJECT,))
    verbs = verbalizer.verbalize(triple, config)
    assert all(not x.full for x in verbs)

    config = VerbalizeConfig(ends_with=TripleComp.SUBJECT)
    verbs = verbalizer.verbalize(triple, config)
    assert len(verbs) == 1 and verbs[0].text == "E2 is the spouse of E1"

    config = VerbalizeConfig(ends_with=TripleComp.OBJECT)
    verbs = verbalizer.verbalize(triple, config)
    assert len(verbs) == 1 and verbs[0].text == "The spouse of E1 is E2"

    config = VerbalizeConfig(index_template=(1,))
    verbs = verbalizer.verbalize(triple, config)
    assert len(verbs) == 1 and verbs[0].text == "E2 is the spouse of E1"

def test_quantity(verbalizer : Relation2TemplatesVerbalizer):
    triple = Triple(Entity('E1'), Relation('population'), Quantity(12345678))
    
    config = VerbalizeConfig(quantity_precision=None)
    verb = list(verbalizer.verbalize(triple, config))[0]
    assert verb.text == "The population of E1 is 12345678"

    config = VerbalizeConfig(quantity_precision=QuantityPrecision(3))
    verb = list(verbalizer.verbalize(triple, config))[0]
    assert verb.text == "The population of E1 is 12.3 M"

def test_date(verbalizer : Relation2TemplatesVerbalizer):
    triple = Triple(Entity('E1'), Relation('date_of_birth'), Date.from_string('2001-02-05'))
    
    config = VerbalizeConfig(date_precision=DatePrecision(TimeUnit.YEAR), date_style=DateStyle.MONTH_LETTER_CLASSIC)
    verb = list(verbalizer.verbalize(triple, config))[0]
    assert verb.text == "The date of birth of E1 is 2001"
    
    config = VerbalizeConfig(date_precision=DatePrecision(TimeUnit.MONTH), date_style=DateStyle.MONTH_LETTER_CLASSIC)
    verb = list(verbalizer.verbalize(triple, config))[0]
    assert verb.text == "The date of birth of E1 is February 2001"

    config = VerbalizeConfig(date_precision=DatePrecision(TimeUnit.DAY), date_style=DateStyle.MONTH_LETTER_CLASSIC)
    verb = list(verbalizer.verbalize(triple, config))[0]
    assert verb.text == "The date of birth of E1 is 5 February 2001"

    config = VerbalizeConfig(date_precision=DatePrecision(TimeUnit.DAY), date_style=DateStyle.MONTH_LETTER_AMERICAN)
    verb = list(verbalizer.verbalize(triple, config))[0]
    assert verb.text == "The date of birth of E1 is February 5, 2001"


    triple = Triple(Entity('E1'), Relation('date_of_birth'), Date.from_string('2001-02'))
    config = VerbalizeConfig(date_precision=DatePrecision(TimeUnit.MONTH), date_style=DateStyle.RAW_ATTACHED)
    verb = list(verbalizer.verbalize(triple, config))[0]
    assert verb.text == "The date of birth of E1 is 200102"

    config = VerbalizeConfig(date_precision=None, date_style=DateStyle.RAW_HYPHEN)
    verb = list(verbalizer.verbalize(triple, config))[0]
    assert verb.text == "The date of birth of E1 is 2001-02"

# Deprecated test
# def test_wrong_verbalization(verbalizer : Relation2TemplatesVerbalizer):
#     triple = Triple(Entity('E1'), Relation('wrong'), Entity('E5'))
#     try:
#         out = verbalizer.verbalize(triple, VerbalizeConfig(max_num_verbs=None))
#         assert False
#     except VerbError:
#         pass

#     triple = Triple(Entity('E1'), Relation('nonexistant'), Entity('E5'))
#     try:
#         verbalizer.verbalize(triple, VerbalizeConfig(max_num_verbs=None))
#         assert False
#     except TemplateNotFound:
#         pass

def test_time_indicator(verbalizer : Relation2TemplatesVerbalizer):
    triple = Triple(Entity('E1'), Relation('population'), Quantity(12345678))
    
    config = VerbalizeConfig(temporal_indicator=DateIndicator(Date.from_string('2001-01')), quantity_precision=None)
    verb = list(verbalizer.verbalize(triple, config))[0]
    assert verb.text == "In January 2001, the population of E1 is 12345678"

    config = VerbalizeConfig(temporal_indicator=DateIndicator(Date.from_string('2001-01')), quantity_precision=None,
                             date_precision_temporal_indicator=DatePrecision(TimeUnit.YEAR))
    verb = list(verbalizer.verbalize(triple, config))[0]
    assert verb.text == "In 2001, the population of E1 is 12345678"

def test_entity_formatter():
    config = VerbalizeConfig(date_precision=DatePrecision(TimeUnit.YEAR), date_style=DateStyle.MONTH_LETTER_CLASSIC)
    formatter = EntityFormatter(config)
    res = formatter.format_date_static(Date.from_string('1900'), DatePrecision(TimeUnit.YEAR), DateStyle.MONTH_LETTER_CLASSIC)
    assert res == '1900'