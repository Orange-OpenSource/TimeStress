# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from itertools import chain, combinations_with_replacement
import random
import tempfile
from wikidata_tools._wikidata_test import MiniTempWikidata, MiniWikidata
from wikidata_tools.core import (
    Date,
    Entity,
    Interval,
    Quantity,
    Relation,
    String,
    TimedTriple,
    TimedTripleQuery,
    Triple,
    TripleQuery,
)
from wikidata_tools.wikidata import InMemoryWikidata, TempWikidata, Wikidata, WikidataPrepStage
import pytest



def get_raw_tempwikitest() -> MiniTempWikidata:
    wd = MiniTempWikidata(time='20230227', stage=WikidataPrepStage.ALMOST_RAW)
    wd.build(confirm=False)
    return wd
raw_tempwikitest = get_raw_tempwikitest()

def get_prep_tempwikitest():
    wd = MiniTempWikidata(time='20230227', stage=WikidataPrepStage.PREPROCESSED)
    wd.build(confirm=False)
    return wd
prep_tempwikitest = get_prep_tempwikitest()

def get_raw_wikitest() -> MiniWikidata:
    wd = MiniWikidata(time='20230227', stage=WikidataPrepStage.ALMOST_RAW)
    wd.build(confirm=False)
    return wd
raw_wikitest = get_raw_wikitest()

def get_prep_wikitest():
    wd = MiniWikidata(time='20230227', stage=WikidataPrepStage.PREPROCESSED)
    wd.build(confirm=False)
    return wd
prep_wikitest = get_prep_wikitest()

# Relation triple examples
france = Entity('Q142')
head_of_state = Relation('P35')
emmanuel_macron = Entity("Q3052772")

estonia = Entity('Q191')
capital = Relation('P36')
tallinn = Entity('Q1770')

belgium, china = Entity('Q31'), Entity('Q148')


list_wds_mongo = [raw_wikitest, prep_wikitest, raw_tempwikitest, prep_tempwikitest]
def get_wds():
    yield from list_wds_mongo
    for wd in list_wds_mongo:
        wd: Wikidata
        triples = list(wd)
        aliases = {x: wd.get_all_names_of_entity([x])[0] for x in [china, belgium]}
        # Only Belgium amd China aliases were provided because they are the only aliases that will be used in the tests
        wd.inject_info(triples)
        relation2examples = {
            head_of_state: Triple(france, head_of_state, emmanuel_macron),
            capital: Triple(estonia, capital, tallinn)
        }
        wdim = InMemoryWikidata(triples, aliases, relation2examples, isinstance(wd, TempWikidata), time=wd.time, stage=wd.stage)
        yield wdim
list_wds = list(get_wds())
pairs_raw_prep = [(list_wds[i], list_wds[i+1]) for i in range(0, len(list_wds), 2)]
list_temp_wds = [wd for wd in list_wds if isinstance(wd, MiniTempWikidata) or isinstance(wd, InMemoryWikidata) and wd.use_temporal_triples]
list_inmem_wds = [wd for wd in list_wds if isinstance(wd, InMemoryWikidata)]

@pytest.fixture
def triple_example():
    # Triple = (Belgium, instance of, sovereign state)
    subject = Entity("Q31")
    relation = Relation("P31")
    object = Entity("Q3624078")
    triple = Triple(subject, relation, object)
    return triple


@pytest.fixture
def pop_triple_example():
    # Triple = (Belgium, population, 11584008, 1 January 2022)
    subject = Entity("Q31")
    relation = Relation("P1082")
    object = Quantity(11584008)
    triple = Triple(subject, relation, object)
    return triple




@pytest.mark.parametrize('wd', list_wds)
def test_contains(wd: MiniWikidata, triple_example: Triple):
    assert wd.contains(triple_example)

    query = TripleQuery.from_triple(triple_example)
    query_res = list(wd.find(query))
    assert len(query_res) == 1
    res_triple = query_res[0]
    assert type(res_triple.subject) is Entity
    assert type(res_triple.relation) is Relation
    assert type(res_triple.object) is Entity
    if isinstance(wd, MiniTempWikidata):
        assert type(res_triple) is TimedTriple
        assert (
            res_triple.valid_between.start is None
            and res_triple.valid_between.end is None
        )

    query.object = None
    assert len(list(wd.find(query))) == 7

@pytest.mark.parametrize('wd', list_wds)
def test_quantity(wd: MiniWikidata, pop_triple_example: Triple):
    query = TripleQuery.from_triple(pop_triple_example)
    query_res = list(wd.find(query))
    assert len(query_res) == 1
    res_triple = query_res[0]
    assert type(res_triple.subject) is Entity
    assert type(res_triple.relation) is Relation
    assert type(res_triple.object) is Quantity
    assert res_triple.object.value == 11584008
    if isinstance(wd, MiniTempWikidata):
        assert type(res_triple) is TimedTriple
        assert res_triple.valid_between.is_point()
        assert res_triple.valid_between.start == Date.from_string("2022-01-01")

    query.object = None
    res = list(wd.find(query))
    assert len(res) > 10 if wd.stage == WikidataPrepStage.ALMOST_RAW else len(res) == 1

@pytest.mark.parametrize('wd', list_wds)
def test_batch_find(wd: MiniWikidata):
    random.seed(4562)
    triples = [t for _, t in zip(range(10000), wd)]
    subjects = random.sample(list(set(x.subject for x in triples)), 10)
    relations = random.sample(
        list(set(x.relation for x in triples if x.subject in subjects)), 10
    )
    objects = random.sample(
        list(
            set(
                x.object
                for x in triples
                if x.subject in subjects and x.relation in relations
            )
        ),
        10,
    )

    res = list(wd.find(TripleQuery(subject=subjects)))
    assert len(res) > 0
    assert all(triple.subject in subjects for triple in res)

    res = list(wd.find(TripleQuery(subject=subjects, relation=relations)))
    assert len(res) > 0
    assert all(
        triple.subject in subjects and triple.relation in relations
        for triple in res
    )

    res = list(
        wd.find(TripleQuery(subject=subjects, relation=relations, object=objects))
    )
    assert len(res) > 0
    assert all(
        triple.subject in subjects
        and triple.relation in relations
        and triple.object in objects
        for triple in res
    )

@pytest.mark.parametrize('wd', list_wds)
def test_iterate(wd: MiniWikidata):
    for triple in wd:
        assert isinstance(triple, Triple)
        break

@pytest.mark.parametrize('wd', list_wds_mongo)
def test_built(wd: MiniWikidata):
    assert wd.built()

@pytest.mark.parametrize('pair_raw_prep', pairs_raw_prep)
def test_number_of_subjects(pair_raw_prep: tuple[MiniWikidata, MiniWikidata]):
    raw, prep = pair_raw_prep
    assert raw.number_of_subjects() > 10
    assert prep.number_of_subjects() < raw.number_of_subjects()

@pytest.mark.parametrize('wd', list_wds)
def test_iterate_subject(wd: MiniWikidata):
    n = 0
    for sub in wd.iterate_subjects():
        n += 1
        assert isinstance(sub, Entity)
    assert wd.number_of_subjects() == n

@pytest.mark.parametrize('wd', list_wds)
def test_sample_subjects(wd: MiniWikidata):
    assert len(list(wd.sample_subjects(10))) == 10

@pytest.mark.parametrize('wd', list_temp_wds)
def test_period(wd: TempWikidata):
    # Triple (Belgium, head of state,  Philippe of Belgium)
    subject = Entity("Q31")
    relation = Relation("P35")
    object = Entity("Q155004")

    query = TripleQuery(subject, relation, object)
    res = list(wd.find(query))
    assert len(res) == 1
    rest = res[0]
    assert rest.subject == subject
    assert rest.relation == relation
    assert rest.object == object
    assert rest.valid_between.start == Date.from_string("2013-07-21")
    assert rest.valid_between.end is None

@pytest.mark.parametrize('wd', list_wds)
def test_inject(wd: MiniWikidata):
    ent = Entity("Q31")
    qu = Quantity(2712389)
    qu_copy = Quantity(2712389)
    s = String("HAHAHAH")
    s_copy = String("HAHAHAH")
    entities = [qu, ent, s, ent, qu]
    wd.inject_info(entities)
    for e in entities:
        if type(e) is Quantity:
            assert (
                e.id == qu_copy.id
                and e.label == qu_copy.label
                and e.description == qu_copy.description
            )
        elif type(e) is String:
            assert (
                e.id == s_copy.id
                and e.label == s_copy.label
                and e.description == s_copy.description
            )
        else:
            assert (
                ent.id == "Q31"
                and ent.label == "Belgium"
                and ent.description == "country in western Europe"
            )

    # Batch injection
    ent = Entity("Q31")
    l = [ent] * 50
    wd.inject_info(l)
    assert (
        l[0].id == "Q31"
        and l[0].label == "Belgium"
        and l[0].description == "country in western Europe"
    )
    for e in l[1:]:
        assert e is ent

@pytest.mark.parametrize('wd', list_wds)
def test_find_small(wd: Wikidata):
    query = TripleQuery([Entity("Q31")], None, None)
    assert len(list(wd.find(query))) > 0

@pytest.mark.parametrize('wd', list_wds)
def test_find_from_label(wd: Wikidata):
    entities = wd.find_from_label(['Belgium'])['Belgium']
    assert len(entities) == 1
    assert entities[0].id == 'Q31'

    entities = wd.find_from_label(['BEL'])['BEL']
    assert len(entities) == 1
    assert entities[0].id == 'Q31'

    entities = wd.find_from_label(['China'])['China']
    assert len(entities) == 1
    assert entities[0].id == 'Q148'

    entities = wd.find_from_label(['CHN'])['CHN']
    assert len(entities) == 1
    assert entities[0].id == 'Q148'

    res = wd.find_from_label(['Belgium', 'CHN'])
    assert len(res) == 2
    entities = res['Belgium']
    assert len(entities) == 1
    assert entities[0].id == 'Q31'
    entities = res['CHN']
    assert len(entities) == 1
    assert entities[0].id == 'Q148'

@pytest.mark.parametrize('wd', list_wds)
def test_get_all_names_of_entity(wd: Wikidata):
    belgium, china = Entity('Q31'), Entity('Q148')
    aliases = wd.get_all_names_of_entity([belgium, china])
    expected_res = [
        [
            'Belgium',
            'Kingdom of Belgium',
            'BEL',
            'be',
            '🇧🇪',
            'BE',
            'België',
            'Koninkrijk België',
            'Royaume de Belgique',
            'Belgique',
            'Königreich Belgien',
            'Belgien'
        ],
        [
            "People's Republic of China",
            'CN',
            'PR China',
            'PRC',
            'cn',
            'CHN',
            '🇨🇳',
            'China PR',
            'Mainland China',
            'China',
            'RPC'
        ]
    ]
    assert aliases == expected_res

    aliases = wd.get_all_names_of_entity([china, belgium])
    assert aliases == expected_res[::-1]

    aliases = wd.get_all_names_of_entity([china, belgium]*2)
    assert aliases == expected_res[::-1]*2


@pytest.mark.parametrize('wd', list_inmem_wds)
def test_save_load_inmemory(wd: InMemoryWikidata):
    triples_original = set(list(wd))
    with tempfile.TemporaryDirectory() as tmpdirname:
        wd.save(tmpdirname)
        wd2 = InMemoryWikidata.from_disk(tmpdirname)
        triples_loaded = set(list(wd2))
    assert triples_loaded == triples_original
    assert wd2.time == wd.time and wd2.stage == wd.stage and wd2.use_temporal_triples == wd.use_temporal_triples
    assert not wd2._first_query_add_relation_examples and not wd2._first_query_add_aliases and not wd2._first_query_add_triples


@pytest.mark.parametrize('wd', list_temp_wds)
def test_find_valid_at(wd: Wikidata):
    belgium, head_of_state = Entity('Q31'), Relation("P35")
    test_dates_neg = [
        Interval(Date.from_string('2013'), Date.from_string('2020')), 
        Date.from_string('2013-07'), 
        Date.from_string('2013-07-20'), 
        Date.from_string('2013'), 
        Date.from_string('2012'),
        Interval(Date.from_string('2013-07'), Date.from_string('2020-06-24'))
    ]

    for date in test_dates_neg:
        query = TimedTripleQuery(belgium, head_of_state, None, valid_at=date)
        objects = list(tr.object for tr in wd.find(query))
        assert Entity("Q155004") not in objects

    test_dates_pos = [
        "present", wd.time_date, 
        Date.from_string('2020'), 
        Date.from_string('2020-12-01'), 
        Date.from_string('2013-07-22'),
        Interval(Date.from_string('2014'), Date.from_string('2020')),
        Interval(Date.from_string('2015'), Date.from_string('2016'))
    ]
    all_tests_pos = chain(
        test_dates_pos,
        combinations_with_replacement(test_dates_pos, r=2),
        combinations_with_replacement(test_dates_pos, r=3)
    )
    for date in all_tests_pos:
        query = TimedTripleQuery(belgium, head_of_state, None, valid_at=date)
        triples = list(wd.find(query))
        assert len(triples) == 1 and triples[0].object.id == "Q155004"

    test_date_transition = Date.from_string('2013-07-21')
    query = TimedTripleQuery(belgium, head_of_state, None, valid_at=test_date_transition)
    objects = list(tr.object for tr in wd.find(query))
    assert len(objects) == 0

@pytest.mark.parametrize('wd', list_inmem_wds)
def test_retrieve_example(wd : Wikidata):
    example = wd.retrieve_example(head_of_state)
    assert example == Triple(france, head_of_state, emmanuel_macron)

    example = wd.retrieve_example(capital)
    assert example == Triple(estonia, capital, tallinn)

    example = wd.retrieve_example(Relation("PBLABLA"))
    assert example is None
    
@pytest.mark.parametrize('wd', list_temp_wds)
def test_get_all_types_of_entity(wd : Wikidata):
    types_belgium = [
        'Q3624078',
        'Q43702',
        'Q6256',
        'Q20181813', # colonial power
        'Q185441',
        'Q1250464',
        'Q113489728'
    ]
    types_china = ['Q3624078', 'Q842112', 'Q859563', 'Q1520223', 'Q6256', 'Q465613', 'Q118365', 'Q15634554', 'Q849866']

    types_belgium = set(map(Entity, types_belgium))
    types_china = set(map(Entity, types_china))
    
    types = [set(l) for l in wd.get_all_types_of_entity(belgium)]
    assert types == [types_belgium]

    types = [set(l) for l in wd.get_all_types_of_entity(belgium, valid_at='present')]
    assert types == [types_belgium - set([Entity('Q20181813')])]

    types = [set(l) for l in wd.get_all_types_of_entity([belgium])]
    assert types == [types_belgium]

    types = [set(l) for l in wd.get_all_types_of_entity([belgium, belgium])]
    assert types == [types_belgium, types_belgium]

    types = [set(l) for l in wd.get_all_types_of_entity(china)]
    assert types == [types_china]

    types = [set(l) for l in wd.get_all_types_of_entity([china, belgium])]
    assert types == [types_china, types_belgium]