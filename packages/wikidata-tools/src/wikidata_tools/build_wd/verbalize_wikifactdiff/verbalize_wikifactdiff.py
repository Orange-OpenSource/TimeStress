# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from collections import Counter, defaultdict
from wikidata_tools.build_wd.utils.wd import (
    db,
    get_info_wikidata,
    get_value,
    get_single_valued_properties,
)
from wikidata_tools.build_wd.gpt3_5_verbalization.verbalize_wikidata import (
    PropertyInfoRetriever,
    Entity,
    Property,
    Literal,
    KnowledgeTriple,
)
from typing import Union
from wikidata_tools.build_wd.utils.ku import Feature, classify_algorithm_lite
from wikidata_tools.build_wd.verbalize_wikifactdiff.utils import Verbalizer
from multiprocessing.pool import ThreadPool
from threading import Lock
import random
import json
import os.path as osp

from wikidata_tools.build_wd.config import STORAGE_FOLDER

SAVE_PATH = osp.join(STORAGE_FOLDER, "wikifactdiff_dirty.jsonl")


def value_to_object(value: dict) -> Union[Literal, Entity]:
    obj, t = get_value(
        value,
        get_info_dict_for_entities=True,
        add_type=True,
        add_unit=True,
        version="old",
    )
    if obj is None or len(obj) == 0:
        obj, t = get_value(
            value,
            get_info_dict_for_entities=True,
            add_type=True,
            add_unit=True,
            version="new",
        )
        if obj is None or len(obj) == 0:
            return None
    if isinstance(obj, dict):
        obj = Entity(obj["id"], obj["name"], obj["description"])
    else:
        obj = Literal(obj, t)
    return obj


class Finish(Exception):
    pass


def get_same_entity_triples(
    ent_id: str, rel_to_exclude: Property, ent_is_ph_new: bool, k=5
) -> list[KnowledgeTriple]:
    if ent_is_ph_new:
        return []
    ent_dict = db["wikidata_old_prep"].find_one(
        {"_id": ent_id}, {"claims": 1, "label": 1, "description": 1}
    )
    if ent_dict is None:
        return []
    claims = ent_dict["claims"]
    claims.pop(rel_to_exclude, None)
    ent = Entity(ent_dict["_id"], ent_dict["label"], ent_dict.get("description", None))
    triples = []
    closes = tuple(claims.items())
    for prop_id, snaks in random.sample(closes, k=k) if len(closes) > k else closes:
        prop = prop_info_ret.retrieve(prop_id)
        prop = Property(prop["id"], prop["name"], prop["description"])
        objects = [
            value_to_object(snak["mainsnak"]["datavalue"]["value"]) for snak in snaks
        ]
        if any(obj is None for obj in objects):
            continue
        triples.extend([KnowledgeTriple(ent, prop, obj) for obj in objects])

    return triples


def get_nearest_triples(
    knn, ent_id: str, rel: Property, k=500, banned_ent: list[str] = []
) -> list[KnowledgeTriple]:
    # Get the 10 most similar entities to ent_id that have relation rel and collect the group (neighbor_ent, rel) and put all triples a list and return it
    vector = knn.get_vectors(ent_id)
    if vector is None:
        return [], []

    neighbor_entities, distances = knn.get_nearest(vector, k=k)
    to_remove = [
        i
        for i in range(len(neighbor_entities))
        if neighbor_entities[i] == ent_id or neighbor_entities[i] in banned_ent
    ]
    neighbor_entities = [
        x for i, x in enumerate(neighbor_entities) if i not in to_remove
    ]
    distances = [x for i, x in enumerate(distances) if i not in to_remove]

    step = 20
    result = []
    dists = []
    n_neighobrs_found = 0
    try:
        for i in range(0, len(neighbor_entities), step):
            ents = neighbor_entities[i : i + step]
            ds = distances[i : i + step]
            ent_dicts = [
                x
                for x in db["wikidata_old_prep"].find(
                    {"_id": {"$in": ents}, "claims.%s" % rel.id: {"$exists": 1}},
                    {"claims.%s" % rel.id: 1, "label": 1, "description": 1},
                )
            ]

            # Remove entities with no labels
            ent_dicts = [x for x in ent_dicts if x.get("label", None) is not None]

            ents_found = set(x["_id"] for x in ent_dicts)
            ds_found = []
            for i in range(len(ds)):
                if ents[i] in ents_found:
                    ds_found.append(ds[i])
            for ent, d in zip(ent_dicts, ds_found):
                prop_values = ent["claims"].get(rel.id, None)
                n_neighobrs_found += 1
                for snak in prop_values:
                    feature = Feature(snak)
                    if classify_algorithm_lite(feature, version="old") != "keep":
                        continue
                    sub = Entity(
                        ent["_id"], ent.get("label", None), ent.get("description", None)
                    )
                    obj = value_to_object(snak["mainsnak"]["datavalue"]["value"])
                    if obj is None:
                        n_neighobrs_found -= 1
                        break
                    triple = KnowledgeTriple(sub, rel, obj)
                    result.append(triple)
                    dists.append(float(d))
                if n_neighobrs_found >= 10:
                    raise Finish
    except Finish:
        pass
    return result, dists


def _is_replace(objects: list[dict]):
    c = Counter([x["decision"] for x in objects])
    return len(c) == 2 and c["learn"] == 1 and c["forget"] == 1


def verbalize_and_compact(
    verbalizer: Verbalizer,
    triples_decisions: list[tuple[KnowledgeTriple, str]],
    neighbor_triples: list[KnowledgeTriple],
    distances: list[float],
    close_triples: list[list[KnowledgeTriple]],
    ent_is_ph_new: bool,
    rel_is_functional: bool,
    ent_imp: float,
) -> dict:
    triples, decisions = list(zip(*triples_decisions))
    triple = triples[0]

    verbs = verbalizer.verbalize(triple, exclude=["subject", "object"])
    if verbs is None:
        return None
    rev_verbs = verbalizer.verbalize(
        KnowledgeTriple(triple.object, triple.relation, triple.subject),
        exclude=["object", "subject"],
        reverse=True,
    )
    if rev_verbs:
        reverse_prompt = rev_verbs[0]
    else:
        reverse_prompt = None

    objects_decisions = []
    for t, d in zip(triples, decisions):
        dd = t.object.to_dict()
        dd["original_label"] = dd["label"]
        dd["label"] = verbalizer.human_formatter.format(t.object)
        dd["decision"] = d
        objects_decisions.append(dd)

    neighbors = defaultdict(list)
    for t, dist in zip(neighbor_triples, distances):
        verbs_t = verbalizer.verbalize(t, exclude=["object", "subject"])
        dd = t.object.to_dict()
        dd["original_label"] = dd["label"]
        dd["label"] = verbalizer.human_formatter.format(t.object)
        neighbors[t.subject].append(
            {
                "object": dd,
                "prompt": random.choice(verbs_t),
            }
        )
    neighbors = [
        {"objects": v, "subject": k.to_dict(), "dist": dist}
        for k, v in neighbors.items()
    ]

    closes = defaultdict(list)
    relations_to_skip = []
    for t in close_triples:
        verbs_t = verbalizer.verbalize(t, exclude=["object", "subject"])
        if verbs_t is None:
            relations_to_skip.append(t.relation)
            continue
        dd = t.object.to_dict()
        dd["original_label"] = dd["label"]
        dd["label"] = verbalizer.human_formatter.format(t.object)
        closes[t.relation].append(
            {
                "object": dd,
                "prompt": random.choice(verbs_t),
            }
        )
    closes = [
        {"relation": k.to_dict(), "objects": v}
        for k, v in closes.items()
        if k not in relations_to_skip
    ]

    instance = {
        "subject": triple.subject.to_dict(),
        "relation": triple.relation.to_dict(),
        "update_prompt": verbs[0],
        "generalization_prompts": verbs[1:],
        "closeness": closes,
        "subject_is_ph_new": ent_is_ph_new,
        "relation_is_temp_func": rel_is_functional,
        "is_replace": _is_replace(objects_decisions),
        "subject_popularity": ent_imp,
        "objects": objects_decisions,
        "neighborhood": neighbors,
    }
    subject_alias = entity_aliases.get(triple.subject.id, None)
    if subject_alias is not None:
        instance["subject"]["alias"] = subject_alias
    for obj in instance["objects"]:
        if "id" in obj:
            object_alias = entity_aliases.get(obj["id"], None)
            if object_alias is not None:
                obj["alias"] = object_alias
    if reverse_prompt is not None:
        instance["reverse_prompt"] = reverse_prompt
    return instance


def process_one_group(x):
    ent_id = x["_id"]["ent_id"]
    ent_ph_new = ent_id in ph_new_entities
    if ent_ph_new:
        ent_info = get_info_wikidata(ent_id, version="new")
    else:
        ent_info = get_info_wikidata(ent_id, version="old")
        if len(ent_info) == 0:
            ent_info = get_info_wikidata(ent_id, version="new")
    prop_id = x["_id"]["prop_id"]
    with lock:
        prop_info = prop_info_ret.retrieve(prop_id)
    sub = Entity(ent_info["id"], ent_info["name"], ent_info["description"])
    rel = Property(prop_info["id"], prop_info["name"], prop_info["description"])
    ent_ph_new = ent_id in ph_new_entities
    rel_is_functional = prop_id in functional_relations

    snaks = [y for y in x["snaks"]]
    triples_decisions = []
    banned_ent = []
    for snak in snaks:
        value = snak["value"]
        try:
            obj = value_to_object(value)
            if obj is None:
                break
            triple = KnowledgeTriple(sub, rel, obj)
            triples_decisions.append((triple, snak["decision"]))
            if snak["decision"] != "keep":
                if isinstance(obj, Entity):
                    banned_ent.append(obj.id)
        except:
            print("Serious error:", ent_info, prop_id)
            break
    if len(triples_decisions) != len(snaks):
        return None

    neighbor_triples, distances = get_nearest_triples(
        knn, ent_id, rel, banned_ent=banned_ent
    )
    close_triples = get_same_entity_triples(ent_id, rel, ent_ph_new)

    instance = verbalize_and_compact(
        verbalizer,
        triples_decisions,
        neighbor_triples,
        distances,
        close_triples,
        ent_ph_new,
        rel_is_functional,
        x["ent_imp"],
    )
    return instance


def main(force_reset: bool):
    # Retrieve progress
    seen_groups = set()
    if not force_reset:
        print("Retrieving already verbalized groups...")
        try:
            f = open(SAVE_PATH, "r")
            instances = [json.loads(x) for x in f.readlines()]
            seen_groups.update(
                (x["subject"]["id"], x["relation"]["id"]) for x in instances
            )
            f.close()
            print("%s groups retrieved." % len(seen_groups))
        except FileNotFoundError:
            print("File not found! Starting from scratch.")
            pass

    f = open(SAVE_PATH, "w" if force_reset else "a")

    cursor = db["wkd_fd"].find()
    n = db["wkd_fd"].estimated_document_count()
    knn.load_index()
    instances = []
    # thread_pool = ThreadPool(8)
    for i, instance in enumerate(cursor, 1):
        ent_id, prop_id = instance["_id"]["ent_id"], instance["_id"]["prop_id"]
        # Debug
        # if ent_id != 'Q615' or prop_id != 'P286':
        #     continue
        if (ent_id, prop_id) in seen_groups:
            continue
        instance = process_one_group(instance)
        if instance is None:
            continue
        # instances.append(instance)
        f.write(json.dumps(instance) + "\n")
        f.flush()
        if i % 50 == 0:
            print("Progress : %s/%s groups (%0.2f%%)" % (i, n, i / n * 100))
    f.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann_method",
        type=str,
        help='What Approximate nearest neighbor method to use? Can take two values : "sparse", "dense".',
        required=True,
        choices=["sparse", "dense"],
    )
    parser.add_argument(
        "--force_reset",
        action="store_true",
        help="Reset the process from the beginning (lose all precedent progress!)",
    )
    # parser.add_argument('option', type=str, help='What type of dataset you want to create? (For which experiment) Can take three values : "ku","nn_eval"', required=True)
    args = parser.parse_args()

    count_entities = 0
    ph_new_entities = set(x["_id"] for x in db["physically_new_entities"].find())
    functional_relations = set(get_single_valued_properties())
    if args.ann_method == "dense":
        from wikidata_tools.build_wd.find_neighbors.knn_dense import KNearestNeighbors
    else:
        from wikidata_tools.build_wd.find_neighbors.knn_sparse_nmslib import (
            KNearestNeighbors,
        )
    knn = KNearestNeighbors()
    verbalizer = Verbalizer()
    lock = Lock()
    prop_info_ret = PropertyInfoRetriever(versions=["old", "new"])

    entity_aliases = {}
    for x in open(osp.join(STORAGE_FOLDER, "old_wikidata_aliases.jsonl")):
        j = json.loads(x)
        if isinstance(j[1], dict):
            continue
        id_, label = j[0]
        # Take the first alias
        for al in j[1]:
            if al.lower() != label.lower():
                entity_aliases[id_] = al
                break

    main(args.force_reset)
