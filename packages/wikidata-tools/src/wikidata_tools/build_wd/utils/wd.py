# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

import sys
from typing import Union
from frozendict import frozendict
from pymongo import MongoClient
import tqdm
from wikidata_tools.build_wd.config import *
from collections import defaultdict
import numpy as np
import os.path as osp
import os
import json
from scipy.optimize import linear_sum_assignment
from wikidata_tools.build_wd.utils.general import run_in_process, subrel2str
from wikidata_tools.build_wd.utils.ku import get_point_in_time
from wikidata_tools.build_wd.utils.ku import Feature


client = MongoClient(MONGO_URL)
db = client["wiki"]
wikidata_new_json = db["wikidata_new_json"]
wikidata_old_json = db["wikidata_old_json"]
wikipedia_diff_json = db["wikipedia_diff_json"]
ignore_hash = hash("__special__")

WIKIDATA_DUMP_INDEX_PATH = osp.join(STORAGE_FOLDER, "wikidata_dumps_index.json")


def _validate_version_arg(version: str):
    assert version in (
        "old",
        "new",
    ), "argument version must be one of the following values : ('old', 'new)"


def process_property(entity: dict):
    """Clean inplace property dictionary in raw JSON dump of wikidata

    Args:
        entity (dict): property dictionary
    """
    try:
        entity["_id"] = entity["id"]
        del entity["id"]
    except KeyError:
        return False
    try:
        _tmp = entity["labels"]
        if "en" in _tmp:
            entity["label"] = _tmp["en"]["value"]
        else:
            k = next(iter(_tmp.keys()))
            entity["label"] = _tmp[k]["value"]
    except (KeyError, StopIteration):
        pass
    if "labels" in entity:
        del entity["labels"]

    try:
        _tmp = entity["descriptions"]
        if "en" in _tmp:
            entity["description"] = _tmp["en"]["value"]
        else:
            k = next(iter(_tmp.keys()))
            entity["description"] = _tmp[k]["value"]
    except (KeyError, StopIteration):
        pass
    if "descriptions" in entity:
        del entity["descriptions"]

    try:
        entity["aliases"] = tuple(x["value"] for x in entity["aliases"]["en"])
    except KeyError:
        pass

    try:
        entity["enwiki_title"] = entity["sitelinks"]["enwiki"]["title"]
    except KeyError:
        pass
    if "sitelinks" in entity:
        del entity["sitelinks"]

    for i in ("pageid", "ns", "title"):
        if i in entity:
            del entity[i]
    return True


def get_info_wikidata(
    entity_or_relation_id: Union[str, tuple[str]],
    validate_with_relations: list[str] = None,
    version="new",
    return_types_ids=False,
    batch_size=1000,
) -> dict:
    """Return id, name and description of an entity or relation and additionaly a type if it's an entity.

    The source of data is a mongoDB that contains a wikidata dump

    Args:
        entity_or_relation_id (str, tuple[str]): The ID of the entitiy/relation (Example : "Q1234" or "P413"). A tuple of str can be passed to retrieve for many entities at once
        validate_with_relations (list[str], optional): If the entity or relation does not possess all the relations in 'validate_with_relation' return 'unknown', 'unknown', 'unknown'. This is basically to be sure we are retrieving the right object by checking that it contains the relations that we are sure are there.
        version (str, optional): Can take two values : ['old', 'new']. Specify from which version of wikidata we should retrieve information.
        return_types_ids (bool, Optional): Whether to return the types' ids in the dictionary. Default to False.
        batch_size (int, optional) : Number of entities in one MongoDB request.

    Returns:
        dict : Infos or dict of infos where keys = IDs and values = Infos (if a list is passed)
    """
    only_one = isinstance(entity_or_relation_id, str)
    if only_one:
        entity_or_relation_id = (entity_or_relation_id,)
    else:
        entity_or_relation_id = tuple(set(entity_or_relation_id))
    if len(entity_or_relation_id) > batch_size:
        res = {}
        for i in range(0, len(entity_or_relation_id), batch_size):
            info = get_info_wikidata(entity_or_relation_id[i : i + batch_size])
            # Handle the case when len(entity_or_relation_id) == 1
            if isinstance(info, dict):
                res.update(info)
            else:
                res.update({entity_or_relation_id[0]: info})

    coll = db[adapt_name(version)]
    entity = list(coll.find({"_id": {"$in": entity_or_relation_id}}))
    entity = {x["_id"]: x for x in entity}
    for id_, d in entity.items():
        # WHY : I forgot to process properties when pushing Wikidata JSON dump to Wikidata
        if id_.startswith("P"):
            process_property(d)
    entity = {k: defaultdict(lambda: "", d) for k, d in entity.items()}
    entity = {
        k: x
        for k, x in entity.items()
        if not (
            not x
            or (
                validate_with_relations is not None
                and any(y not in x["claims"] for y in validate_with_relations)
            )
        )
    }
    res = {
        k: {"id": k, "name": x["label"], "description": x["description"]}
        for k, x in entity.items()
    }
    types_to_retrieve = {}
    for k, d in entity.items():
        if d["type"] == "item":
            try:
                instance_of = [
                    x["mainsnak"]["datavalue"]["value"]["id"]
                    for x in d["claims"]["P31"]
                ]
            except KeyError:
                instance_of = []
            types_to_retrieve[k] = instance_of
    instance_of = list(set(y for x in types_to_retrieve.values() for y in x))
    types = list(coll.find({"_id": {"$in": instance_of}}, {"label": 1}))
    types = {x["_id"]: x["label"] for x in types if "label" in x}
    for k, v in res.items():
        if k.startswith("P"):
            continue
        v["types"] = [types[x] if x in types else None for x in types_to_retrieve[k]]
        if return_types_ids:
            v["types_ids"] = types_to_retrieve[k]
    if only_one:
        if len(res) == 0:
            return {}
        return next(iter(res.values()))
    return res


def _util_get_value_return(value: str, type: str, add_type: bool):
    if add_type:
        return value, type
    return value


def get_unit_symbol(unit_id: str):
    # unit_id is an entity ID

    symb_lang = db["wikidata_new_json"].find_one(
        {"_id": unit_id},
        {
            "symb": "$claims.P5061.mainsnak.datavalue.value.text",
            "lang": "$claims.P5061.mainsnak.datavalue.value.language",
        },
    )
    if symb_lang is None:
        symb_lang = db["wikidata_old_json"].find_one(
            {"_id": unit_id},
            {
                "symb": "$claims.P5061.mainsnak.datavalue.value.text",
                "lang": "$claims.P5061.mainsnak.datavalue.value.language",
            },
        )
    if symb_lang is None:
        return None
    if "symb" not in symb_lang:
        return None

    for symb, lang in zip(symb_lang["symb"], symb_lang["lang"]):
        if lang == "en":
            return symb
    return symb_lang["symb"][0]


def get_value(
    value_dict: Union[dict, str],
    version: str = "new",
    retrieve_entity_name=True,
    get_info_dict_for_entities=False,
    add_unit=False,
    format_literals=False,
    add_type=False,
) -> Union[str, dict]:
    """Get printable value given value_dict from JSON Wikidata

    Args:
        value_dict (dict): Dictionary : {'type' : str, 'value' : format depends on the 'type'}. This argument can be a string in case the value type is 'string'.
        version (str, Optional): What version of Wikidata to use
        retrieve_entity_name (bool, Optional): Whether to convert entity ID to entity name which requires querying the Mongo Database
        get_info_dict_for_entities (bool, Optional): If True, return a dictionary containing the entitiy information instead of a string. The entity information are retrieved using the function utils_json.get_info_wikidata
        add_unit (bool, Optional) : If True, add unit when converting literals to string
        format_literals (bool, Optional) : (Not implemented yet) If True, do not return raw value for literals but a more refined human digestable format.
        return_type (bool, Optional) : If it's an entity, return "Entity", time ==> "Time", quantity ==> "Quantity", etc. In this case the function returns two elements.

    Returns:
        str or dict : Value
    """
    version = "old" if version == "old" else "new"

    if isinstance(value_dict, str):
        return _util_get_value_return(value_dict, "String", add_type)
    if "id" in value_dict:
        # It's an entity
        if get_info_dict_for_entities:
            return _util_get_value_return(
                get_info_wikidata(value_dict["id"], version=version), "Entity", add_type
            )
        if not retrieve_entity_name:
            return value_dict["id"]
        info = get_info_wikidata(value_dict["id"], version=version)
        if "name" not in info:
            return _util_get_value_return(None, "Entity", add_type)
        return _util_get_value_return(
            "%s (%s)" % (info["name"], value_dict["id"]), "Entity", add_type
        )
    elif "time" in value_dict:
        date = value_dict["time"]
        precision = value_dict["precision"]
        sign = date[0] if date[0] == "-" else ""
        if precision == 11:
            return _util_get_value_return(date.split("T")[0], "Date", add_type)
        elif precision == 10:
            return _util_get_value_return(
                sign + "-".join(date[1:].split("-")[:-1]), "Month", add_type
            )
        else:
            return _util_get_value_return(
                sign + "-".join(date[1:].split("-")[:-2]), "Year", add_type
            )
    elif "text" in value_dict:
        return _util_get_value_return(value_dict["text"], "String", add_type)
    elif "amount" in value_dict:
        ret = value_dict["amount"].lstrip("+")
        if add_unit:
            unit = value_dict["unit"]
            if unit != "1":
                unit_id = unit.split("/")[-1]
                symb = get_unit_symbol(unit_id)
                if symb is not None:
                    ret += symb
        return _util_get_value_return(ret, "Quantity", add_type)
    return _util_get_value_return(value_dict, "Unknown", add_type)


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


point_in_time_prop_id = "P585"


def remove_deprecated(prop_id: str, statements: list, single_valued_properties: dict):
    # Keep the most up-to-date triple in temporal functional relations
    if point_in_time_prop_id in single_valued_properties.get(prop_id, []):
        point_in_times_ = [
            (i, get_point_in_time(statement)) for i, statement in enumerate(statements)
        ]
        point_in_times = [(i, x) for i, x in point_in_times_ if x is not None]
        if len(point_in_times):
            max_idx, _ = max(point_in_times, key=lambda x: x[1])
            utd_statement = statements[max_idx]
            statements.clear()
            statements.append(utd_statement)


def ent_to_vec(ent_id: str, claims: dict, add_singles=False):
    # Transforms a "claims" dictionary from Wikidata 'json' or 'prep' collections to a sparse vector in the form of a list of tuples where each tuple is a couple (rel, obj)
    # Add separeted relations and objects in the vectors
    ent_vec = []
    reverse_info = defaultdict(list)
    for prop_id, statements in claims.items():
        snaks = [
            x["mainsnak"]["datavalue"]
            for x in statements
            if x["mainsnak"]["snaktype"] == "value"
        ]
        # decisions = classify_algorithm_lite([Feature(x) for x in snaks], version)
        # objects = [x['value'] for i,x in enumerate(snaks) if decisions[i] != 'ignore']
        objects = [x["value"] for x in snaks]
        objects = [x["id"] for x in objects if isinstance(x, dict) and ("id" in x)]
        if len(objects) and add_singles:
            ent_vec.append(prop_id)
        for obj in objects:
            ent_vec.append(subrel2str(prop_id, obj))
            reverse_info[obj].extend([ent_id, prop_id, subrel2str(ent_id, prop_id)])
            if add_singles:
                ent_vec.append(obj)
    # Add Entity ID. Scenario : This ID appears in another vector and indicates these two entities are linked
    ent_vec.append(ent_id)
    return ent_vec, reverse_info


def get_all_properties(version="new") -> list:
    name = adapt_name(version)
    d = json.load(
        open(
            osp.join(
                STORAGE_FOLDER,
                "resources/script_stats/process_json_dump_%s.json"
                % name,
            )
        )
    )
    props = list(d["relation_count"].keys())
    return props


def get_single_valued_properties(
    version="new", return_separator=False
) -> Union[list, dict]:
    """Get the list of single valued properties which are properties that can only have one value at a specific point in time, according to the wikidata

    Args:
        version (str, optional): The version of wikipedia to use ('old' or 'new'). Defaults to 'new'.
        return_separator (bool, optional): If True, returns the separator of each single-value property if it exists. In this case, the function returns a dictionary (key=property_id, value=separator (which can be None)). Defaults to False.

    Returns:
        [list, dict]: list of property ids or dictionary (key=property_id, value=separator (which can be None))
    """

    def flatten(container):
        for i in container:
            if isinstance(i, (list, tuple)):
                for j in flatten(i):
                    yield j
            else:
                yield i

    props = get_all_properties(version)
    query = [
        {"$match": {"_id": {"$in": props}}},
        {"$project": {"constraints": "$claims.P2302"}},
        {"$match": {"constraints": {"$exists": 1}}},
        {
            "$project": {
                "constraints": {
                    "$filter": {
                        "input": "$constraints",
                        "cond": {
                            "$in": [
                                "$$c.mainsnak.datavalue.value.id",
                                ["Q52060874", "Q19474404"],
                            ]
                        },
                        "as": "c",
                    }
                }
            }
        },
        {"$match": {"constraints.0": {"$exists": 1}}},
        {
            "$project": {
                "separators": "$constraints.qualifiers.P4155.datavalue.value.id"
            }
        },
    ]
    cursor = db[adapt_name(version)].aggregate(query)
    res = {}
    for x in cursor:
        separators = x.get("separators", [])
        separators = list(flatten(separators))
        res[x["_id"]] = separators
    if not return_separator:
        res = list(res.keys())
    return res


def get_restrictive_qualifiers(version: str):
    # assert version in ['new', 'old']
    cursor = db["qualifier_properties"].aggregate(
        [
            {
                "$lookup": {
                    "from": adapt_name(version),
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "wd",
                }
            },
            {
                "$project": {
                    "instance_of": {
                        "$getField": {"field": "claims", "input": {"$first": "$wd"}}
                    }
                }
            },
            {
                "$project": {
                    "instance_of": "$instance_of.P31.mainsnak.datavalue.value.id"
                }
            },
            {"$match": {"instance_of": {"$exists": 1}}},
            {
                "$match": {
                    "$expr": {
                        "$and": [
                            {"$in": ["Q61719275", "$instance_of"]},
                            {"$not": {"$in": ["Q61719274", "$instance_of"]}},
                        ]
                    }
                }
            },
            {"$project": {"_id": 1}},
        ]
    )
    # point in time (P585) is needed to determine new info
    hard_restrictive = [x["_id"] for x in cursor if x["_id"] != "P585"]

    cursor = db["qualifier_properties"].aggregate(
        [
            {
                "$lookup": {
                    "from": adapt_name(version),
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "wd",
                }
            },
            {
                "$project": {
                    "instance_of": {
                        "$getField": {"field": "claims", "input": {"$first": "$wd"}}
                    }
                }
            },
            {
                "$project": {
                    "instance_of": "$instance_of.P31.mainsnak.datavalue.value.id"
                }
            },
            {"$match": {"instance_of": {"$exists": 1}}},
            {
                "$match": {
                    "$expr": {
                        "$and": [
                            {"$in": ["Q61719275", "$instance_of"]},
                        ]
                    }
                }
            },
            {"$project": {"_id": 1}},
        ]
    )
    soft_restrictive = [
        x["_id"] for x in cursor if x["_id"] not in hard_restrictive + ["P585"]
    ]
    # hard_restrictive are always restrictive
    # soft_restrictive can be restrictive sometimes and not restrictive other times

    cursor = db["qualifier_properties"].aggregate(
        [
            {
                "$lookup": {
                    "from": adapt_name(version),
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "wd",
                }
            },
            {
                "$project": {
                    "instance_of": {
                        "$getField": {"field": "claims", "input": {"$first": "$wd"}}
                    }
                }
            },
            {
                "$project": {
                    "instance_of": "$instance_of.P31.mainsnak.datavalue.value.id"
                }
            },
            {"$match": {"instance_of": {"$exists": 1}}},
            {
                "$match": {
                    "$expr": {
                        "$and": [
                            {"$in": ["Q105388954", "$instance_of"]},
                        ]
                    }
                }
            },
            {"$project": {"_id": 1}},
        ]
    )

    #  Wikidata property to identify online accounts (Q105388954)
    #  review score by (P447)
    others = [x["_id"] for x in cursor if x["_id"]] + ["P447"]
    return {"soft": soft_restrictive, "hard": hard_restrictive, "others": others}


def get_wikidata_json_dump_download_url(
    date: str, option: str, force_build_index=False
) -> tuple[str, str]:
    """Get the download URL for the Full Wikidata JSON dump given a certain date.
    This function begins by crawling Wikidata dump website and Internet Archive to search for JSON dumps.
    Once they are found, an index is created containing a dump download URL for every date in the two websites

    Args:
        date (str): The date for which we want to find the Wikidata dump. Format = YYYYMMDD

        option (str): Can take three values:

            - "after" : Get the dump just after the given date
            - "before" : Get the dump just before the given date
            - "closest" : Get the closest dump to the given date

        force_build_index (bool, Optional): If True, the index build is forced even if an index already exist. Defaults False.

    Returns:
        (str, str): (Download URL, The date of the dump) or None it nothing was found
    """
    if not osp.exists(WIKIDATA_DUMP_INDEX_PATH) or force_build_index:
        exit_status = os.system(
            "python3 %s"
            % osp.join(
                osp.dirname(WIKIDATA_DUMP_INDEX_PATH), "build_wikidata_dumps_index.py"
            )
        )
        if exit_status != 0:
            raise Exception(
                "Something wrong happened with Wikidata Dumps Index creation!"
            )
    with open(WIKIDATA_DUMP_INDEX_PATH) as f:
        wikidata_dump_index = json.load(f)

    if option == "exact":
        url = wikidata_dump_index.get(date)
        if url is not None:
            return url, date
        else:
            return None

    to_datetime64 = lambda d: np.datetime64(d[:4] + "-" + d[4:6] + "-" + d[6:8])
    datetime64_to_date = {to_datetime64(k): k for k in wikidata_dump_index}
    datetimes64 = np.array(list(datetime64_to_date.keys()))
    target_datetime64 = to_datetime64(date)
    zero = target_datetime64 - target_datetime64
    try:
        if option == "closest":
            found_datetime64 = datetimes64[
                np.argmin(np.abs(datetimes64 - target_datetime64))
            ]
        elif option == "after":
            datetimes64 = datetimes64[datetimes64 - target_datetime64 > zero]
            found_datetime64 = datetimes64[np.argmin(datetimes64 - target_datetime64)]
        elif option == "before":
            datetimes64 = datetimes64[datetimes64 - target_datetime64 < zero]
            found_datetime64 = datetimes64[np.argmax(datetimes64 - target_datetime64)]
    except ValueError:
        return None

    date = datetime64_to_date[found_datetime64]
    url = wikidata_dump_index[date]
    return url, date


from wikidata_tools.wikidata import Wikidata, WikidataPrepStage


def adapt_name(version, stage: WikidataPrepStage = WikidataPrepStage.ALMOST_RAW) -> str:
    if version in ("old", "new"):
        name = (
            "wikidata_%s_json" % version
            if stage == WikidataPrepStage.ALMOST_RAW
            else "wikidata_%s_prep" % version
        )
    else:
        name = Wikidata.generate_collection_name(version, stage)
    return name


def get_direct_types(ent_id: str, version: str) -> list[str]:
    name = adapt_name(version)
    types = db[name].find_one(
        {"_id": ent_id}, {"types": "$claims.P31.mainsnak.datavalue.value.id"}
    )["types"]
    return types


def get_subclasses(types: list[str], version: str):
    # Types are entities IDs
    return [
        y
        for x in db[adapt_name(version)].find(
            {"_id": {"$in": types}},
            {"subclasses": "$claims.P279.mainsnak.datavalue.value.id"},
        )
        for y in x.get("subclasses", [])
    ]


def get_indirect_types(types: list[str], version: str, depth: int = None) -> list[str]:
    subclasses = set(types)
    size = 0
    count = 0
    while len(subclasses) > size:
        size = len(subclasses)
        subclasses = subclasses.union(
            x for x in get_subclasses(list(subclasses), version)
        )
        count += 1
        if depth is not None and count >= depth:
            break
    return list(subclasses - set(types))


def get_direct_indirect_types(
    ent_id: str, version: str, depth: int = False
) -> list[str]:
    types = get_direct_types(ent_id, version)
    ind_types = get_indirect_types(types, version, depth)
    return types + ind_types


def human_readable_ku_rdf_deprecated(ku_rdf: dict) -> str:
    """(DEPRECATED) Translate KU RDF stored in MongoDB to a human readable textual version

    Args:
        ku_rdf (dict): KU RDF found in MongoDB

    Returns:
        str : Human readable textual representation
    """
    value = ku_rdf["content"]["mainsnak"]["data"]["value"]
    replacement = ku_rdf["content"]["replacement"]
    # imp = ku_rdf['importance']
    ent_id = ku_rdf["_id"]["ent"]
    prop_id = ku_rdf["_id"]["prop"]
    value = get_value(value)
    replacement_type = replacement["type"]
    ent_name, prop_name = (
        get_info_wikidata(ent_id)["name"],
        get_info_wikidata(prop_id)["name"],
    )
    res = "[%s (%s), %s (%s), %s, %s]" % (
        ent_name,
        ent_id,
        prop_name,
        prop_id,
        value,
        replacement_type,
    )

    if replacement_type == "single_value_replace":
        old_value = replacement["old_value"]
        old_value = get_value(old_value)
        res += " ==> %s" % old_value

    elif replacement_type == "add_to_set_knowledge":
        stats = {
            k: replacement[k]
            for k in ["n_values", "n_new_irl", "n_inv_mark", "n_old_mark"]
        }
        res += " stats=%s" % stats
    return res


def deprecated_human_readable_ku_rdf(ku_rdf: dict, include_to_ignore=True) -> str:
    """Translate KU RDF stored in MongoDB to a human readable textual version

    Args:
        ku_rdf (dict): KU RDF found in MongoDB

    Returns:
        str : Human readable textual representation
    """
    ent_id, prop_id = ku_rdf["_id"]["ent_id"], ku_rdf["_id"]["prop_id"]
    diff = ku_rdf["diff"]
    info = get_info_wikidata([ent_id, prop_id])
    ent_name, prop_name = info[ent_id]["name"], info[prop_id]["name"]
    res = "%s (%s) - %s (%s):\n" % (ent_name, ent_id, prop_name, prop_id)
    for d in diff:
        if not include_to_ignore and d["determine"] == "to_ignore":
            continue
        res += "- (%s-%s) %s   (deprecated=%s, deprecated_old=%s, %s)\n" % (
            d["determine"],
            d["kr_state"],
            get_value(d["mainsnak"]["data"]["value"]),
            d.get("is_deprecated", None),
            d.get("is_deprecated_old", None),
            d["kr_flag"],
        )
    return res


def default_to_frozen(d):
    if isinstance(d, dict):
        d = frozendict({k: default_to_frozen(v) for k, v in d.items()})
    if isinstance(d, list):
        d = tuple(default_to_frozen(x) for x in d)
    return d


def human_readable_ku_rdf(ku_rdf: dict, include_to_ignore=True) -> str:
    """Translate KU RDF stored in MongoDB to a human readable textual version

    Args:
        ku_rdf (dict): KU RDF found in MongoDB

    Returns:
        str : Human readable textual representation
    """
    ent_id, prop_id = ku_rdf["_id"]["ent_id"], ku_rdf["_id"]["prop_id"]
    diff = ku_rdf["snaks"]
    info = get_info_wikidata([ent_id, prop_id])
    ent_name, prop_name = info[ent_id]["name"], info[prop_id]["name"]
    res = "%s (%s) - %s (%s):\n" % (ent_name, ent_id, prop_name, prop_id)
    for d in diff:
        if not include_to_ignore and d["decision"] == "ignore":
            continue
        res += "- (%s) %s   %s\n" % (d["decision"], get_value(d["value"]), d["path"])
    return res


def sim_dict(d1: dict, d2: dict, depth_penalty=True):
    def _sim_dict(d1: dict, d2: dict, depth=1) -> int:
        if not depth_penalty:
            depth = 1
        sim = 0
        if type(d1) is not type(d2):
            return 0
        if isinstance(d1, dict):
            inter = set(d1.keys()).intersection(d2.keys())
            for k in inter:
                sim += _sim_dict(d1[k], d2[k], depth + 1)
        elif isinstance(d1, (list, tuple)):
            C = np.array([[_sim_dict(x, y, depth + 1) for y in d2] for x in d1])
            row, col = linear_sum_assignment(C, maximize=True)
            sim += sum(C[i, j] for i, j in zip(row, col))
        elif isinstance(d1, (bool, int, float, str)):
            sim += int(d1 == d2)
        return sim / depth

    return _sim_dict(d1, d2)


def diff_list_of_dicts(
    list_of_dicts1: list[dict], list_of_dicts2: list[dict], to_ignore: str = None
) -> tuple[list, list, list]:
    """Takes as input two lists (of dictionaries) and output the difference between:

    - New : What is present only in the first list
    - Old : What is present only in the second list
    - Both : What is present in both

    Args:
        list_of_dicts1 (list[dict]): First list of dicts
        list_of_dicts2 (list[dict]): Second list of dicts
        to_ignore (str, optional): Field in  dictionaries to ignore when computing the difference (This field cannot be used to determine if something is old, new or invariant). Defaults to None (no effect).

        For example, to_ignore = 'qualifiers' would ignore the 'qualifiers' field in each dictionary

    Returns:
        tuple[list,list,list]: New, Old, Both
    """

    def default_to_frozen(d):
        if isinstance(d, dict):
            d = frozendict({k: default_to_frozen(v) for k, v in d.items()})
        if isinstance(d, list):
            d = tuple(default_to_frozen(x) for x in d)
        return d

    # Ignore field
    if to_ignore is not None:
        to_ignore_list1 = []
        to_ignore_list2 = []
        for x in list_of_dicts1:
            if to_ignore in x:
                ignore = x.pop(to_ignore)
                to_ignore_list1.append((x, ignore))
            else:
                to_ignore_list1.append((x, ignore))

        for x in list_of_dicts2:
            if to_ignore in x:
                ignore = x.pop(to_ignore)
                to_ignore_list2.append((x, ignore))
            else:
                to_ignore_list2.append((x, ignore))

    # Align similar dictionaries
    groups = defaultdict(list)
    for i, x in enumerate(list_of_dicts1):
        groups[default_to_frozen(x)].append((i, 1))

    for i, x in enumerate(list_of_dicts2):
        groups[default_to_frozen(x)].append((i, 2))

    for _, list_where in groups.items():
        if len(list_where) == 1:
            continue
        if len(set(x[1] for x in list_where)) < 2:
            # Nothing to align
            continue
        if len(list_where) == 2:
            [(idx1, _), (idx2, _)] = sorted(list_where, key=lambda x: x[1])
            list_of_dicts2[idx2] = list_of_dicts1[idx1]
            to_ignore_list2[idx2] = to_ignore_list1[idx1][0], to_ignore_list2[idx2][1]
            continue

        to_ignores1 = []
        to_ignores2 = []
        idxs1 = []
        idxs2 = []
        for idx, where in list_where:
            if where == 1:
                idxs1.append(idx)
                to_ignores1.append(to_ignore_list1[idx][1])
            else:
                idxs2.append(idx)
                to_ignores2.append(to_ignore_list2[idx][1])
        C = np.array(
            [
                [sim_dict(x, y, depth_penalty=True) for y in to_ignores2]
                for x in to_ignores1
            ]
        )
        row, col = linear_sum_assignment(C, maximize=True)
        for r, c in zip(row, col):
            if C[r, c] > 0:
                list_of_dicts2[idxs2[c]] = list_of_dicts1[idxs1[r]]
                to_ignore_list2[idxs2[c]] = (
                    to_ignore_list1[idxs1[r]][0],
                    to_ignore_list2[idxs2[c]][1],
                )

    new, both = [], []
    for d in list_of_dicts1:
        try:
            idx = list_of_dicts2.index(d)
        except ValueError:
            idx = -1
        if idx != -1:
            both.append(d)
            list_of_dicts2.pop(idx)
            for i, (x, ignore) in enumerate(to_ignore_list2):
                if x == d:
                    break
            else:
                x = None
            if x is not None:
                to_ignore_list2[i] = (d, ignore)
            continue
        else:
            new.append(d)
    old = list_of_dicts2

    # Restore to_ignore field
    if to_ignore is not None:
        for x, ignore in to_ignore_list1:
            if len(ignore):
                x[to_ignore + "_new"] = ignore
        for x, ignore in to_ignore_list2:
            if len(ignore):
                x[to_ignore + "_old"] = ignore

    return new, old, both


def get_objects_subrel_pair(subject: str, relation: str, version: str) -> list:
    """Get all objects of the given (subject, relation)-group and the period of validity of each of its objet if it exists

    Args:
        subject (str): Entity ID (e.g., Q123)
        relation (str): Relation ID (e.g., P123)
        version (str): Wikidata version to use. It must be in ('old', 'new')

    Returns:
        list: list of (o, validity_period if it exists else None)
    """
    # _validate_version_arg(version)
    col = db[adapt_name(version)]
    doc = col.find_one({"_id": subject}, {"claims.%s" % relation: 1})
    if doc is None or (claims := doc.get("claims", None)) is None or len(claims) == 0:
        return []
    res = []
    for obj in claims[relation]:
        try:
            value_dict = obj["mainsnak"]["datavalue"]["value"]
        except KeyError:
            continue
        f = Feature(obj)
        value = get_value(value_dict, version=version, retrieve_entity_name=False)
        if f.point_in_time is not None:
            res.append((value, f.point_in_time))
        elif f.start_time is not None or f.end_time is not None:
            res.append((value, [f.start_time, f.end_time]))
        else:
            res.append((value, None))
    return res


def wikidata_iterator(version: str, entities_to_ignore: set[str], wikipedia_only=True):
    """Returns an iterator over entities in Wikidata. Each element of the iterator is a couple (entity ID, claims). "claims" is a dictionary that contains the triples of the form (entity ID, *, *).

    Args:
        version (str): The version of Wikidata used. One of the values : "old" or "new"
        seen_entities (set[str]): Entities to ignore
        wikipedia_only (bool, optional): Restrain the iterator to Wikipedia entities (entities that possess a dedicated Wikipedia article). Defaults to True.
    """

    @run_in_process
    def generator_wikipedia(version: str, entities_to_ignore: set[str]):
        total = db["wikidata_%s_prep" % version].estimated_document_count()
        entities_to_ignore = set(entities_to_ignore)

        for x in tqdm.tqdm(
            db[adapt_name(version, WikidataPrepStage.PREPROCESSED)].aggregate(
                [
                    {"$project": {"_id": 1}},
                    {
                        "$lookup": {
                            "from": adapt_name(version),
                            "localField": "_id",
                            "foreignField": "_id",
                            "as": "wd",
                        }
                    },
                    {"$replaceRoot": {"newRoot": {"$first": "$wd"}}},
                    {"$project": {"claims": 1}},
                    # {"$limit" : 10000}
                ]
            ),
            total=total,
            mininterval=5,
            file=sys.stdout,
        ):
            if x["_id"] in entities_to_ignore:
                continue
            yield x["_id"], x["claims"]

    @run_in_process
    def generator_full(version: str, entities_to_ignore: set[str]):
        total = db[adapt_name(version)].estimated_document_count()
        entities_to_ignore = set(entities_to_ignore)

        for x in tqdm.tqdm(
            db[adapt_name(version)].find({}, {"claims": 1}),
            total=total,
            mininterval=5,
            file=sys.stdout,
        ):
            if x["_id"] in entities_to_ignore:
                continue
            yield x["_id"], x["claims"]

    args = version, entities_to_ignore
    if wikipedia_only:
        return generator_wikipedia(*args)
    else:
        return generator_full(*args)


if __name__ == "__main__":
    print(adapt_name("old"))
    print(adapt_name("new"))
    print(adapt_name("old", WikidataPrepStage.PREPROCESSED))
    print(adapt_name("new", WikidataPrepStage.PREPROCESSED))
    print(adapt_name("20200101", WikidataPrepStage.PREPROCESSED))
    print(adapt_name("20200101", WikidataPrepStage.ALMOST_RAW))

    # Expected output:
    # wikidata_old_json
    # wikidata_new_json
    # wikidata_old_prep
    # wikidata_new_prep
    # wikidata__20200101__PREPROCESSED
    # wikidata__20200101__ALMOST_RAW
