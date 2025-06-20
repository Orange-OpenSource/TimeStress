# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

import numpy as np
from datetime import datetime
from wikidata_tools.build_wd.config import STORAGE_FOLDER
from wikidata_tools.build_wd.utils.ku import FeatureDiff, classify_algorithm_full
from wikidata_tools.build_wd.utils.wd import db, get_single_valued_properties
from collections import defaultdict
import json
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--diff_coll_name",
    help="The Difference Wikidata collection name.",
    required=True,
)
parser.add_argument(
    "--wfd_coll_name",
    help="The WikiFactDiff collection name.",
    required=True,
)
parser.add_argument(
    "--time_old", help="Timestamp of the Old Wikidata (Format : YYYYMMDD)", default=None
)
parser.add_argument(
    "--time_new", help="Timestamp of the New Wikidata (Format : YYYYMMDD)", default=None
)
args = parser.parse_args()

old_wikidata_timestamp = np.datetime64(
    datetime.strptime(args.time_old, "%Y%m%d"), "D"
)
new_wikidata_timestamp = np.datetime64(
    datetime.strptime(args.time_new, "%Y%m%d"), "D"
)


class GlobalStat:
    validity_indicators_count = defaultdict(lambda: 0)
    group_determined_count = defaultdict(lambda: 0)
    decision_count = defaultdict(lambda: 0)
    path_count = defaultdict(lambda: 0)
    triple_decision_count = defaultdict(lambda: 0)
    n_triple_ph_new_entities = 0
    n_triple = 0
    both_triples_count = 0
    both_group_count = 0
    n_entities = 0

    @staticmethod
    def save(path: str):
        d = {
            k: (dict(v) if isinstance(v, defaultdict) else v)
            for k, v in vars(GlobalStat).items()
            if isinstance(v, (int, defaultdict))
        }
        d = {
            k: (
                {
                    (
                        " → ".join([str(x) for x in k2])
                        if isinstance(k2, tuple)
                        else k2
                    ): v2
                    for k2, v2 in v.items()
                }
                if isinstance(v, dict)
                else v
            )
            for k, v in d.items()
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=4, ensure_ascii=False)


coll = db[args.diff_coll_name]
ph_new_entities = set(
    x["_id"]
    for x in db[
        "physically_new_entities__%s__%s" % (args.time_old, args.time_new)
    ].find()
)
single_valued_properties = get_single_valued_properties(return_separator=True)
temporal_single_value_properties = [
    k for k, v in single_valued_properties.items() if "P585" in v
]

authorized_decisions = ["keep", "learn", "forget", "ignore"]
decisive_decisions = ["keep", "learn", "forget"]


def classify(snaks: list[dict]):
    contains_decisive = False
    contains_authorized_only = True
    for snak in snaks:
        decision = snak["decision"]
        GlobalStat.triple_decision_count[decision] += 1
        contains_decisive |= decision in decisive_decisions
        contains_authorized_only &= decision in authorized_decisions
    if all(x["decision"] in ["keep", "ignore"] for x in snaks):
        return "all_keep_or_ignore"
    elif contains_authorized_only:
        return "fully_determined"
    elif contains_decisive:
        return "partially_determined"
    return "fully_undetermined"


def is_value_ph_new(value: dict):
    if isinstance(value, dict) and "id" in value:
        return value["id"] in ph_new_entities
    return False


# ent_importance_coll = db['new_entities_importance']

partially_determined_coll = db["wkd_pd__%s__%s" % (args.time_old, args.time_new)]
fully_determined_coll = db["wikifactdiff__%s__%s" % (args.time_old, args.time_new)]

partially_determined_coll.drop()
fully_determined_coll.drop()

# partially_determined_coll.create_index([('ent_imp', -1)])
# fully_determined_coll.create_index([('ent_imp', -1)])

partially_determined_coll_to_push = []
fully_determined_coll_to_push = []

n_documents = coll.estimated_document_count()

# cursor = db['new_entities_importance'].aggregate([
#     {'$sort' : {'ent_imp' : -1}},
#     {'$lookup' : {
#         'from' : args.diff_coll_name,
#         'localField' : "_id",
#         'foreignField' : "_id",
#         'as' : 'wd'
#     }},
#     {'$match' : {'$expr':  {'$gt' : [{'$size' : '$wd'}, 0]}}},
#     {'$project' : {'wd' : {'$first' : '$wd'}, 'ent_imp' : 1}},
#     {'$addFields' : {'wd.ent_imp':  '$ent_imp'}},
#     {'$replaceRoot' : {'newRoot' : '$wd'}}
# ])
cursor = coll.find()

for i, ent_dict in enumerate(cursor):
    ent_id = ent_dict["_id"]
    ent_importance = ent_dict.get("ent_imp", None)
    is_old = all(x["diff_flag"] == "old" for y in ent_dict["diff"].values() for x in y)
    for prop_id, snaks in ent_dict["diff"].items():
        GlobalStat.n_triple += len(snaks)
        if all(snak["diff_flag"] == "both" for snak in snaks):
            # Skip subject-relation groups where nothing changed
            GlobalStat.both_triples_count += len(snaks)
            GlobalStat.both_group_count += 1
            continue
        features = [
            FeatureDiff(
                snak,
                ent_id=ent_id,
                prop_id=prop_id,
                is_old=is_old,
                obj_is_ph_new=is_value_ph_new(snak["value"]),
                ent_is_ph_new=(ent_id in ph_new_entities),
            )
            for snak in snaks
        ]
        decisions, paths = classify_algorithm_full(features, old_wikidata_timestamp, new_wikidata_timestamp)
        for snak, decision, path in zip(snaks, decisions, paths):
            snak["decision"] = decision
            snak["path"] = path
            GlobalStat.decision_count[decision] += 1
            GlobalStat.path_count[path] += 1

        c = classify(snaks)
        # Filter decision == 'ignore'
        snaks = [s for s in snaks if s["decision"] != "ignore"]
        GlobalStat.group_determined_count[c] += 1
        to_push = {
            "_id": {"ent_id": ent_id, "prop_id": prop_id},
            "snaks": snaks,
            "ent_imp": ent_importance,
        }
        if c == "fully_determined":
            fully_determined_coll_to_push.append(to_push)
        elif c == "partially_determined":
            partially_determined_coll_to_push.append(to_push)
    if i % 1000 == 0:
        if len(fully_determined_coll_to_push):
            fully_determined_coll.insert_many(
                fully_determined_coll_to_push, ordered=False
            )
            fully_determined_coll_to_push.clear()
        if len(partially_determined_coll_to_push):
            partially_determined_coll.insert_many(
                partially_determined_coll_to_push, ordered=False
            )
            partially_determined_coll_to_push.clear()
        print("%s/%s (%s%%)" % (i, n_documents, i / n_documents * 100))
        GlobalStat.save(
            osp.join(
                STORAGE_FOLDER,
                "resources/script_stats/create_wkd__%s__%s.json"
                % (args.time_old, args.time_new),
            )
        )

GlobalStat.save(
    osp.join(STORAGE_FOLDER, "resources/script_stats/create_wkd__%s__%s.json")
    % (args.time_old, args.time_new)
)
