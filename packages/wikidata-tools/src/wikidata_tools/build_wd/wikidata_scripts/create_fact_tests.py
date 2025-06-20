# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

# This script was designed to test the factual knowledge of LLMs using cloze tests, correct answers and 100 neighbors to the correct answers.

TEMPLATES = {
    "P35": {
        "cloze": "The head of state of XXXX is ____",
        "reverse_cloze": "____ is the head of state of XXXX",
        "question": "Question: Who is the head of state of XXXX? Answer: ____",
        "question_yesno": "Question: Is ____ the head of state of XXXX? Answer: Yes",
        "python_style": "XXXX.head_of_state = '____'",
    },
    "P6": {
        "cloze": "The head of government of XXXX is ____",
        "reverse_cloze": "____ is the head of government of XXXX",
        "question": "Question: Who is the head of government of XXXX? Answer: ____",
        "question_yesno": "Question: Is ____ the head of government of XXXX? Answer: Yes",
        "python_style": "XXXX.head_of_government = '____'",
    },
    "P108": {
        "cloze": "XXXX works at ____",
        "reverse_cloze": "____ is one of XXXX's employees",
        "question": "Question: Who is the employer of XXXX? Answer: ____",
        "question_yesno": "Question: Does XXXX work at ____? Answer: Yes",
        "python_style": "XXXX.employer = '____'",
    },
    "P169": {
        "cloze": "XXXX is the CEO of ____",
        "reverse_cloze": "____ has XXXX as its chief executive officer",
        "question": "Question: Who is the chief executive officer of ____? Answer: XXXX",
        "question_yesno": "Question: Is XXXX the chief executive officer of ____? Answer: Yes",
        "python_style": "____.ceo = 'XXXX'",
    },
    "P26": {
        "cloze": "XXXX is married to ____",
        "reverse_cloze": "____ is the spouse of XXXX",
        "question": "Question: Who is the spouse of XXXX? Answer: ____",
        "question_yesno": "Question: Is ____ the spouse of XXXX? Answer: Yes",
        "python_style": "XXXX.spouse = '____'",
    },
    "P286": {
        "cloze": "XXXX is the head coach of ____",
        "reverse_cloze": "____ is coached by XXXX",
        "question": "Question: Who is the head coach of ____? Answer: XXXX",
        "question_yesno": "Question: Is XXXX the head coach of ____? Answer: Yes",
        "python_style": "____.head_coach = 'XXXX'",
    },
    "P38": {
        "cloze": "XXXX uses ____ as its currency",
        "reverse_cloze": "____ is the currency of XXXX",
        "question": "Question: What is the currency used by XXXX? Answer: ____",
        "question_yesno": "Question: Does XXXX use ____ as its currency? Answer: Yes",
        "python_style": "XXXX.currency = '____'",
    },
    "P451": {
        "cloze": "XXXX is in a relationship with ____",
        "reverse_cloze": "____ is the unmarried partner of XXXX",
        "question": "Question: Who is the unmarried partner of XXXX? Answer: ____",
        "question_yesno": "Question: Is ____ the unmarried partner of XXXX? Answer: Yes",
        "python_style": "XXXX.unmarried_partner = '____'",
    },
    "P488": {
        "cloze": "XXXX is the chairperson of ____",
        "reverse_cloze": "____ is chaired by XXXX",
        "question": "Question: Who is the chairperson of ____? Answer: XXXX",
        "question_yesno": "Question: Is XXXX the chairperson of ____? Answer: Yes",
        "python_style": "____.chairperson = 'XXXX'",
    },
    "P54": {
        "cloze": "XXXX is a member of ____ sports team",
        "reverse_cloze": "____ has XXXX as a team member",
        "question": "Question: Which sports team is XXXX a member of? Answer: ____",
        "question_yesno": "Question: Is XXXX a member of the ____ sports team? Answer: Yes",
        "python_style": "XXXX.sports_team = '____'",
    },
}


from tqdm import tqdm
from wikidata_tools.build_wd.utils.wd import get_info_wikidata, get_objects_subrel_pair
from wikidata_tools.build_wd.utils.ku import new_wikidata_timestamp, old_wikidata_timestamp
from wikidata_tools.build_wd.utils.general import subrel2str, uniquifier
import numpy as np
import os.path as osp
import json
from wikidata_tools.build_wd.config import STORAGE_FOLDER
import re


class Continue(Exception):
    pass


def infuse_valid_objects(knn, valid_objects, n_neighbors: int):
    new = knn.time == "new"
    rev_feat = knn.add_reverse_features
    for i, obj in enumerate(valid_objects):
        vect = knn.get_vectors(obj["id"])
        if vect is None:
            raise Continue
        alt, _ = knn.get_nearest(vect, k=n_neighbors + len(valid_objects) - 1)
        alt_objects = get_info_wikidata(alt, version="new" if new else "old")
        alt_objects = [alt_objects[y] for y in alt]
        for a in alt_objects:
            a["label"] = a.pop("name")
        alt_objects = [x for x in alt_objects if len(x["label"])]
        valid_objects[i][
            ("alt_new" if new else "alt") + ("_revfeat" if rev_feat else "")
        ] = alt_objects


def get_temporally_adjacent_objects(subject: str, relation: str, new: bool) -> list:
    version = "old" if not new else "new"
    adj_objects = get_objects_subrel_pair(subject, relation, version)
    non_period_objects = [x[0] for x in adj_objects if x[1] is None]
    adj_objects = [x for x in adj_objects if x[1] is not None]
    if len(adj_objects) == 0:
        return []
    objects, timestamps = list(zip(*adj_objects))
    timestamps = list(timestamps)
    for i in range(len(timestamps)):
        ts = timestamps[i]
        if isinstance(ts, (tuple, list)):
            for j in range(2):
                if ts[j] is None:
                    ts = ts[(j + 1) % 2], ts[(j + 1) % 2]
                    break
            timestamps[i] = ts

    timestamps = np.array(
        [
            x if not isinstance(x, (list, tuple)) else (x[0] + (x[1] - x[0]) / 2)
            for x in timestamps
        ]
    )
    reference_time = new_wikidata_timestamp if new else old_wikidata_timestamp
    ordered_objects = np.array(objects)[np.argsort(np.abs(timestamps - reference_time))]
    ordered_objects = uniquifier(ordered_objects)
    ordered_objects += non_period_objects
    info = get_info_wikidata(ordered_objects, version=version)
    ordered_objects = [info[y] for y in ordered_objects]
    for a in ordered_objects:
        a["label"] = a.pop("name")
    return ordered_objects


def infuse_eps_enhanced_neighbor_objects(
    knn, subject: str, relation: str, valid_objects, eps: float, n_neighbors: int
):
    new = knn.time == "new"
    rev_feat = knn.add_reverse_features
    for i, obj in enumerate(valid_objects):
        vect = knn.get_vectors(obj["id"])
        if vect is None:
            raise Continue
        # We enhance the TF-IDF values of s,r, (s,r) and o by +eps.
        to_enhance = [subject, relation, obj["id"], subrel2str(subject, relation)]
        if eps != 0:
            for enh in to_enhance:
                idx = knn.vocab.get(enh, None)
                if idx is not None:
                    vect[0, idx] += eps
        alt, _ = knn.get_nearest(vect, k=n_neighbors + len(valid_objects) - 1)
        alt_objects = get_info_wikidata(alt, version="new" if new else "old")
        alt_objects = [alt_objects[y] for y in alt]
        for a in alt_objects:
            a["label"] = a.pop("name")
        alt_objects = [x for x in alt_objects if len(x["label"])]
        valid_objects[i][
            f"eps={eps}-enhanced_neigh"
            + ("_new" if new else "")
            + ("_revfeat" if rev_feat else "")
        ] = alt_objects


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
        "--use_custom_templates",
        action="store_true",
        help="Use the diverse templates including questions, yes/no questions, and python-style questions",
    )
    parser.add_argument(
        "--n_facts", type=int, help="Number of tested facts. Default 1000", default=1000
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        help="Number of neighbors for each fact. Defaults to 100",
        default=100,
    )
    parser.add_argument(
        "--neigh_method",
        type=str,
        help='Choose only one neighbor method. \
                        The format of this argument follows the following expression: \
                        "(temp+)ε=XX(+revfeat)".\nCorrect examples include temp+ε=0.2, ε=1000+revfeat, ε=0, temp+ε=0 (the value we suggest using). \
                        Defaults to None (add all neighbors method)',
        default=None,
    )
    # parser.add_argument('--force_reset', action='store_true', help="Reset the process from the beginning (lose all precedent progress!)")
    # parser.add_argument('option', type=str, help='What type of dataset you want to create? (For which experiment) Can take three values : "ku","nn_eval"', required=True)
    args = parser.parse_args()

    pat = r"(temp\+)?ε=([+-]?([0-9]*[.])?[0-9]+)(\+revfeat)?"

    # count_entities = 0
    # ph_new_entities = set(x['_id'] for x in db['physically_new_entities'].find())
    # functional_relations = set(get_single_valued_properties())
    if args.ann_method == "dense":
        from _old_codebase.find_neighbors.knn_dense import KNearestNeighbors
    else:
        from _old_codebase.find_neighbors.knn_sparse_nmslib import KNearestNeighbors

    if args.neigh_method is not None:
        load_all_knns = False
        options = re.findall(pat, args.neigh_method)[0]
        temp, chosen_eps, revfeat = options[0], float(options[1]), options[-1]
    else:
        load_all_knns = True

    if not revfeat or load_all_knns:
        knn_without_rev = KNearestNeighbors(
            time="old", load_vocab=True, add_reverse_features=False
        )
        knn_new_without_rev = KNearestNeighbors(
            time="new", load_vocab=True, add_reverse_features=False
        )
        knn_without_rev.load_index()
        knn_new_without_rev.load_index()
    if revfeat or load_all_knns:
        knn = KNearestNeighbors(time="old", load_vocab=True, add_reverse_features=True)
        knn_new = KNearestNeighbors(
            time="new", load_vocab=True, add_reverse_features=True
        )
        knn_new.load_index()
        knn.load_index()

    wfd = [json.loads(x) for x in open(osp.join(STORAGE_FOLDER, "wikifactdiff.jsonl"))]

    l = []
    pbar = tqdm(total=args.n_facts)
    for x in wfd:
        subject = x["subject"]["id"]
        relation = x["relation"]["id"]
        try:
            valid_objects = [y for y in x["objects"]]
            if len(valid_objects) == 0 or any(
                y["description"] in ("String", "Date", "Year", "Month", "Quantity")
                for y in valid_objects
            ):
                continue
            rel_id = x["relation"]["id"]
            if args.use_custom_templates and rel_id not in TEMPLATES:
                continue
            if args.neigh_method is None:
                # Add all neighboring method
                infuse_valid_objects(knn, valid_objects, args.n_neighbors)
                infuse_valid_objects(knn_new, valid_objects, args.n_neighbors)
                infuse_valid_objects(knn_without_rev, valid_objects, args.n_neighbors)
                infuse_valid_objects(
                    knn_new_without_rev, valid_objects, args.n_neighbors
                )
                x["temp_adj_objects"] = get_temporally_adjacent_objects(
                    subject, relation, new=False
                )
                x["temp_adj_objects_new"] = get_temporally_adjacent_objects(
                    subject, relation, new=True
                )
                for eps in (0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000):
                    infuse_eps_enhanced_neighbor_objects(
                        knn, subject, relation, valid_objects, eps, args.n_neighbors
                    )
                    infuse_eps_enhanced_neighbor_objects(
                        knn_new, subject, relation, valid_objects, eps, args.n_neighbors
                    )
                    infuse_eps_enhanced_neighbor_objects(
                        knn_without_rev,
                        subject,
                        relation,
                        valid_objects,
                        eps,
                        args.n_neighbors,
                    )
                    infuse_eps_enhanced_neighbor_objects(
                        knn_new_without_rev,
                        subject,
                        relation,
                        valid_objects,
                        eps,
                        args.n_neighbors,
                    )
            else:
                neigh_sys, neigh_sys_new = (
                    (knn, knn_new)
                    if revfeat
                    else (knn_without_rev, knn_new_without_rev)
                )
                infuse_eps_enhanced_neighbor_objects(
                    neigh_sys,
                    subject,
                    relation,
                    valid_objects,
                    chosen_eps,
                    args.n_neighbors,
                )
                infuse_eps_enhanced_neighbor_objects(
                    neigh_sys_new,
                    subject,
                    relation,
                    valid_objects,
                    chosen_eps,
                    args.n_neighbors,
                )
                if temp:
                    x["temp_adj_objects"] = get_temporally_adjacent_objects(
                        subject, relation, new=False
                    )
                    x["temp_adj_objects_new"] = get_temporally_adjacent_objects(
                        subject, relation, new=True
                    )
                if chosen_eps == 0:
                    infuse_valid_objects(neigh_sys, valid_objects, args.n_neighbors)
                    infuse_valid_objects(neigh_sys_new, valid_objects, args.n_neighbors)

            if args.use_custom_templates:
                x["update_prompt"] = TEMPLATES[rel_id]

            x["objects"] = valid_objects
            l.append(x)
            pbar.update()
        except Continue:
            pass
        if len(l) == args.n_facts:
            break

    custom_suffix = "_custom" if args.use_custom_templates else ""
    neigh_method_suffix = (
        ("_" + args.neigh_method) if args.neigh_method is not None else ""
    )
    filename = "neighbor_factual_tests%s.json" % (custom_suffix + neigh_method_suffix)
    with open(osp.join(STORAGE_FOLDER, filename), "w") as f:
        json.dump(l, f)
