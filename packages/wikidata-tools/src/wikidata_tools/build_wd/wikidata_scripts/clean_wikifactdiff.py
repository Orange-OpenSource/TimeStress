# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

#### WHY CLEAN? look at this example :
# Jack Dorsey, position held:
# chief executive officer (('forget', 'keep'))
####
# NOTE-RELEASE : adapt n1, n2, ... as a set
# NOTE-RELEASE : Rename old --> obsolete, learn --> new, keep --> stable

from collections import Counter
import random
from wikidata_tools.build_wd.config import STORAGE_FOLDER
import os.path as osp
import json
import numpy as np

random.seed(456)


if __name__ == "__main__":
    wfd = [
        json.loads(x)
        for x in open(osp.join(STORAGE_FOLDER, "wikifactdiff_dirty.jsonl"))
    ]
    n1, n2, n3 = set(), set(), set()
    n_drop = 0
    for x in wfd:
        count_objects = Counter([y["label"] for y in x["objects"]])

        if len(count_objects) == len(x["objects"]):
            continue
        new_objects = []
        for k, v in count_objects.items():
            if v == 1:
                new_objects.extend([y for y in x["objects"] if y["label"] == k])
                continue
            labels, decisions = list(
                zip(
                    *[
                        (y["label"], y["decision"])
                        for y in x["objects"]
                        if y["label"] == k
                    ]
                )
            )
            count_decisions = Counter(decisions)
            n_learn = count_decisions.get("learn", 0)
            n_forget = count_decisions.get("forget", 0)
            n_keep = count_decisions.get("keep", 0)
            if n_learn == 0 and n_forget > 0 and n_keep == 0:
                new_objects.append([y for y in x["objects"] if y["label"] == k][0])
                n1.add((x["subject"]["id"], x["relation"]["id"]))
            elif n_learn > 0 and n_forget == 0 and n_keep >= 0:
                # We retain the first "keep" by default (since labels are ordered increasingly)
                new_objects.append([y for y in x["objects"] if y["label"] == k][0])
                n2.add((x["subject"]["id"], x["relation"]["id"]))
            elif n_learn == 0 and n_forget == 0 and n_keep > 0:
                new_objects.append([y for y in x["objects"] if y["label"] == k][0])
                n3.add((x["subject"]["id"], x["relation"]["id"]))
            else:
                new_objects.clear()
                n_drop += 1
                break
        x["objects"] = new_objects
    wfd = [x for x in wfd if len(x["objects"])]

    # wfd_repl = [x for x in wfd if x['is_replace'] and (random.random() < 1/14 or x['relation']['label'] != 'population')]
    # with open(osp.join(STORAGE_FOLDER, 'wikifactdiff_replacement.jsonl'), 'w') as f:
    #     for x in wfd_repl:
    #         f.write(json.dumps(x) + '\n')

    # Remove potenial redundancies
    for i, x in enumerate(wfd):
        x["case_id"] = i
        _, idx = np.unique([y["label"] for y in x["objects"]], return_index=True)
        x["objects"] = [x["objects"][i] for i in idx]
        for k in ("neighborhood", "closeness"):
            triples = x[k]
            for t in triples:
                _, idx = np.unique(
                    [y["object"]["label"] for y in t["objects"]], return_index=True
                )
                t["objects"] = [t["objects"][i] for i in idx]

    with open(osp.join(STORAGE_FOLDER, "wikifactdiff.jsonl"), "w") as f:
        for x in wfd:
            f.write(json.dumps(x) + "\n")

    print("Filter process statistics:")
    print("Only forgets : %s" % len(n1))
    print("Only learns and keeps : %s" % len(n2))
    print("Only keeps : %s" % len(n3))
    print("Number of deleted groups : %s" % n_drop)
