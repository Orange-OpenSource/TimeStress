# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

import argparse
import sys

import tqdm
from _old_codebase.config import STORAGE_FOLDER
from _old_codebase.utils.wd import db
import os.path as osp
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "--version",
    choices=["old", "new"],
    help="The version of Wikidata from which the script will collect aliases",
    required=True,
)

if __name__ == "__main__":
    args = parser.parse_args()
    coll = db["wikidata_%s_prep" % args.version]
    to_save = []
    f = open(osp.join(STORAGE_FOLDER, "%s_wikidata_aliases.jsonl" % args.version), "w")
    for i, d in tqdm.tqdm(
        enumerate(coll.find({}, {"aliases": 1, "label": 1}), start=1),
        total=coll.estimated_document_count(),
        mininterval=5,
        file=sys.stdout,
    ):
        id = d.get("_id", None)
        label = d.get("label", None)
        aliases = d.get("aliases", None)
        if label is None or id is None or aliases is None or isinstance(aliases, dict):
            continue
        label_lower, aliases_lower = label.lower(), [alias.lower() for alias in aliases]
        aliases = [
            alias
            for alias, alias_lower in zip(aliases, aliases_lower)
            if label_lower not in alias_lower
            and alias_lower not in label_lower
            and len(alias_lower) > 4
        ]
        if len(aliases) == 0:
            continue

        to_save.append([(id, label), aliases])
        if i % 10000 == 0:
            s = "\n".join(json.dumps(x) for x in to_save) + "\n"
            to_save.clear()
            f.write(s)
