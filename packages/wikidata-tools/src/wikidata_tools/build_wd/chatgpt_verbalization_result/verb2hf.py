# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

# Convert verbalizations to parquet and push them to huggingface

import json
import shutil
from huggingface_hub import HfApi
import os.path as osp
import pandas as pd
import os


HF_VERB_PATH = {
    "path": "OrangeInnov/WikiFactDiff",  # Huggingface ID
    "name": "triple_verbs_V2",  # Config name
}

path = osp.dirname(__file__)
import lzma

old_wiki_verb = [
    json.loads(x)
    for x in lzma.open(
        osp.join(path, "verbalizations_old_wikidata.jsonl.xz"),
        mode="rt",
        encoding="utf-8",
    )
]
wfd_verb = [
    json.loads(x)
    for x in lzma.open(
        osp.join(path, "verbalizations_wfd.jsonl.xz"), mode="rt", encoding="utf-8"
    )
]

old_wiki_verb = pd.DataFrame(old_wiki_verb)
wfd_verb = pd.DataFrame(old_wiki_verb)

old_wiki_verb["triple_origin"] = "wiki_20210104"
wfd_verb["triple_origin"] = "wfd_20210104-20230227"

all_verbs = pd.concat([old_wiki_verb, wfd_verb], axis=0, ignore_index=True)
path_verb_folder = "/tmp/%s" % HF_VERB_PATH["name"]
shutil.rmtree(path_verb_folder, ignore_errors=True)
os.makedirs(path_verb_folder)
all_verbs.to_parquet(osp.join(path_verb_folder, "train.parquet"))

api = HfApi()
api.upload_folder(
    repo_id=HF_VERB_PATH["path"],
    folder_path=path_verb_folder,
    repo_type="dataset",
    path_in_repo=HF_VERB_PATH["name"],
)
print("Verbalizations uploaded!! ")
print("IMPORTANT : Don't forget update the README file in the huggingface repo")
