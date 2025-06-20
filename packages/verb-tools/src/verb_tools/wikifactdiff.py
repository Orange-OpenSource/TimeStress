# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

# from typing import Iterable
from collections import Counter, defaultdict
import os
import re

from wikidata_tools.build_wd.verbalize_wikifactdiff.utils import blank_out
from ke_utils.glob_core import MongoableEnum
from ke_utils.globals import STORAGE_FOLDER
from wikidata_tools.core import Relation
from ke_utils.general import dump_json, load_json
from .core import Relation2TemplatesVerbalizer, Template
from datasets import load_dataset
import os.path as osp

class WFDVerbVersion(MongoableEnum):
    V1 = 1
    V2 = 2

def _get_fill_in_the_blank(verb, test_label):
    # Replaces the test_label in the verbalization with "____". If it does not exist return None
    fill_in_the_blank = None
    v = verb['verbalization']
    matches = list(re.finditer(rf'{re.escape(test_label)}[\.]?', v, re.IGNORECASE))
    if len(matches) != 1:
        return None
    search_label = matches[0]
    # ends_with_label = search_label is not None and search_label.end() == len(v)
    if search_label is not None:
        fill_in_the_blank = v.replace(search_label.group(0), '____')
    # if ends_with_label:
    #     fill_in_the_blank = v.replace(search_label.group(0), '____')
    # else:
    #     search_label = re.search(rf'{re.escape(test_label)}', v, re.IGNORECASE)
    #     label_found = search_label is not None
    #     if label_found:
    #         fill_in_the_blank = re.sub(rf'{re.escape(test_label)}', r'____', v, flags=re.IGNORECASE)
        
    return fill_in_the_blank

def clean_verbs(verbs : list[str]) -> list:
    res = []
    for verb in verbs:
        if re.search(r'[A-Za-z0-9\+\-]____', verb):
            continue
        verb = re.sub(r"(?<=[^ \(\[\{\$:])____", " ____", verb)
        res.append(verb)
    return res

BANNED_TEMPLATES = [
    "[SUB] is a [OBJ] being",
    "[SUB] is a member of the [OBJ] species",
    "[SUB] belongs to the class of [OBJ]",
]

class WikiFactDiffVerbalizer(Relation2TemplatesVerbalizer):
    def __init__(self, version : WFDVerbVersion = WFDVerbVersion.V2, use_cache=True) -> None:
        self.version = version
        if use_cache and osp.exists(self.cache_path):
            best_merge_templates = load_json(self.cache_path)
        else:
            hf_info = {
                'path' : 'Orange/WikiFactDiff',
                'name' : 'triple_verbs'
            }
            if version == WFDVerbVersion.V2:
                hf_info['name'] += '_V2'
            print('Switching to Huggingface version located in %s' % hf_info['name'])
            to_iterate = load_dataset(**hf_info)['train'].to_list()
            
            prop_fitb = defaultdict(list)
            for x in to_iterate:
                if x['error'] is not None:
                    continue
                prop = x['triple']['relation']['id']
                subject_label = x['triple']['subject']['label']
                object_label = x['triple']['object']['label']
                verbs = [_get_fill_in_the_blank(y, object_label) for y in x['verbalizations']]

                verbs = [blank_out(y, subject_label) for y in verbs if y is not None]
                verbs = [y for y in verbs if y is not None]

                verbs = clean_verbs(verbs)

                if len(verbs):
                    prop_fitb[prop].extend(verbs)
            best_merge_templates = {prop_id: Counter(verbs) for prop_id, verbs in prop_fitb.items()}
            os.makedirs(osp.dirname(self.cache_path), exist_ok=True)
            dump_json(self.cache_path, best_merge_templates)
        
        templates_scores = defaultdict(dict)
        for prop_id, counter in best_merge_templates.items():
            rel = Relation(prop_id)
            for temp, count in counter.items():
                temp : str
                # try:
                #     sub_pos = temp.index('XXXX')
                #     obj_pos = temp.index('____')
                # except ValueError:
                #     continue
                # if sub_pos > obj_pos:
                #     sub_pos -= 4
                # else:
                #     obj_pos -= 4
                temp = temp.replace('XXXX', '[SUB]').replace('____', '[OBJ]')
                if temp in BANNED_TEMPLATES:
                    continue
                temp = Template(temp, rel)
                templates_scores[rel][temp] = count

            
        
        super().__init__(templates_scores)

    @property
    def cache_path(self) -> str:
        return osp.join(STORAGE_FOLDER, "cache_template_verbalizers", "%s_%s.json" % (self.__class__.__name__, self.version.name))