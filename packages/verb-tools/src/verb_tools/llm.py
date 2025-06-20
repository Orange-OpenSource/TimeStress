# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

import dataclasses
import functools
import os
import re
import threading
from ke_utils.globals import STORAGE_FOLDER
from verb_tools.options import Tense
from wikidata_tools.core import TimedTriple, Triple
from wikidata_tools.wikidata import TempWikidata, WikidataPrepStage
from .core import TemplateVerbalizer, Verbalizer, Template
from diskcache import FanoutCache, JSONDisk
import os.path as osp

diskcache_path = os.environ.get('DISKCACHE_JSON_PATH')
if not diskcache_path:
    diskcache_path = osp.join(STORAGE_FOLDER, 'diskcache_json')
else:
    print("WARNING: Diskcache JSON path=%s" % diskcache_path)
cache = FanoutCache(osp.join(diskcache_path, 'llmu_verb'), timeout=100000, disk=JSONDisk) # Calling GPT-4o is too expensive so no timeout

SYSTEM_PROMPT_PAST = """You are an advanced knowledge verbalization system.
You take as input a knowledge quadruple (subject, relation, object, time) and generate a list of %s linguistically diverse questions on the quadruple. 
For example, the input could be : (British India, capital, Kolkata, 1929) and one of your questions may be : "In 1929, what was the capital of British India? Kolkata.".  

All the questions you generate must be in past tense because the facts are not valid anymore. 
The questions must always start with the year, then a comma, then the question itself, and then finally the answer.
The questions must always be asked on the object.
The questions must be straightforward and concise.
The questions must not contain details that could make them easier to answer.

Examples of questions:
- (Jimmy Butler, member of sports team, Chicago Bulls, 2014) --> "In 2014, which team did Jimmy Butler play for? Chicago Bulls."
- (Philippines, head of state, Emilio Aguinaldo, 1900) --> "In 1900, who was the head of state of Philippines? Emilio Aguinaldo."
- (Coretta Scott King, spouse, Martin Luther King Jr., 1960) --> "In 1960, who was Coretta Scott King married to? Martin Luther King Jr."
- (European Union, currency, pound sterling, 2002) --> "In 2002, what was one of the currencies of the European Union? Pound sterling.\""""

SYSTEM_PROMPT_PRESENT = """You are an advanced knowledge verbalization system.
You take as input a knowledge triple (subject, relation, object, time) and generate a list of %s linguistically diverse questions on the triple. 
For example, the input could be : (India, capital, New Delhi) and one of your questions may be : "What was the capital of British India? Kolkata.".  

All the questions you generate must be in the present tense even if these facts are not valid anymore. 
The questions must always start with the question, then the question mark, and then finally the answer.
The questions must always be asked on the object.
The questions must be straightforward and concise.
The questions must not contain details that could make them easier to answer.

Examples of questions:
- (Jimmy Butler, member of sports team, Chicago Bulls) --> "What team does Jimmy Butler play for? Chicago Bulls."
- (Philippines, head of state, Emilio Aguinaldo) --> "Who is the head of state of Philippines? Emilio Aguinaldo."
- (Coretta Scott King, spouse, Martin Luther King Jr.) --> "Who is Coretta Scott King married to? Martin Luther King Jr."
- (European Union, currency, pound sterling) --> "What is one of the currencies of the European Union? Pound sterling.\""""


MAIN_PROMPT_PAST = """Here is the knowledge quadruple to verbalize: (%s, %s, %s, %s)."""
MAIN_PROMPT_PRESENT = """Here is the knowledge triple to verbalize: (%s, %s, %s)."""

SUPPORT_PROMPT = """Due to the ambiguity that could arise from the provided labels, here is their meaning:
- (subject) "%s" : "%s"
- (relation) "%s" : "%s"
- (object) "%s" : "%s\""""

EXAMPLE_PROMPT = """Finally, here is an example where the relation "%s" is employed : (%s, %s, %s)."""





@cache.memoize(typed=False, expire=None)
def get_response_model(prompt, model_name : str) -> str:
    # TODO: Write this function that sends the prompt to the model and then returns its answer (using packages such as litellm for example) 
    # VERY IMPORTANT: Use SYSTEM_PROMPT as a system prompt for the model.
    pass


def extract_verbalizations(message : str, triple : TimedTriple) -> list[str] | None:
    try:
        verbalizations = re.findall(r"(^|\n)([0-9]+?\.|\-) ['\"]?(.+?)['\"]*(?=(\n|$))", message)
        verbalizations = [strip_useless_characters(x[2].strip(), triple).strip() for x in verbalizations]
    except:
        return None

    return [x for x in verbalizations if len(x)]

def strip_useless_characters(verb : str, triple : TimedTriple) -> str:
    # Remove ** surrounding subjects or objects.
    verb = re.sub(rf'\*({re.escape(triple.subject.label)}|{re.escape(triple.object.label)})\*', "\1", verb)
    m = re.search(rf"\? ({re.escape(triple.subject.label)}|{re.escape(triple.object.label)})(\.?)(\"?)$", verb, flags=re.IGNORECASE)
    if m:
        dot = m.group(2)
        quote = m.group(3)
        if len(dot) or len(quote):
            return verb[:-len(dot)-len(quote)]
    return verb

def remove_date(verb : str) -> str:
    return re.split(r'In -?[0-9]+?( BCE)?, ', verb)[-1]


class QuestionAnswer:
    def __init__(self, text : str, triple : Triple):
        self.text = text
        self.answer = text.rsplit('?', 1)[-1].strip()
        self.triple = triple
        anslow = self.answer.lower()
        self.answer_type = 'subject' if anslow == triple.subject.label.lower() else 'object' if anslow == triple.object.label.lower() else 'none'

    def to_template(self, empty=False) -> Template | None:
        if self.answer.lower() != self.triple.object.label.lower() or re.match(r'^.+?\? .+?$', self.text) is None:
            return None

        question = self.text.rsplit('?', 1)[0]
        m = re.search(self.triple.subject.label, question, re.IGNORECASE)
        if m is None:
            return None
        
        temp = replace_last(self.text, self.answer, "[OBJ]")
        temp = temp.replace(m.group(0), '[SUB]', 1)
        temp = Template.from_string(temp, self.triple.relation)
        if not empty:
            temp.inject(self.triple.subject, self.triple.object)
        return temp

class LlmuVerbalizer(Verbalizer):
    def __init__(self, kb : TempWikidata, model_name: str="openai/gpt-4o", num_verbs=4, tense=Tense.PRESENT, default_year=2019):
        super().__init__()
        assert kb.stage == WikidataPrepStage.ALMOST_RAW
        self.kb = kb
        self.model_name = model_name
        self.num_verbs = num_verbs
        self.tense = tense
        self.default_year = default_year
        self._lock = threading.RLock()

    
    def verbalize(self, triple : TimedTriple) -> list[QuestionAnswer]:
        infos = tuple(getattr(ent, a) for ent in triple.to_sro() for a in ('_label', '_description'))
        try:
            year = triple.valid_between.midpoint().year
        except AttributeError:
            year = self.default_year
        assert year is not None
        if self.tense == Tense.PAST:
            prompt = MAIN_PROMPT_PAST % (triple.to_sro_str() + (year,)) + "\n\n"
            system = SYSTEM_PROMPT_PAST 
        elif self.tense == Tense.PRESENT:
            system = SYSTEM_PROMPT_PRESENT 
            prompt = MAIN_PROMPT_PRESENT % triple.to_sro_str() + "\n\n"
        else:
            raise Exception('ERROR')
        if all(x is not None for x in infos):
            prompt += SUPPORT_PROMPT % infos
            with self._lock:
                example = self.kb.retrieve_example(triple.relation)
            if example is not None:
                prompt += "\n\n" + EXAMPLE_PROMPT % ((triple.relation.label, ) + example.to_sro_str())
        prompt = [
            dict(role='system', content=system % self.num_verbs),
            dict(role='user', content=prompt)
        ]
        try:
            response = get_response_model(prompt, self.model_name)
        except ValueError:
            print('call to get_response_model failed (ValueError) with model_name=%s and prompt=%s' % (self.model_name, prompt))
            return []
        verbs = extract_verbalizations(response, triple)
        if verbs is None:
            return []
        verbs = [remove_date(v) for v in verbs]
        return [QuestionAnswer(v, triple) for v in verbs]

    def __eq__(self, value):
        if not isinstance(value, LlmuVerbalizer):
            return False
        return self.kb == value.kb and self.model_name == value.model_name
    
    def __hash__(self):
        return hash(self.kb) + hash(self.model_name)
    

def replace_last(source_string, replace_what, replace_with):
    head, _sep, tail = source_string.rpartition(replace_what)
    return head + replace_with + tail

class LlmuTempVerbalizerWrapper(TemplateVerbalizer):
    def __init__(self, llmu_verbalizer: LlmuVerbalizer):
        def follow_right_format(verb : QuestionAnswer) -> bool:
            return re.match(r'^.+?\? .+?$', verb.text) is not None
        self.llmu_verbalizer = llmu_verbalizer
        def f(triple: TimedTriple) -> list[Template]:
            qas = self.llmu_verbalizer.verbalize(triple)
            templates_scores = []
            for i, qa in enumerate(qas):
                # Skip flawed verbalizations
                if not follow_right_format(qa) or qa.answer_type != 'object':
                    continue
                temp = qa.to_template(empty=True)
                if temp is None:
                    continue
                templates_scores.append((temp, -i))
            return templates_scores
        return super().__init__(f)