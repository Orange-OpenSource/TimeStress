# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from dataclasses import dataclass
import os
import time
import openai
import os.path as osp
from glob_core import Mongoable, SaveableNotFoundError
from globals import STORAGE_FOLDER
from utils.general import load_json
from verb.core import Template


# OpenAI API config
# IMPORTANT : Set these constants to use ChatGPT to verbalize triples
OPENAI_API_TYPE = ""
OPENAI_API_BASE = ""
OPENAI_API_VERSION = ""
OPENAI_ENGINE = ""

PRE_PROMPT = """You are a expert typo corrector. You will receive a sentence and correct it.
You are only interested in the syntax of the sentence.

If a sentence has no syntax error, return it as it is.

Example:
"The population of the Japan is 126 millions" --> "The population of Japan is 126 millions" (incorrect use of "the" determinant)
"John Smith worked as an doctor" --> "John Smith worked as a doctor" (confusion between "a" and "an")
"Electromagnetism was a branch of physics being" --> "Electromagnetism was a branch of physics" ("being" has nothing to do in the sentence)
"The president of USA is Keanu Reeves" --> "The president of USA is Keanu Reeves" (no syntax error, return the sentence as it is)

You must output the corrected sentence only"""


MAX_TOKENS = 500

@dataclass
class ChatGPTReturn(Mongoable):
    input_text : str
    output_text : str
    pre_prompt : str

    def _identifier(self) -> os.Any:
        return {
            'input_text' : self.input_text,
            'pre_prompt' : self.pre_prompt
        }

class ChatGPTSpellingCorrector:
    def __init__(self, caching=True) -> None:
        openai.api_type = OPENAI_API_TYPE
        openai.api_base = OPENAI_API_BASE
        openai.api_version = OPENAI_API_VERSION
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if openai.api_key is None or len(openai.api_key) == 0:
            raise Exception('Error : OPENAI_API_KEY not set!')

        self.openai_config = {
            'api_type' : openai.api_type,
            'api_base' : openai.api_base,
            'api_version' : openai.api_version,
        }
        self.caching = caching
    
    def correct(self, template : Template):
        prompt = template.text
        if self.caching:
            try:
                ret = ChatGPTReturn.from_id({"input_text" : prompt, "pre_prompt" : PRE_PROMPT})
                template.spelling_correction = ret.output_text
                return ret.output_text
            except SaveableNotFoundError:
                pass
        try:
            response = openai.ChatCompletion.create(
                engine=OPENAI_ENGINE,
                messages = [{"role":"system","content":PRE_PROMPT},{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=MAX_TOKENS,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
        except (openai.error.InvalidRequestError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout) as e:
            time.sleep(1)
            self.correct(template)

        message = None
        if response['choices'][0]['finish_reason'] != 'content_filter':
            message = response['choices'][0]['message']['content']
        if self.caching:
            ret = ChatGPTReturn(prompt, message, PRE_PROMPT)
            ret.save()
        template.spelling_correction = message
        return message