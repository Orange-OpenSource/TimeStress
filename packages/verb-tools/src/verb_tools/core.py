# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from __future__ import annotations
from curses.ascii import isalnum, islower, isupper
from dataclasses import dataclass
from functools import lru_cache
import re
import string
from typing import Any, Callable, Iterable
from abc import ABC, abstractmethod

import pandas as pd
from ke_utils.glob_core import Mongoable, TimeUnit
from wikidata_tools.core import Date, Interval, Proposition, Quantity, Relation, String, Triple, Entity, TripleComp
from ke_utils.general import PrintableException, load_json
from .options import DateIndicator, DatePrecision, Order, PeriodIndicator, TempIndicator, Tense, BlankOut, QuantityStyle, QuantityPrecision, DateStyle
# import spacy
# TODO : spacy support is dropped until they support numpy >2.0
from mlconjug3 import Conjugator
import os.path as osp
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
nltk.download('punkt_tab')

class Verbalizer(ABC):
    """A Verbalizer is a object that can verbalize a proposition in some language. For example, the semantic triple 
    (France, capital, Paris) can be verbalized in natural language as "The capital of France is Paris".
    """
    @abstractmethod
    def verbalize(self, prop : Proposition) -> Iterable[Template]:
        """Verbalize a proposition

        Args:
            prop (Proposition): Proposition

        Returns:
            Iterable[Template]: A sequence of templates
        """
        pass

# _powers = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 100)
# _human_powers = (
#     "",
#     "thousand",
#     "million",
#     "billion",
#     "trillion",
#     "quadrillion",
#     "quintillion",
#     "sextillion",
#     "septillion",
#     "octillion",
#     "nonillion",
#     "decillion",
#     "googol",
# )
_abreviate_powers = (
    '',
    "K",
    "M",
    "G",
    "P",
    "E",
    "Z",
    "Y"
)

_month_dict = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

class EntityFormatter:
    SUPPORTED_DATE_STYLES = [DateStyle.MONTH_LETTER_CLASSIC, DateStyle.MONTH_LETTER_AMERICAN, DateStyle.RAW_ATTACHED, DateStyle.RAW_HYPHEN]
    def __init__(self, config : VerbalizeConfig) -> None:
        self.config = config
        assert self.config.quantity_style != QuantityStyle.LETTERS, "Formating numbers as letters is not supported yet"
        assert self.config.date_style in EntityFormatter.SUPPORTED_DATE_STYLES, "Formating date other that %s is not supported yet" % EntityFormatter.SUPPORTED_DATE_STYLES

    def format(self, entity : Entity) -> str:
        assert isinstance(entity, Entity), "entity must be of type Entity"
        if isinstance(entity, Quantity):
            return self.format_quantity(entity)
        elif isinstance(entity, Date):
            return self.format_date(entity)
        elif isinstance(entity, String):
            return self.format_string(entity)
        else:
            return self.format_pure_entity(entity)
        
        
    def format_string(self, string : String) -> str:
        return '"%s"' % string.id
    
    def get_date_format(self, date_precision : TimeUnit) -> str:
        conf_prec = self.config.date_precision
        if conf_prec is None:
            conf_prec = date_precision
        date_style = self.config.date_style
        formatted_date = None
        for prec in sorted((e for e in TimeUnit), key=lambda x : x.value):
            if date_precision <= prec and conf_prec <= date_precision :
                if date_precision == TimeUnit.DAY:
                    if date_style == DateStyle.MONTH_LETTER_CLASSIC:
                        formatted_date = "{day} {month_name} {year} {minus}"
                    elif date_style == DateStyle.MONTH_LETTER_AMERICAN:
                        formatted_date = "{month_name} {day}, {year} {minus}"
                elif date_precision == TimeUnit.MONTH:
                    formatted_date = "{month_name} {year} {minus}"
                elif date_precision == TimeUnit.YEAR:
                    formatted_date = "{year} {minus}"
                break
            
        assert formatted_date is not None, "Date format"
        return formatted_date

    # Constant useful for format_date function
    TIMEUNIT2KEEP = {
        TimeUnit.DAY : 3,
        TimeUnit.MONTH : 2,
        TimeUnit.YEAR : 1
    }
    @staticmethod
    def format_date_static(date : Date, precision : DatePrecision, style : DateStyle, use_preposition_comma=False) -> str:
        """
        Formats a date string based on its precision ("Day", "Month", or "Year").
        
        Parameters:
        - date : Date
        - precision : To what precision the date is verbalized
        - style : Which style to use to verbalize a date (UK, American, YYYY-MM-DD, etc.)
        - use_preposition_comma : Whether to additionally format the date in the right format "In [DATE], " or "On [DATE], " depending on the date precision
        
        Returns:
        - str: The formatted date string.
        """
        input_date = str(date)
        # input_date = input_date.lstrip('+')
        formatted_date = ""

        minus = 'BCE' if input_date.startswith('-') else ''
        input_date = input_date.lstrip('-')

        # Extract _description_the day, month, and year
        comp = input_date.split('-')
        if len(comp) < 3:
            comp += [None] * (3-len(comp))
        year, month, day = comp
        year = year.lstrip('0')

        date_precision = date.level
        date_style = style

        # Use the _month_dict to get the month name
        try:
            month_name = _month_dict[int(month)]
        except TypeError:
            month_name = None
        conf_prec = precision
        if conf_prec is None:
            conf_prec = date_precision
        else:
            conf_prec = conf_prec.unit
        formatted_date = None
        for prec in sorted((e for e in TimeUnit), key=lambda x : x.value):
            if date_precision <= prec and conf_prec >= date_precision:
                if date_style in [DateStyle.RAW_ATTACHED, DateStyle.RAW_HYPHEN]:
                    sep = "" if date_style == DateStyle.RAW_ATTACHED else "-"
                    formatted_date = sep.join((minus, year, month, day)[:EntityFormatter.TIMEUNIT2KEEP[conf_prec]+1])
                    if date_style == DateStyle.RAW_HYPHEN:
                        formatted_date = formatted_date[1:] # remove extra hyphen at the start
                elif conf_prec == TimeUnit.DAY:
                    if date_style == DateStyle.MONTH_LETTER_CLASSIC:
                        formatted_date = f"{day.lstrip('0')} {month_name} {year} {minus}"
                    elif date_style == DateStyle.MONTH_LETTER_AMERICAN:
                        formatted_date = f"{month_name} {day.lstrip('0')}, {year} {minus}"
                elif conf_prec == TimeUnit.MONTH:
                    formatted_date = f"{month_name} {year} {minus}"
                elif conf_prec == TimeUnit.YEAR:
                    formatted_date = f"{year} {minus}"
                break
            
        assert formatted_date is not None, "Date could not be verbalized (Date=%s, level=%s, date_precision=%s, date_style=%s)" % (date, date.level, date_precision, date_style)
        formatted_date = formatted_date.strip(' ')
        if use_preposition_comma:
            # Add preprosition and comme to date
            if date.level == TimeUnit.DAY:
                formatted_date = "On %s, " % formatted_date
            else:
                formatted_date = "In %s, " % formatted_date
        return formatted_date

    def format_date(self, date : Date) -> str:
        return self.format_date_static(date, self.config.date_precision, self.config.date_style)

    def format_quantity(self, quantity : Quantity) -> str:
        res = ""
        q_str = str(quantity.value)
        if self.config.quantity_precision is None:
            # Print the whole quantity
            res = q_str
        elif self.config.quantity_precision is not None:
            n = self.config.quantity_precision.n
            q_str = (f"%.{n-1}E") % quantity.value
            decimal_part, exponent = q_str.split('E')
            exponent = int(exponent)
            if exponent >= 0:
                degree = exponent // 3
                remaining = exponent % 3
                symbol = _abreviate_powers[degree]
                if remaining > 0:
                    n_dec = remaining + 1
                    # decimal_part = (f"%.{n}E" % float(decimal_part*10**remaining)).split('E')[0]
                    decimal_part = decimal_part.replace('.', '')
                    decimal_part = decimal_part[:n_dec] + '.' + decimal_part[n_dec:]
            else:
                symbol = ""
                decimal_part = decimal_part + ("e%s" % exponent)
            res = decimal_part + " " + symbol
        
        if self.config.include_unit and quantity.unit is not None:
            res += quantity.unit.symbol
        res = res.strip()
        return res

    def format_pure_entity(self, entity : Entity) -> str:
        return entity.label
    
    def __call__(self, *args: Any, **kwds: Any) -> str:
        return self.format(*args, **kwds)

@dataclass(eq=True, frozen=True)
class VerbalizeConfig(Mongoable):
    max_num_verbs : int = None # The maximal number of verbalizations to produce per triple (templates are ordered with decreasing qulity)
    index_template : tuple[int] = None # The tuple of template indices to use
    verb_tense : Tense = None # Verbalize in present tense. If None, keep the tense of the original template.
    temporal_indicator : TempIndicator = None # No temporal indicator used
    date_precision : DatePrecision = None # Take full date
    date_style : DateStyle = DateStyle.MONTH_LETTER_AMERICAN # Classic way of representing dates (e.g. 1 February 2020)
    date_precision_temporal_indicator : DatePrecision = None # Same thing for temporal_indicator
    date_style_temporal_indicator : DateStyle = DateStyle.MONTH_LETTER_AMERICAN
    quantity_precision : QuantityPrecision = QuantityPrecision(3) # Keep only 3 most left digits + an indicator of magnitude (e.g., M,B,K)
    quantity_style : QuantityStyle = QuantityStyle.NUMBERS # Write numbers using digits
    include_unit : bool = True # Whether to include the unit in a quantity. Defaults to True
    order : Order = None # Whether the object should come before the subject or not. Defaults to None which means no preference
    blank_outs : tuple[BlankOut] = tuple() # Do not blank-out/delete subject or object
    ends_with : TripleComp = None # Keep only verbalizations that end with the subject or object. Defaults to None, which means keep everything
    

DEFAULT_ENTITY_FORMATTER = EntityFormatter(VerbalizeConfig())
DEFAULT_VERB_CONFIG = VerbalizeConfig()


class VerbError(PrintableException):
    pass
class VerbNotFound(VerbError):
    pass

class TemplateNotFound(VerbError):
    pass

def conjugate_verb_forms(verb : str, row: int, tense : Tense) -> str:
    # Special case for "are"
    if verb == 'are':
        if tense == Tense.FUTURE:
            return 'will be'
        if tense == Tense.PRESENT:
            return 'are'
        if tense == Tense.PAST:
            return 'were'
    base = TemplateVerbalizer.dictionary_verb_forms.loc[row, 'base']
    if tense == Tense.FUTURE:
        return "will " + base
    if tense == Tense.PRESENT:
        if base == verb:
            return base
        third = TemplateVerbalizer.dictionary_verb_forms.loc[row, 'present_simple_3rd']
        return third
    elif tense == Tense.PAST:
        return TemplateVerbalizer.dictionary_verb_forms.loc[row, 'past_simple']


def conjugate_mlconjug3(verb : str, lemma : str, conjug : dict, tense : Tense) -> str:
    if tense == Tense.FUTURE:
        return "will " + lemma
    path = _conjugate_find_value(verb, [], conjug)
    if path is None:
        return
    if tense == Tense.PRESENT:
        return conjug['indicative']['indicative present'][path[-1]]
    elif tense == Tense.PAST:
        return conjug['indicative']['indicative past tense'][path[-1]]



def _conjugate_find_value(verb : str, parent : list, conjug : dict) -> str:
    out = None
    for k,v in conjug.items():
        if isinstance(v, dict):
            parent.append(k)
            out = _conjugate_find_value(verb, parent, v)
            if out is not None:
                break
            parent.pop()
        else:
            if v == verb:
                parent.append(k)
                return parent 
    return out        
    
    

class TemplateVerbalizer(Verbalizer):
    """This verbalizer verbalizes propositions using predefined templates. Each template is associated to a relation.
    Each template verbalizes specifically the target fact without any other facts involved such as "[SUB] is the capital of [OBJ]"
    """
    
    def __init__(self, proposition2templates_func : Callable[[Proposition], tuple[tuple[Template, float]]]) -> None:
        """Initialize template-based verbalizer.

        Args:
            proposition2templates_func (Callable[[Proposition], tuple[tuple[Template, float]]]): A verbalization function that 
            takes as input a proposition (e.g., a knowledge triple) and returns a tuples of templates with their associated scores,
            that indicates the quality of the template. Higher quality templates will be ordered first when a proposition is verbalized.
            This function should return None when the verbalization fails.
        """  
        # template_scores = {rel : sorted(score.items(), key=lambda x : x[1], reverse=True) for rel, score in template_scores.items()}
        # self._rel2temp : dict[Relation, tuple[tuple[Template, int]]] = template_scores
        self.proposition2templates_func = proposition2templates_func

    nlp = None # Spacy model
    conjugator = None # mlconjug3 English Conjugator 
    dictionary_verb_forms = None # verb forms scrapped from dictionary (https://github.com/monolithpl/verb.forms.dictionary/tree/master)
    verb2row = None

    TENSE2STR = {
        Tense.PRESENT : "present",
        Tense.FUTURE : "future",
        Tense.PAST : "past"
    }

    @staticmethod
    def load_verb_forms():
        if TemplateVerbalizer.verb2row is not None:
            return 
        verb_forms = load_json(osp.join(osp.dirname(__file__), 'verbs-dictionaries.json'))
        TemplateVerbalizer.dictionary_verb_forms = pd.DataFrame(verb_forms, columns=['base', 'present_simple_3rd', 'past_simple', 'past_participle', 'present_participle'])
        TemplateVerbalizer.verb2row = {verb : i for i, verbs in enumerate(verb_forms) for verb in verbs}
        print('Verb forms from dictionary loaded!')

    @staticmethod
    def load_spacy_model():
        if TemplateVerbalizer.nlp is None:
            model_name = "en_core_web_sm"
            print('Loading spacy model (%s)...' % model_name)
            # TemplateVerbalizer.nlp = spacy.load(model_name)
            print('Model loaded!')

    @staticmethod
    def load_mlconjug3():
        if TemplateVerbalizer.conjugator is None:
            TemplateVerbalizer.conjugator = Conjugator(language='en')

    def conjugate_template(self, template : Template, config : VerbalizeConfig, type = "dictionary") -> None:
        assert type in ('spacy+mlconjug3', 'dictionary', 'dictionary+spacy')
        if type == 'spacy+mlconjug3':
            self._conjugate_template_spacy_mlconjug3(template, config)
        elif type in ('dictionary', 'dictionary+spacy'):
            self._conjugate_template_dictionary(template, config, use_spacy=type == 'dictionary+spacy')


    def _conjugate_template_dictionary(self, template : Template, config : VerbalizeConfig, use_spacy=False) -> None:
        # Load verb forms
        TemplateVerbalizer.load_verb_forms()
        if use_spacy:
            TemplateVerbalizer.load_spacy_model()

        # Initialize a variable to store the unique verb
        unique_verb = None
        
        if use_spacy:
            # Define the sentence
            sentence = template.apply_str("ABC", "DEF")
            doc = TemplateVerbalizer.nlp(sentence)

            # Iterate over the tokens to find the unique verb
            for token in doc:
                row = TemplateVerbalizer.verb2row.get(token.text)
                if token.text == 'are' or row is not None and token.pos_ not in ["ADJ"]:  # Check if the token is a verb
                    unique_verb = token.text  # Store the verb
                    break  # Stop after finding the first verb

        else:
            # Iterate over the tokens to find the unique verb
            tokens = nltk.tokenize.word_tokenize(template._template)
            det_or_prop_before = False # Special rule to avoid treating nouns as 
            for token in tokens:
                row = TemplateVerbalizer.verb2row.get(token)
                if row is not None and not det_or_prop_before or token == 'are':  # Check if the token is a verb
                    unique_verb = token  # Store the verb
                    break  # Stop after finding the first verb
                det_or_prop_before = token.lower() in prepositions_and_determinants
        # if unique_verb is None:
        #     raise VerbNotFound("Problem finding verb in template (%s) for relation %s" % (template, template.relation))
        
        if unique_verb is not None:
            # Conjugate the verb
            conjugated_verb = conjugate_verb_forms(unique_verb, row, tense=config.verb_tense)
            if conjugated_verb is None:
                raise VerbNotFound("Conjugations of the verb %s could not be found in verb forms for template = %s" % (unique_verb.lemma_, template))
            
            template.replace_in_template(' ' + unique_verb + ' ', ' ' + conjugated_verb + ' ')
        
    

    def _conjugate_template_spacy_mlconjug3(self, template : Template, config : VerbalizeConfig) -> None:
        # Load the spaCy model
        TemplateVerbalizer.load_spacy_model()
        # Load mlconjug3 English conjugator
        TemplateVerbalizer.load_mlconjug3()


        # Define the sentence
        sentence = template.apply_str("the subject", "the object")

        # Process the sentence using spaCy
        doc = TemplateVerbalizer.nlp(sentence)

        # Initialize a variable to store the unique verb
        unique_verb = None

        # Iterate over the tokens to find the unique verb
        for token in doc:
            if token.pos_ in ["AUX", "VERB"] and unique_verb is None:  # Check if the token is a verb
                unique_verb = token  # Store the verb
                break  # Stop after finding the first verb
        if unique_verb is None:
            raise VerbNotFound("Problem finding verb in template (%s) for relation %s" % (template, template.relation))
        
        # Conjugate the verb to past tense using Pattern
        conjugations = TemplateVerbalizer.conjugator.conjugate(unique_verb.lemma_)

        if conjugations.conjug_info is None:
            raise VerbNotFound("Conjugations of the verb %s could not be found in mlconjug3" % unique_verb.lemma_)
        conjugated_verb = conjugate_mlconjug3(unique_verb.text, unique_verb.lemma_, conjugations.conjug_info, tense=config.verb_tense)
        if conjugated_verb is None:
            raise VerbNotFound("Conjugations of the verb %s could not be found in mlconjug3" % unique_verb.lemma_)
        
        template.replace_in_template(unique_verb.text, conjugated_verb)

    @lru_cache()
    def verbalize(self, triple : Triple, config : VerbalizeConfig = None, skip_failed_verbalizations = False) -> list[Template]:
        """Verbalize a triple

        Args:
            triple (Triple): A semantic triple. E.g., (UK, prime minister, Rishi Sunak)
            config (VerbalizeConfig, optional): Sequence of options that describe how a proposition should be verbalized. 
            Defaults to None (use default options).

        Returns:
            list[Template]: Templates (full or not)
        """
        if config is None:
            config = DEFAULT_VERB_CONFIG
        templates = self.proposition2templates_func(triple)
        templates = [temp.copy() for temp, _ in sorted(templates, key=lambda x : x[1], reverse=True)]
        if config.order is not None:
            templates = [x for x in templates if x.order == config.order]
        
        if config.ends_with is not None:
            templates = [x for x in templates if x.ends_with == config.ends_with]
        
        if templates in (None, []):
            exc = TemplateNotFound("The templates for proposition %s failed." % triple)
            if skip_failed_verbalizations:
                print(exc)
                return []
            else:
                raise exc
        
        subject, object = triple.subject, triple.object
        
        if config.index_template is None:
            idx_temp = range(config.max_num_verbs) if config.max_num_verbs is not None else range(len(templates))
        else:
            idx_temp = config.index_template

        conjugated = []
        for temp in [templates[i] for i in idx_temp if i < len(templates)]:
            temp : Template
            temp.config = config
            if config.verb_tense is not None:
                if skip_failed_verbalizations:
                    try:
                        self.conjugate_template(temp, config=config)
                    except VerbError as e:
                        print(e)
                        continue
                else:
                    self.conjugate_template(temp, config=config)
            if isinstance(config.temporal_indicator, DateIndicator):
                # Lower case first letter of template
                temp._template = temp._template[0].lower() + temp._template[1:]
                date_txt = temp.entity_formatter.format_date_static(config.temporal_indicator.date, 
                                                                    config.date_precision_temporal_indicator,
                                                                    config.date_style_temporal_indicator)
                # Adapt preprosition to date precision
                if config.temporal_indicator.date.level == TimeUnit.DAY:
                    temp.insert_in_template("On %s, " % date_txt, pos=0)
                else:
                    temp.insert_in_template("In %s, " % date_txt, pos=0)

            elif isinstance(config.temporal_indicator, PeriodIndicator):
                # Lower case first letter of template
                temp._template = temp._template[0].lower() + temp._template[1:]
                start, end = config.temporal_indicator.interval
                assert start is not None or end is not None, "When using a Period Temporal Indicator, it must contain at least the start or end date."
                start_txt = temp.entity_formatter.format_date_static(start, 
                                                                    config.date_precision_temporal_indicator,
                                                                    config.date_style_temporal_indicator) \
                            if start is not None else None
                end_txt = temp.entity_formatter.format_date_static(end, 
                                                                    config.date_precision_temporal_indicator,
                                                                    config.date_style_temporal_indicator) \
                            if end is not None else None
                if start_txt is not None and end_txt is None:
                    temp.insert_in_template("Since %s, " % start_txt, pos=0)
                elif start_txt is None and end_txt is not None:
                    temp.insert_in_template("Until %s, " % end_txt, pos=0)
                else:
                    temp.insert_in_template("Between %s and %s, " % (start_txt, end_txt), pos=0)
                    
            temp.inject(subject, object)
            
            # Correct for determiner errors
            # correct_determiner_errors(temp)
            capitalize(temp)

            if BlankOut.OBJECT in config.blank_outs:
                temp.delete(TripleComp.OBJECT)
            if BlankOut.SUBJECT in config.blank_outs:
                temp.delete(TripleComp.SUBJECT)
            
            conjugated.append(temp)

        return conjugated

def capitalize(temp : Template) -> None:
    temp._template = temp._template[0].upper() + temp._template[1:]

def correct_determiner_errors(template : Template):
    words = [x for x in re.split(r'(\[OBJ\]|\[SUB\]| )', template.apply_str('[SUB]', '[OBJ]')) if x not in ('', ' ')]
    for x in ('[SUB]', '[OBJ]'):
        i = words.index(x)
        if i == 0:
            continue
        prev_word = words[i-1]
        try:
            target = template.subject.label if x == '[SUB]' else template.object.label
        except AttributeError:
            continue
        # first_word_target = target.split(' ', maxsplit=1)[0]
        pos = template.subject_pos if x == '[SUB]' else template.object_pos
        if prev_word == 'an' and target[0] in consonants:
            replace_word_before_idx(template, pos, 'a ')
        elif prev_word == 'a' and target[0] in vowels:
            replace_word_before_idx(template, pos, 'an ')
        elif prev_word == 'the' and target.lower() not in common_nouns:
            replace_word_before_idx(template, pos, '')
        # elif first_word_target.lower() in common_nouns and isupper(target[0]) and target.lower() not in ('a', 'an', 'the') and target.count(' ') > 0:
        #     template.insert_in_template('the ' if pos > 0 else "The ", pos)

def replace_word_before_idx(template : Template, idx : int, new_word : str):
    a = template._template[:idx]
    add_space = False
    if a[-1] == ' ':
        a = a[:-1]
        add_space = True
    for i in range(1,len(a)+1):
        if a[-i] == ' ':
            break
    else:
        raise Exception('Previous word not found')
    pos = idx-i-int(add_space)+1
    template.delete_in_template(pos, i)
    template.insert_in_template(new_word, pos)



vowels = "aeiou" + "AEIOU"
consonants = set([char for char in string.ascii_letters if char.isalpha() and char.lower() not in {'a', 'e', 'i', 'o', 'u'}])
common_nouns = [x for x in open(osp.join(osp.dirname(__file__), 'words.txt')).read().split('\n') if len(x) and islower(x[0])]
prepositions_and_determinants = set(
    ["about", "like", "above", "near", "across", "of", "after", "off", "against", 
     "on", "along", "onto", "among", "opposite", "around", "out", "as", "outside",
       "at", "over", "before", "past", "behind", "round", "below", "since", "beneath", 
       "than", "beside", "through", "between", "to", "beyond", "towards", "by", "under", 
       "despite", "underneath", "down", "unlike", "during", "until", "except", "up", "for",
         "upon", "from", "via", "in", "with", "inside", "within", "into", "without"] +
         ['the', 'a', 'an']
)

# class Verbalization():
#     def __init__(self, text : str, triple : Triple, verbalizer : Verbalizer, config: VerbalizeConfig) -> None:
#         self.text = text
#         self.triple = triple
#         self.verbalizer = verbalizer
#         self.config = config
    
#     def __repr__(self) -> str:
#         return self.text
        
class TemporalContext:
    def __init__(self, is_present : bool, interval : Interval, date : Date) -> None:
        self.is_present = is_present
        self.interval = interval
        self.date = date

    @property
    def time(self) -> Date | Interval | str:
        if self.date is not None:
            return self.date
        elif self.interval is not None:
            return self.interval
        elif self.is_present:
            return "present"
        raise Exception("A timestamp couldn't be infered from this temporal context")

    @staticmethod
    def from_verb_config(verb_config : VerbalizeConfig | None) -> TemporalContext:
        # assert verb_config.verb_tense is not None, "verb_config.verb_tense needs to be specified!"
        if (verb_config.temporal_indicator is None) and (verb_config.verb_tense in (Tense.PRESENT, None)):
            return TemporalContext(is_present=True, interval=None, date=None)
        elif verb_config.verb_tense != Tense.PRESENT:
            if isinstance(verb_config.temporal_indicator, DateIndicator):
                return TemporalContext(is_present=False, interval=None, date=verb_config.temporal_indicator.date)
            elif isinstance(verb_config.temporal_indicator, PeriodIndicator):
                return TemporalContext(is_present=False, interval=None, date=verb_config.temporal_indicator.interval)
        raise ValueError("The temporal context couldn't be constructed from the verbalization configuration")

class Template(Mongoable):
    _IDX2BLANK = ['[SUB]', '[OBJ]']
    _COMP2ATTR = {
        TripleComp.SUBJECT : "subject",
        TripleComp.OBJECT : "object"
    }
    def __init__(self, template : str, relation : Relation, subject : Entity = None, object : Entity = None, verb_config : VerbalizeConfig = None) -> None:
        assert isinstance(relation, Relation), "Cannot initialize a template without a relation"
        # assert insert_sub_at is not None and insert_obj_at is not None, "insert_sub_at and insert_obj_at must be specified"
        self._template = template
        # self.insert_obj_at = insert_obj_at
        # self.insert_sub_at = insert_sub_at
        self.relation = relation
        self.subject = subject
        self.object = object
        self._verb_config = verb_config if verb_config is not None else DEFAULT_VERB_CONFIG

        # self.time can take 3 type of values: a Date, an interval, or a string equal to present
        self.time = TemporalContext.from_verb_config(self._verb_config).time 
        self.entity_formatter = EntityFormatter(self._verb_config)

    def insert_in_template(self, text : str, pos : int) -> None:
        # obj_cond = pos < self.insert_obj_at if not before else pos <= self.insert_obj_at
        # sub_cond = pos < self.insert_sub_at if not before else pos <= self.insert_sub_at
        # if obj_cond:
        #     self.insert_obj_at += len(text)
        # if sub_cond:
        #     self.insert_sub_at += len(text)
        self._template = self._template[:pos] + text + self._template[pos:]
    
    def delete_in_template(self, pos : int, n : int) -> None:
        new_pos = min(pos+n, len(self._template))
        # real_n = new_pos-pos
        # obj_cond = pos < self.insert_obj_at if not before else pos <= self.insert_obj_at
        # sub_cond = pos < self.insert_sub_at if not before else pos <= self.insert_sub_at
        # if obj_cond:
        #     self.insert_obj_at -= real_n
        # if sub_cond:
        #     self.insert_sub_at -= real_n
        self._template = self._template[:pos] + self._template[new_pos:] 

    def replace_in_template(self, txt : str, sub : str) -> None:
        self._template = self._template.replace(txt, sub)
        # n = len(txt)
        # pos = self._template.index(txt)
        # self.delete_in_template(pos, n)
        # self.insert_in_template(sub, pos)

    @property
    def subject_pos(self) -> int:
        return self._template.index("[SUB]")

    @property
    def object_pos(self) -> int:
        return self._template.index("[OBJ]")

    @property
    def template(self) -> str:
        return self._template

    @staticmethod
    def from_string(template : str, relation : Relation) -> Template:
        """Construct template from a string. The template should contain two special strings [SUB] and [OBJ] to specify where to put the subject and object 

        Args:
            template (str): A template as a string. For example, "The capital of [SUB] is [OBJ]"

        Returns:
            Template
        """
        [template.index(x) for x in Template._IDX2BLANK]
        # if obj_idx > sub_idx:
        #     obj_idx -= len(Template._IDX2BLANK[0])
        # else:
        #     sub_idx -= len(Template._IDX2BLANK[1])
        # for x in Template._IDX2BLANK:
        #     template = template.replace(x, '')
        return Template(template, relation)
        
    @property
    def config(self) -> VerbalizeConfig:
        return self._verb_config
    
    @config.setter
    def config(self, value : VerbalizeConfig):
        self._verb_config = value
        self.entity_formatter = EntityFormatter(self._verb_config)
        try:
            self.time = TemporalContext.from_verb_config(self._verb_config).time
        except ValueError:
            self.time = None

    def to_dict(self) -> dict:
        # Not everything is implemented yet
        d = {
            'template' : self._template,
            'relation' : self.relation.to_dict(),
            # 'insert_sub_at' : self.insert_sub_at,
            # 'insert_obj_at' : self.insert_obj_at
        }
        d.update({k:v.to_dict() for k, v in {'subject': self.subject, 'object' : self.object, 'verb_config' : self._verb_config} if v is not None})
        return d
    
    @staticmethod
    def from_dict(d : dict) -> Template:
        return Template(**d)

    def inject(self, sub : Entity = None, obj : Entity = None):
        assert not (sub is not None) or (self.subject is None), "Subject provided but this template already has a subject. Set sub=None"
        assert not (obj is not None) or (self.object is None), "Object provided but this template already has an object. Set obj=None"
        if self.subject is None:
            self.subject = sub
        else:
            sub = self.subject
        if self.object is None:
            self.object = obj
        else:
            obj = self.object
        return self

    def delete(self, comp : TripleComp) -> Template:
        attr = Template._COMP2ATTR.get(comp)
        assert attr is not None, "Component %s cannot be deleted from this template" % comp.name
        to_delete = getattr(self, attr)
        assert to_delete is not None, "There is nothing to delete because self.%s is already None" % attr
        setattr(self, attr, None)
        return self
    
    @property
    def order(self) -> Order:
        if self.object_pos > self.subject_pos:
            return Order.STRAIGHT
        else:
            return Order.REVERSE

    def copy(self) -> Template:
        return Template(self._template, self.relation, self.subject, self.object)
    
    @property
    def full(self) -> bool:
        return self.subject is not None and self.object is not None
    
    @property
    def ends_with(self) -> TripleComp:
        if self._template.endswith('[OBJ]'):
            return TripleComp.OBJECT
        elif self._template.endswith('[SUB]'):
            return TripleComp.SUBJECT
        return None

    @property
    def triple(self) -> Triple:
        return Triple(self.subject, self.relation, self.object)
        
    @property
    def text(self) -> str:
        args = tuple((x if x is None else self.entity_formatter(x)) for x in (self.subject, self.object))
        return self.apply_str(*args)

        
    def apply_str(self, sub : str, obj : str) -> str:
        # order = [(x if x is not None else Template._IDX2BLANK[i],pos) for i, (x, pos) in enumerate([(sub, self.insert_sub_at), (obj, self.insert_obj_at)])]
        # order = sorted(order, key=lambda x : x[1])
        # last_pos = 0
        # res = ''
        # for x, pos in order:
        #     res += self._template[last_pos:pos] + x
        #     last_pos = pos
        # else:
        #     res += self._template[last_pos:]
        res = self._template
        if sub is not None:
            res = res.replace('[SUB]', sub)
        
        if obj is not None:
            res = res.replace('[OBJ]', obj)            
        
        return res
    
    def __repr__(self) -> str:
        return self.text

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Template):
            return False
        return self._template == value._template and self.triple == value.triple

    def __hash__(self) -> int:
        return hash(self._template) + hash(self.triple)
    
    def _to_json(self) -> dict | list:
        return {
            'template' : self._template,
            "relation" : self.relation,
            # "insert_obj_at" :  self.insert_obj_at,
            # "insert_sub_at" : self.insert_sub_at,
            "subject" : self.subject,
            "object" : self.object,
            "verb_config" : self._verb_config
        }
    

class Relation2TemplatesVerbalizer(TemplateVerbalizer):
    def __init__(self, template_scores : dict[Relation, dict[Template, int]]) -> None:
        self.template_scores = {rel : sorted(score.items(), key=lambda x : x[1], reverse=True) for rel, score in template_scores.items()}
        def f(p : Triple) -> tuple[tuple[Template, float]]:
            temps = self.template_scores.get(p.relation, None)
            return temps
        super().__init__(f)