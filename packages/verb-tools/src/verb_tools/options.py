# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from __future__ import annotations
from collections import defaultdict

from ke_utils.glob_core import Mongoable, MongoableEnum, TimeUnit
from wikidata_tools.core import Date, Interval

def iter_subclasses(cls : type, count_classes : defaultdict, stop_class : type = None):
    count_classes[cls] += 1
    for b in cls.__bases__:
        if b is stop_class:
            continue
        iter_subclasses(b, count_classes, stop_class)


class VerbalizeOption():
    """A verbalization option describes one aspect of how a proposition (e.g., semantic triple) is verbalized
    """
    pass
    # @staticmethod
    # def validate_options(options : tuple[VerbalizeOption], verbose : bool = False) -> bool:
    #     """Detect if there is a contradiction in verbalization options. 
        
    #     For example, the presence of two options:
    #     - Option 1 : Verbalize using the present tense 
    #     - Option 2 : Verbalize using the past tense

    #     Args:
    #         options (tuple[VerbalizeOption]): Verbalization options
    #         verbose (bool, optional): Whether to print the reason why a set of options is not valid

    #     Returns:
    #         bool: True if a contradiciton exists, else, False 
    #     """
    #     print = get_print_verbose(verbose)
    #     assert all(isinstance(opt, VerbalizeOption) for opt in options), "All options must be an instance of VerbalizeOption"
    #     count_classes = defaultdict(lambda : 0)
    #     for opt in options:
    #         iter_subclasses(type(opt), count_classes, VerbalizeOption)
        
    #     for cls, n in count_classes.items():
    #         if LeafOption in cls.__bases__ and n > 1:
    #             print("There is more than one option of %s which is forbidden!" % cls.__name__)
    #             return False
        
    #     if (n := count_classes[TempIndicator]) > 1:
    #         print("There can be at most one TemporalIndicator (found : %s)!" % n)
    #         return False
    #     return True
                

class TempIndicator(VerbalizeOption, Mongoable):
    """Contextualize the verbalization using a temporal indicator (a date, a period, ...)
    """
    pass


class AbsoluteTempIndicator(TempIndicator):
    """Contextualize the verbalization using an absolute temporal indicator which states precisely and with no ambiguity the temporal context.
    """
    pass

class DateIndicator(AbsoluteTempIndicator):
    """Contextualize the verbalization using a date
    """
    def __init__(self, date : Date) -> None:
        self.date = date

    def _to_json(self) -> dict | list:
        return {"date" : self.date}

class PeriodIndicator(AbsoluteTempIndicator):
    """Contextualize the verbalization using an interval (using two dates)
    """
    def __init__(self, interval : Interval) -> None:
        self.interval = interval

    def _to_json(self) -> dict | list:
        return {"interval" : self.interval}


class RelativeTempIndicator(TempIndicator):
    """Contextualize the verbalization using a relative temporal indicator which states the time relative to the information in the triple to verbalize.
    """
    pass

class ObjectRelativeIndicator(RelativeTempIndicator, MongoableEnum):
    """Contextualize the verbalization using a relative temporal indicator which states the time relative to the object in the triple to verbalize.
    
    For example, using PRECEDENT, the triple (UK, prime minister, Boris Johnson) is verbalized as "The previous prime minister of UK is Boris Johnson" 
    ("is" could be in the past or future tense depending on the Tense option).
    """
    PRECEDENT = 0
    NEXT = 1
    CURRENT = 2
    FIRST = 3
    LAST = 4
    
    
class PresentRelativeIndicator(RelativeTempIndicator):
    """Contextualize the verbalization using a relative temporal indicator which states the time relative to the present time.
    """
    def __init__(self, n : int, time_unit : TimeUnit) -> None:
        """Initialize a present-relative time indicator 

        Args:
            n (int): The distance from the present time. It can be negative
            time_unit (TimeUnit): The unit of the distance (day, month, year)
        """
        self.n = n
        self.time_unit = time_unit

    def _to_json(self) -> dict | list:
        return dict(n=self.n, time_unit=self.time_unit)

class Order(VerbalizeOption, MongoableEnum):
    """This option says whether the object should come after the subject (NO) or in reverse order (YES).
    """
    REVERSE = 0
    STRAIGHT = 1
        
class BlankOut(VerbalizeOption, MongoableEnum):
    """Verbalize Option : Remove subject or object from the verbalization, thus, creating a template.
    """
    SUBJECT = 0
    OBJECT = 1    
    
class QuantityStyle(VerbalizeOption, MongoableEnum):
    """Verbalize Option : Write quantities in letters or numbers.
    """
    NUMBERS = 0
    LETTERS = 1

class QuantityPrecision(VerbalizeOption, Mongoable):
    """Verbalize Option : Define the precision of verbalized quantities
    """
    def __init__(self, n : int) -> None:
        """Define the precision of verbalized quantities

        Args:
            n (int): The number of decimal digits used to represent a quantity. If None, use the full precision
        """
        assert n >= 3; "Precision lower than 3 digits is not supported"
        self.n = n

    def _to_json(self) -> dict | list:
        return {
            'n' : self.n
        }
        
class DatePrecision(VerbalizeOption, Mongoable):
    """Verbalize Option : Define the precision of the verbalized dates
    """
    def __init__(self, unit : TimeUnit) -> None:
        """Define the precision of the verbalized dates

        Args:
            unit (TimeUnit): Precision of dates (year, month, day). If None, use the full precision
        """
        self.unit = unit

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, DatePrecision):
            return False
        return self.unit == value.unit
    
    def _to_json(self) -> dict | list:
        return {'unit' : self.unit}
    
    def __hash__(self):
        return hash(self.unit)

class DateStyle(VerbalizeOption, MongoableEnum):
    """Verbalize Option : Define the verbalization style of a date
    
    Example for each style:
    - RAW_ATTACHED : 20200201
    - RAW_HYPHEN : 2020-02-01
    - FULL_LETTERS : first of February two thousand and twenty
    - MONTH_LETTER_CLASSIC : 1 February 2020
    - MONTH_LETTER_AMERICAN : February 1, 2020
    """
    RAW_ATTACHED = 0
    RAW_HYPHEN = 1
    FULL_LETTERS = 2
    MONTH_LETTER_CLASSIC = 3
    MONTH_LETTER_AMERICAN = 4
    
class Tense(VerbalizeOption, MongoableEnum):
    """Verbalize Option : Define the tense of the verb
    """
    PRESENT = 0
    PAST = -1
    FUTURE = 1
    
