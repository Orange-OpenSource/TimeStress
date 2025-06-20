# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

import json
import re

import os.path as osp
from collections import defaultdict, Counter
from ..gpt3_5_verbalization.utils import (
    KnowledgeTriple,
    Entity,
    Literal,
    Property,
)
from typing import Union
from ..config import STORAGE_FOLDER


DEFAULT_CHATGPT_VERBALIZATIONS_PATH = osp.join(
    osp.dirname(__file__), "../chatgpt_verbalization_result"
)
DEFAULT_CHATGPT_VERBALIZATIONS_RAW_THEN_TEMPLATES_PATH = osp.join(
    STORAGE_FOLDER, "chatgpt_wfd_verbalization"
)


def blank_out(fill_in_the_blank: str, subject_label: str):
    p1 = rf'["\']?(The )?{re.escape(subject_label)}((?=\'s)|["\']?)'
    m = re.search(p1, fill_in_the_blank, re.IGNORECASE)
    if m is None:
        return None
    m.span(0)
    a = re.sub(p1, "XXXX", fill_in_the_blank, count=1, flags=re.IGNORECASE)
    p2 = rf'["\']?____'
    if re.search(p2, a, re.IGNORECASE):
        a = re.sub(p2, "____", a, flags=re.IGNORECASE)
    return a


month_dict = {
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
    12: "December",
}


class FormatterForHumans:
    def __init__(self) -> None:
        pass

    def format(self, obj: Union[Entity, Literal]):
        if isinstance(obj, Entity):
            return obj.label

        if obj.description in ["Date", "Month", "Year"]:
            return self.format_date(obj.label, obj.description)
        elif obj.description == "Quantity":
            return self.format_quantity(obj.label)
        return obj.label

    def format_quantity(self, quantity: str):
        quantity = quantity.lstrip("+")
        return quantity

    def format_date(self, input_date: str, date_type: str):
        """
        Formats a date string based on its type ("Date", "Month", or "Year").

        Parameters:
        - input_date (str): The date string to be formatted.
            - For date_type="Date", it should be in "DD-MM-YYYY" format.
            - For date_type="Month", it should be in "MM-YYYY" format.
            - For date_type="Year", it should be in "YYYY" format.

        - date_type (str): The type of the date, which can be one of the following:
            - "Date": Full date (day, month, year)
            - "Month": Only month and year
            - "Year": Only year

        Returns:
        - str: The formatted date string.
            - For date_type="Date", returns in "Month Day, Year" format.
            - For date_type="Month", returns in "Month Year" format.
            - For date_type="Year", returns the year as is.
            - For invalid date_type, returns None
        """
        input_date = input_date.lstrip("+")
        formatted_date = ""

        minus = " BCE" if input_date.startswith("-") else ""
        input_date = input_date.lstrip("-")

        if date_type == "Date":
            # Extract _description_the day, month, and year
            year, month, day = input_date.split("-")

            # Use the month_dict to get the month name
            month_name = month_dict[int(month)]

            # Format the date as required
            formatted_date = f"{month_name} {day}, {year}{minus}"

        elif date_type == "Month":
            # Parse the input date using datetime

            # Extract the month and year
            year, month = input_date.split("-")

            # Use the month_dict to get the month name
            month_name = month_dict[int(month)]

            # Format the date as required
            formatted_date = f"{month_name} {year}{minus}"

        elif date_type == "Year":
            # Since it's just a year, no parsing is required
            formatted_date = input_date + minus

        else:
            return None

        return formatted_date


def get_fill_in_the_blank(verb, test_label):
    # Replaces the test_label in the verbalization with "____". If it does not exist return None
    v = verb["verbalization"]
    search_label = re.search(rf"{re.escape(test_label)}[\.]?$", v, re.IGNORECASE)
    ends_with_label = search_label is not None
    if ends_with_label:
        fill_in_the_blank = re.sub(
            rf"{re.escape(test_label)}[\.]?$", r"____", v, flags=re.IGNORECASE
        )
    else:
        fill_in_the_blank = None
    return fill_in_the_blank


def iterator_verbalization_files(verbalization_path: str):
    files = []
    for filename in ("verbalizations_old_wikidata", "verbalizations_wfd"):
        verb_jsonl_path = osp.join(verbalization_path, "%s.jsonl" % filename)
        verb_jsonl_xz_path = osp.join(verbalization_path, "%s.jsonl.xz" % filename)
        if osp.exists(verb_jsonl_path):
            print("Verbalizations file found: %s" % verb_jsonl_path)
            files.append(open(verb_jsonl_path))
        else:
            print("Verbalizations file not found: %s" % verb_jsonl_path)
            print("Looking for compressed version: %s" % verb_jsonl_xz_path)
            import lzma

            f = lzma.open(verb_jsonl_xz_path, mode="rt", encoding="utf-8")
            files.append(f)
    for f in files:
        for x in f:
            yield x


class Verbalizer:
    def __init__(self, verbalization_path=None) -> None:
        print("LOADING TRIPLE VERBALIZER...")
        if verbalization_path is None:
            verbalization_path = DEFAULT_CHATGPT_VERBALIZATIONS_PATH

        self.human_formatter = FormatterForHumans()

        best_templates_path = osp.join(verbalization_path, "best_templates.json")
        best_templates_reverse_path = osp.join(
            verbalization_path, "best_templates_reverse.json"
        )

        if osp.exists(best_templates_path) and osp.exists(best_templates_reverse_path):
            self.best_templates = json.load(open(best_templates_path))
            self.best_reverse_templates = json.load(open(best_templates_reverse_path))
            return

        prop_fitb = defaultdict(list)
        prop_rev_fitb = defaultdict(list)

        for x in iterator_verbalization_files(verbalization_path):
            x = json.loads(x)
            if x["error"] is not None:
                continue
            prop = x["triple"]["relation"]["id"]
            subject_label = x["triple"]["subject"]["label"]
            object_label = x["triple"]["object"]["label"]
            verbs = [
                get_fill_in_the_blank(y, object_label) for y in x["verbalizations"]
            ]

            verbs = [blank_out(y, subject_label) for y in verbs if y is not None]
            verbs = [y for y in verbs if y is not None]

            reverse_verbs = [
                get_fill_in_the_blank(y, subject_label) for y in x["verbalizations"]
            ]
            reverse_verbs = [
                blank_out(y, object_label) for y in reverse_verbs if y is not None
            ]
            reverse_verbs = [y for y in reverse_verbs if y is not None]

            if len(verbs):
                prop_fitb[prop].extend(verbs)
            if len(reverse_verbs):
                prop_rev_fitb[prop].extend(reverse_verbs)
        self.best_templates = {
            prop_id: Counter(verbs).most_common(5)
            for prop_id, verbs in prop_fitb.items()
        }
        self.best_reverse_templates = {
            prop_id: Counter(verbs).most_common(5)
            for prop_id, verbs in prop_rev_fitb.items()
        }

        if not osp.exists(best_templates_reverse_path):
            json.dump(self.best_templates, open(best_templates_path, "w"))

        if not osp.exists(best_templates_reverse_path):
            json.dump(
                self.best_reverse_templates, open(best_templates_reverse_path, "w")
            )

    def verbalize(
        self,
        triple: KnowledgeTriple,
        exclude: Union[str, list[str]] = [],
        reverse=False,
        return_formatted_subject_and_object=False,
    ) -> list[str]:
        """Verbalize the given triple using templates. The verbalizations are ordered in decreasing "quality".

        Args:
            triple (KnowledgeTriple): Triple to verbalize
            exclude (Union[str, list[str]], optional): Can contain only two values 'subject' and 'object'. This argument specifies which part of the template should not be replaced with their corresponding label. Defaults to [].

        Returns:
            list[str]: list of verbalizations (5 maximum)
        """
        if isinstance(exclude, str):
            exclude = [exclude]

        prop = triple.relation.id
        best_templates = (
            self.best_templates if not reverse else self.best_reverse_templates
        )
        if prop not in best_templates:
            return None
        templates = best_templates[prop]
        verbs = []
        for t, _ in templates:
            if "subject" not in exclude:
                subject_formatted = self.human_formatter.format(triple.subject)
                t = t.replace("XXXX", subject_formatted)
            if "object" not in exclude:
                object_formatted = self.human_formatter.format(triple.object)
                t = t.replace("____", object_formatted)
            verbs.append(t)
        if return_formatted_subject_and_object:
            return verbs, subject_formatted, object_formatted
        return verbs


if __name__ == "__main__":
    v = Verbalizer()
    sub = Entity("Q30", "USA", "")
    obj = Entity("Q6279", "Joe", "")
    rel = Property("P35", "", "")
    triple = KnowledgeTriple(sub, rel, obj)
    print(v.verbalize(triple))
