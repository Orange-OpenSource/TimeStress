# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from __future__ import annotations
from abc import abstractmethod
from collections import abc, defaultdict
import functools
import itertools
import json
import multiprocessing as mp
import os
import pickle
import random
import re
import shutil
from typing import Iterable, Union
from warnings import warn

from internetarchive import search_items
import numpy as np
import requests
from tqdm import tqdm

import duckdb as db
import pandas as pd

from .build_wd.utils.general import run_in_process
from .build_wd.wikidata_scripts.build_wikidata_dumps_index import (
    process_internet_archive,
    process_wikidata_dump,
)
from .core import (
    Date,
    EntityContainer,
    Interval,
    KnowledgeBase,
    Entity,
    Literal,
    Proposition,
    Quantity,
    Relation,
    String,
    TimedTriple,
    TimedTripleQuery,
    Triple,
    TripleQuery,
    Unit,
    enable_level_heterogeneous_comparison,
)
from pymongo.collection import Collection
from .mongo_utils import handle_mongodb_url, remove_from_progress_file
from ke_utils.general import (
    date_is_correct,
    get_mongodb_client,
    get_print_verbose,
    instance_tracker_factory,
    load_json,
    singleton_by_args,
    str2number,
)
from ke_utils.glob_core import Blueprint, Mongoable, MongoableEnum, SameType
from .build_wd.wikifactdiff_builder import DockLayoutExample, TaskLauncher, progress_folder
from ke_utils.globals import STORAGE_FOLDER
import os.path as osp


WIKIDATA_DUMP_URL = "https://dumps.wikimedia.org/wikidatawiki/entities/"
WIKIDATA_DUMP_PATTERN = r"^wikidata-[0-9]{8}-all.json.bz2$"
INSTANCE_OF_RELATION = Relation("P31")
SUBCLASS_OF_RELATION = Relation("P279")


class WikidataPrepStage(MongoableEnum):
    ALMOST_RAW = 0
    PREPROCESSED = 1

class ABCWikidata(KnowledgeBase):
    PROPOSITION_TYPE = Triple
    QUERY_BATCHSIZE = 1000

    def __init__(self):
        super().__init__()
        self._suffix_name = ''

    @abstractmethod
    def sample_subjects(self, n: int, contains_label=False) -> Iterable[Entity]:
        pass

    @abstractmethod
    def number_of_subjects(self) -> int:
        pass
    
    @abstractmethod
    def sample_relations(self, n: int) -> Iterable[Relation]:
        pass

    def inject_info(self, entity_container: EntityContainer | Iterable) -> None:
        batch = []
        already_injected = {}
        for i, ent in enumerate(
            EntityContainer.iter_comp_extended(entity_container), 1
        ):
            ent_cache = already_injected.get(ent)
            if ent_cache is not None:
                ent.load_info(ent_cache)
                continue
            batch.append(ent)
            if i % self.__class__.QUERY_BATCHSIZE == 0:
                self._inject_info_batch(batch)
                already_injected.update({x:x for x in batch})
                batch.clear()
        else:
            self._inject_info_batch(batch)

    @abstractmethod
    def _inject_info_batch(self, entities: list[Entity]) -> None:
        pass
    
    @abstractmethod
    def find(self, query: TripleQuery) -> Iterable[Triple]:
        pass

    @abstractmethod
    def find_from_label(self, labels: list[str]) -> dict[str, list[Entity]]:
        pass

    @abstractmethod
    def iterate_subjects(
            self, spawn_process=False, show_progress=False
        ) -> Iterable[Entity]:
        pass

    @abstractmethod
    def get_all_names_of_entity(self, entity: Entity | tuple[Entity], filter_out_none=True, show_progress=False) -> list[list[str]]:
        pass

    def retrieve_example(self, rel: Relation) -> Triple:
        raise NotImplementedError
    
    def get_all_types_of_entity(self, entity: Entity | list[Entity], 
                                valid_at: Interval | Date | str | Iterable[Interval | Date | str] = None) -> list[list]:
        if isinstance(entity, Entity):
            entity = [entity]
        it = self.find(
            TimedTripleQuery(
                subject=entity,
                relation=[INSTANCE_OF_RELATION, SUBCLASS_OF_RELATION],
                valid_at=valid_at,
            )
        )
        types = defaultdict(list)
        for tr in it:
            types[tr.subject].append(tr.object)

        return [types[ent] for ent in entity]

    

class Wikidata(ABCWikidata, Blueprint, Mongoable):
    def __init__(
        self,
        time: str,
        stage: WikidataPrepStage,
        with_triple_extension: bool = False,
        mongodb_url: str | None = None,
    ) -> None:
        super().__init__()
        mongodb_url = handle_mongodb_url(mongodb_url)

        assert (
            not with_triple_extension
        ), "Triple extension to location hierearchy is not supported yet. Please set with_triple_extension to False."
        self.mongodb_url = mongodb_url
        self.time = time
        self.stage = stage
        self.client = get_mongodb_client(mongodb_url)
        self.db = self.client.get_database("wiki")
        self.with_triple_extension = with_triple_extension
        if self.collection_name not in self.db.list_collection_names():
            warn(
                "The selected Wikidata instance does not exist in the MongoDB database. Call .build() function to download, push the Wikidata to MongoDB, and eventually preprocess Wikidata (depending on stage you set)"
            )

        if date_is_correct(time):
            self.time_date = Date(
                np.datetime64("-".join((time[:-4], time[-4:-2], time[-2:])))
            )


    @classmethod
    def get_buildings(cls, mongodb_url: str | None= None) -> list[Wikidata]:
        client = get_mongodb_client(mongodb_url)
        list_collections = client.get_database("wiki").list_collection_names()
        available = []
        pat = r"^wikidata__([0-9]{8})__([A-Z_]+$)"
        for file in list_collections:
            m = re.match(pat, file)
            if not m:
                continue
            date = m.group(1)
            stage = m.group(2)
            try:
                stage = getattr(WikidataPrepStage, stage)
            except AttributeError:
                raise Exception(
                    'Found this collection in MongoDB "%s" but the stage %s is not recognized by the system!'
                    % (m.group(0), stage)
                )

            available.append(Wikidata(date, stage))

        return available

    def _get_build_tasks(self):
        collection_raw = self.__class__.generate_collection_name(
            self.time, WikidataPrepStage.ALMOST_RAW
        )
        collection_prep = self.__class__.generate_collection_name(
            self.time, WikidataPrepStage.PREPROCESSED
        )

        tasks = [
            (
                "Download Wikidata",
                f"build_wd/wikidata_scripts/download_dump.py --date {self.time}",
                [],
            ),
            (
                "Push Wikidata",
                f"build_wd/wikidata_scripts/process_json_dump.py --date {self.time} --collection_name {collection_raw}",
                ["Download Wikidata"],
            ),
            (
                "Preprocess Wikidata",
                f"build_wd/wikidata_scripts/preprocess_dump.py --collection_in {collection_raw} --collection_out {collection_prep} --time {self.time}",
                ["Push Wikidata"],
            ),
        ]

        if self.stage == WikidataPrepStage.ALMOST_RAW:
            tasks = tasks[:-1]

        return tasks

    def sample_subjects(self, n: int, contains_label=False) -> Iterable[Entity]:
        for ent in self.collection.aggregate(
            [
                {"$sample": {"size": n}},
                {"$project": {"_id": 1, "label": 1, "description": 1}},
            ]
        ):
            ent = _mongo_dict_to_ent(ent)
            if ent._label is None and contains_label:
                continue
            yield ent

    def sample_relations(self, n: int) -> Iterable[Relation]:
        if self.stage == WikidataPrepStage.ALMOST_RAW:
            data = load_json(
                    STORAGE_FOLDER,
                osp.join(
                    "resources/script_stats/process_json_dump_%s.json"
                    % self.collection_name,
                )
            )
            relations = list(data["relation_count"].keys())
            return [Relation(x) for x in random.sample(relations, k=n)]
        elif self.stage == WikidataPrepStage.PREPROCESSED:
            relations = load_json(
                osp.join(
                    STORAGE_FOLDER,
                    "resources/property_lists/post_preproc_properties_%s.json"
                    % self.collection_name,
                )
            )
            return [Relation(x) for x in random.sample(relations, k=n)]

    @staticmethod
    def available_dumps(refresh=False, verbose=True) -> list[str]:
        """List the available Wikidata dumps identified using their date.

        This process involves exploring the internet (Wikidata dump website + Internet Archive) to find URL download links for Wikidata dumps.
        Wikidata dumps already on disk (in MongoDB or unprocessed downloaded dumps) are also added to the list.
        Then, the result is saved in a JSON file so that this internet exploration process is not done each time this function is called.
        You can force the regeneration of the JSON file using the "refresh" argument

        Args:
            refresh (bool, optional): Force the regeneration of the JSON file containing the date and download links of Wikidata dumps. Defaults to False.
            verbose (bool, optional): Print progress and general infos. Defaults to True.

        Returns:
            list[str]: List of available dumps identified by their date
        """
        print = get_print_verbose(verbose)
        json_path = os.path.join(STORAGE_FOLDER, "wikidata_dumps_index.json")
        if os.path.exists(json_path) and not refresh:
            # Index already saved
            print(
                "Existing JSON file found! Loading available dumps directly from there"
            )
            index = load_json(json_path)
        else:
            # Index generation
            index = {}

            # Check on wikidata dumps website
            html = requests.get(WIKIDATA_DUMP_URL).content.decode()
            dates = [
                x[:-1]
                for x in re.findall(r'<a href="(.+?)">.+?</a>', html)
                if x.endswith("/") and x != "../"
            ]

            print("Number of potienial dumps found in Wikidata dump = %s" % len(dates))
            with mp.Pool(max(2, mp.cpu_count() // 4)) as pool:
                results = [
                    x for x in pool.map(process_wikidata_dump, dates) if x is not None
                ]
            for url, d in results:
                index[d] = url

            # Check on Internet Archive
            identifiers = search_items(
                'title:"Wikidata entity dumps (JSON and TTL) of all Wikibase entries for Wikidata generated on"'
            )
            identifiers = [x["identifier"] for x in identifiers]
            print(
                "Number of potienial dumps found in Internet Archive = %s"
                % len(identifiers)
            )
            with mp.Pool(max(1, mp.cpu_count() // 2)) as pool:
                results = [
                    x
                    for x in pool.map(process_internet_archive, identifiers)
                    if x is not None
                ]
            for url, d in results:
                index[d] = url

            print("\n\nAll availables JSON dumps:")
            print("==========================\n")
            for x in sorted(index):
                print(x)
            print()
            print("Total number of JSON dumps found = %s" % len(index))
            os.makedirs(STORAGE_FOLDER, exist_ok=True)

            with open(json_path, "w") as f:
                json.dump(index, f)

        index = sorted(
            list(set(list(index.keys()) + Wikidata._get_available_dumps_on_disk()))
        )
        return index

    @staticmethod
    def _get_available_dumps_on_disk() -> list[str]:
        """Return the wikidata dumps that are present on disk (in STORAGE_FOLDER)

        Returns:
            list[str]: List of dates in format YYYYMMDD
        """
        available = []
        pat = r"^([0-9]{8})_wikidata\.json\.bz2$"
        for file in os.listdir(STORAGE_FOLDER):
            m = re.match(pat, file)
            if not m:
                continue
            date = m.group(1)
            if date_is_correct(date):
                available.append(date)
        return available

    @staticmethod
    def _get_available_dumps_in_mongodb(
        mongodb_url: str = None,
    ) -> list[str]:
        """Return the wikidata dumps that are present in MongoDB

        Returns:
            list[str]: List of dates in format YYYYMMDD
        """
        client = get_mongodb_client(mongodb_url)
        list_collections = client.get_database("wiki").list_collection_names()
        available = []
        pat = r"^wikidata__([0-9]{8})__[A-Z_]+$"
        for file in list_collections:
            m = re.match(pat, file)
            if not m:
                continue
            date = m.group(1)
            if date_is_correct(date):
                available.append(date)
        return list(set(available))

    def _build(self) -> None:

        if self.time not in Wikidata.available_dumps():
            raise Exception(
                f"No Wikidata dump with the specified date '{self.time}' was found!\n"
                "List the available dumps using Wikidata.available_dumps() and choose one date from the dictionary.\n"
                "You can also use Wikidata.available_dumps(refresh=True) to get the most up-to-date Wikidata dumps list."
            )

        tasks = self._get_build_tasks()
        runner = TaskLauncher(tasks)

        os.makedirs(progress_folder, exist_ok=True)
        os.makedirs(osp.join(STORAGE_FOLDER, "errors"), exist_ok=True)
        runner.run()
        app = DockLayoutExample(runner, tasks)
        app.run()
        print("%s built successfully!!" % self)

    def built(self) -> bool:
        return self.collection_name in self.db.list_collection_names()

    def _destroy(self) -> None:
        name, command, _ = self._get_build_tasks()[-1]
        remove_from_progress_file(name, command)
        self.collection.drop()

    @staticmethod
    def generate_collection_name(time: str, stage: WikidataPrepStage) -> str:
        return "wikidata__" + time + "__" + str(stage.name)

    @property
    def collection_name(self) -> str:
        return self.__class__.generate_collection_name(self.time, self.stage)

    @property
    def collection(self) -> Collection:
        return self.db.get_collection(self.collection_name)

    @staticmethod
    def _dateprec_to_date(date: str, precision: int) -> Date:
        sign = date[0] if date[0] == "-" else ""
        if precision == 11:
            return Date.from_string(sign + date[1:].split("T")[0])
        else:
            date = date[1:].split("-")
            if precision == 10:
                return Date.from_string(sign + "-".join(date[:-1]))
            else:
                return Date.from_string(sign + "-".join(date[:-2]))

    @staticmethod
    def _get_value(value_dict) -> Entity:
        """Extract entity from Wikidata dump object dict

        Args:
            value_dict (dict): Dictionary : {'type' : str, 'value' : format depends on the 'type'}. This argument can be a string in case the value type is 'string'.

        Returns:
            Entity
        """

        if isinstance(value_dict, str):
            return String(value_dict)
        if "id" in value_dict:
            # It's an entity
            return Entity(value_dict["id"])
        elif "time" in value_dict:
            date = value_dict["time"]
            precision = value_dict["precision"]
            return Wikidata._dateprec_to_date(date, precision)
        elif "text" in value_dict:
            return String(value_dict["text"])
        elif "amount" in value_dict:
            ret = value_dict["amount"].lstrip("+")

            unit = value_dict["unit"]
            if unit != "1":
                unit = unit.split("/")[-1]
                # Unit without fetching label
                unit = Unit(unit, None)
            else:
                unit = None

            return Quantity(str2number(ret), unit)
        raise ValueError(
            "The value dictionary does not contain neither an entity, nor a string, nor a date, and nor an entity"
        )

    def find(self, query: TripleQuery) -> Iterable[Triple]:
        subject, relation, object = query.subject, query.relation, query.object
        kwargs = {}
        if isinstance(query, TimedTripleQuery):
            kwargs = dict(valid_at=query.valid_at)
        if isinstance(subject, abc.Iterable):
            subs = []
            for i, ent in enumerate(subject, start=1):
                subs.append(ent)
                if i % self.__class__.QUERY_BATCHSIZE == 0:
                    for tr in self._find_batch(
                        type(query)(subs, relation, object, **kwargs)
                    ):
                        yield tr
                    subs.clear()
            else:
                for tr in self._find_batch(
                    type(query)(subs, relation, object, **kwargs)
                ):
                    yield tr

        else:
            for tr in self._find_batch(query):
                yield tr

    def number_of_subjects(self) -> int:
        return self.collection.estimated_document_count()

    def _find_batch(self, query: TripleQuery) -> Iterable[Triple]:
        subject, relation, object = query.subject, query.relation, query.object
        query_mdb, project = _build_mongo_query(query)

        object_not_none = object is not None
        if object_not_none:
            if isinstance(object, abc.Iterable):
                object = list(object)
            else:
                object = [object]

        cursor = self.collection.find(query_mdb, project)
        for ent in tqdm(cursor):
            subject = _mongo_dict_to_ent(ent)
            for prop_id, objects in ent["claims"].items():
                relation = Relation(prop_id)
                for obj_dict in objects:
                    mainsnak = obj_dict["mainsnak"]
                    datavalue = mainsnak.get("datavalue")
                    if datavalue is None:
                        continue
                    try:
                        obj = Wikidata._get_value(datavalue["value"])
                    except ValueError:
                        print(
                            "This value could not be processed :\n%s"
                            % datavalue["value"]
                        )
                        continue
                    if object_not_none and obj not in object:
                        continue
                    yield Triple(subject, relation, obj)

    def find_from_label(self, labels: list[str], show_progress=False) -> dict[str, list[Entity]]:
        labels_set = set(labels)
        labels = list(labels_set)
        label2ents = defaultdict(list)
        normalized_labels = [x.upper() for x in labels]

        cursors = (
            (i, 0, self.collection.find(
                {"label_upper": {"$in": normalized_labels[i:i+self.QUERY_BATCHSIZE]}},
                {"_id": 1, "label": 1, "description": 1}
            ))
            for i in range(0, len(normalized_labels), self.QUERY_BATCHSIZE)
        )

        def it():
            for i in range(0, len(labels), self.QUERY_BATCHSIZE):
                batch = labels[i:i+self.QUERY_BATCHSIZE]
                yield i, 1, self.collection.find(
                    {"aliases": {"$in": batch}},
                    {"_id": 1, "label": 1, "description": 1, "aliases": 1}
                )

        cursors2 = it()
        
        progress_bar = tqdm(total=(len(labels) // self.QUERY_BATCHSIZE + 1) * 2, desc='Finding entities from labels', disable=not show_progress)
        for start, cursor_type, cursor in itertools.chain(cursors, cursors2):
            for i, ent in enumerate(cursor):
                aliases = ent.pop("aliases", None)
                ent = _mongo_dict_to_ent(ent)
                if cursor_type == 0:
                    label2ents[ent.label].append(ent)
                else:
                    # Map each alias to the current entity if it matches
                    for alias in aliases:
                        if alias in labels_set:
                            label2ents[alias].append(ent)

            progress_bar.update()
            cursor.close()
        return label2ents

    def iterate_subjects(
        self, spawn_process=False, show_progress=False
    ) -> Iterable[Entity]:
        def it():
            for ent in self.collection.find(
                {}, {"_id": 1, "label": 1, "description": 1}
            ):
                yield _mongo_dict_to_ent(ent)

        if spawn_process:
            it = run_in_process(it)
        iterator = it()
        if show_progress:
            iterator = tqdm(
                iterator,
                "Iterating through subjects",
                total=self.collection.estimated_document_count(),
            )
        for x in iterator:
            yield x

    def _inject_info_batch(self, entities: list[Entity]) -> None:
        entities = [ent for ent in entities if not isinstance(ent, Literal)]
        cursor = self.collection.find(
            {"_id": _adapt_to_iter(entities)}, {"label": 1, "description": 1}
        ).sort("_id")
        id2info = {x["_id"]: x for x in cursor}
        for ent in entities:
            info = id2info.get(ent.id)
            if info is None:
                continue
            ent._label = info.get("label")
            ent._description = info.get("description")

    def get_all_names_of_entity(self, entity: Entity | tuple[Entity], filter_out_none=True, show_progress=False) -> list[list[str]]:
        """Return all the labels associated to the given entities, in the form of a dictionary {EntityInstance: list of labels}.

        The list of labels contains the main label first (which can be None if not found) and then all the aliases.
        """

        def process_aliases(ent: dict):
            aliases = ent.get("aliases", [])
            if type(aliases) is dict:
                return []
            return aliases

        if isinstance(entity, Entity):
            entity = [entity]
        list_ids = list(set(list(map(lambda x: x.id, [ent for ent in entity if not isinstance(ent, Literal)]))))
        progress_bar = tqdm(total=len(list_ids), desc='Finding entities labels', disable=not show_progress)
        cursors = (
            self.collection.find(
                {"_id": {"$in": list_ids[i : i + 10000]}},
                {"_id": 1, "aliases": 1, "label": 1},
            )
            for i in range(0, len(list_ids), 10000)
        )
        names = {}
        for cursor in cursors:
            for ent in cursor:
                l = [x for x in [ent.get("label")] + process_aliases(ent) if x is not None or not filter_out_none]
                names[ent["_id"]] = l
                progress_bar.update()
        progress_bar.close()
        return [names.get(e.id, list()) for e in entity]

    def __repr__(self) -> str:
        return "%s(time=%s, stage=%s)" % (
            self.__class__.__name__,
            self.time,
            self.stage.name,
        )

    def _to_json(self) -> dict | list:
        return {
            "time": self.time,
            "stage": self.stage.value,
            "with_triple_extension": self.with_triple_extension,
            "mongodb_url": self.mongodb_url,
        }

    @classmethod
    def _from_json(cls: type[SameType], d: dict | list) -> SameType:
        d["stage"] = WikidataPrepStage(d["stage"])
        return cls(**d)

    def _identifier(self) -> json.Any:
        return self._to_json()

    def get_all_relations(self) -> list[Relation]:
        d = json.load(
            open(
                osp.join(
                    STORAGE_FOLDER,
                    "resources/script_stats/process_json_dump_wikidata__%s__ALMOST_RAW.json"
                    % self.time,
                )
            )
        )
        props = list(Relation(x) for x in d["relation_count"].keys())
        return props

    def get_temporal_functional_relations(
        self, return_separator=False
    ) -> Union[list, dict]:
        """Get the list of single valued properties which are properties that can only have one value at a specific point in time, according to the wikidata

        Args:
            version (str, optional): The version of wikipedia to use ('old' or 'new'). Defaults to 'new'.
            return_separator (bool, optional): If True, returns the separator of each single-value property if it exists. In this case, the function returns a dictionary (key=property_id, value=separator (which can be None)). Defaults to False.

        Returns:
            [list, dict]: list of property ids or dictionary (key=property_id, value=separator (which can be None))
        """

        def flatten(container):
            for i in container:
                if isinstance(i, (list, tuple)):
                    for j in flatten(i):
                        yield j
                else:
                    yield i

        props = [r.id for r in self.get_all_relations()]
        query = [
            {"$match": {"_id": {"$in": props}}},
            {"$project": {"constraints": "$claims.P2302"}},
            {"$match": {"constraints": {"$exists": 1}}},
            {
                "$project": {
                    "constraints": {
                        "$filter": {
                            "input": "$constraints",
                            "cond": {
                                "$in": [
                                    "$$c.mainsnak.datavalue.value.id",
                                    ["Q52060874", "Q19474404"],
                                ]
                            },
                            "as": "c",
                        }
                    }
                }
            },
            {"$match": {"constraints.0": {"$exists": 1}}},
            {
                "$project": {
                    "separators": "$constraints.qualifiers.P4155.datavalue.value.id"
                }
            },
        ]
        cursor = self.collection.aggregate(query)
        res = {}
        for x in cursor:
            separators = x.get("separators", [])
            separators = list(Relation(x) for x in flatten(separators))
            res[Relation(x["_id"])] = separators
        if not return_separator:
            res = list(res.keys())
        return res

    def get_types_of_entity(self, entities: list[Entity]) -> dict[Entity, list[Entity]]:
        return TempWikidata.get_types_of_entity(self, entities)
    
    def change_stage(self, stage : WikidataPrepStage) -> Wikidata:
        return self.__class__(self.time, stage, self.with_triple_extension, self.mongodb_url)
    
    @functools.lru_cache(1024)
    def retrieve_example(self, rel: Relation) -> Triple:
        sub_obj = self.collection.find_one(
            {"_id": rel.id},
            {
                "subject": "$claims.P1855.mainsnak.datavalue.value.id",
                "object": "$claims.P1855.qualifiers.%s.datavalue.value" % rel.id,
            },
        )
        try:
            subject = Entity(sub_obj['subject'][0])
            obj_raw = sub_obj["object"][0][0]
        except (KeyError, IndexError, TypeError):
            return None
        try:
            obj = TempWikidata._get_value(obj_raw)
        except ValueError:
            # The object dictionary could not be parsed
            return None
        example = Triple(subject, rel, obj)
        self.inject_info(example)
        return example

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Wikidata):
            return False
        return (
            self.time == value.time
            and self.stage == value.stage
            and self.with_triple_extension == value.with_triple_extension
            and self.mongodb_url == value.mongodb_url
        )

    def __hash__(self) -> int:
        return hash(
            (self.time, self.stage, self.with_triple_extension, self.mongodb_url)
        )


# Helper functions
def _adapt_to_iter(ent: Entity | Iterable[Entity]):
    if isinstance(ent, abc.Iterable):
        ent = [str(x.id) for x in ent]
        return {"$in": ent}
    return ent.id


def _build_mongo_query(query: TimedTripleQuery | TripleQuery) -> tuple[dict, dict]:
    query_mdb = {}
    subject, relation = query.subject, query.relation
    project = {"_id": 1, "label": 1, "description": 1}
    if subject is not None:
        query_mdb["_id"] = _adapt_to_iter(subject)
    if relation is not None:
        if not isinstance(relation, abc.Iterable):
            relations = [relation]
        else:
            relations = relation
        for rel in relations:
            project["claims.%s" % rel.id] = 1
    else:
        project["claims"] = 1
    return query_mdb, project


def _mongo_dict_to_ent(ent_dict: dict) -> Entity:
    id_ = ent_dict["_id"]
    cls = Entity if id_.startswith("Q") else Relation
    ent = cls(ent_dict["_id"], ent_dict.get("label"), ent_dict.get("description"))
    return ent


end_time_property = "P582"
start_time_property = "P580"
point_in_time_property = "P585"


def _get_start_end_time(w: dict):
    qualifs = w.get("qualifiers", {})
    start_time, end_time = None, None
    if (
        end_time_property in qualifs
        and qualifs[end_time_property][0]["snaktype"] == "value"
    ): 
        snaks = qualifs[end_time_property]
        if len(snaks) > 1:
            raise ValueError("Found multiple 'end time' qualifiers")
        snak = snaks[0]
        time = snak["datavalue"]["value"]["time"]
        prec = snak["datavalue"]["value"]["precision"]
        end_time = Wikidata._dateprec_to_date(time, prec)

    if (
        start_time_property in qualifs
        and qualifs[start_time_property][0]["snaktype"] == "value"
    ):
        snaks = qualifs[start_time_property]
        if len(snaks) > 1:
            raise ValueError("Found multiple 'end time' qualifiers")
        snak = snaks[0]
        time = snak["datavalue"]["value"]["time"]
        prec = snak["datavalue"]["value"]["precision"]
        start_time = Wikidata._dateprec_to_date(time, prec)

    return start_time, end_time


def _get_point_in_time(snak: dict):
    """snak : ent_dict['claims']['P123'][i]"""
    qualifs = snak.get("qualifiers", {})
    point_in_time = None
    if (
        point_in_time_property in qualifs
        and qualifs[point_in_time_property][0]["snaktype"] == "value"
    ):
        snak = qualifs[point_in_time_property][0]
        time = snak["datavalue"]["value"]["time"]
        prec = snak["datavalue"]["value"]["precision"]
        try:
            point_in_time = Wikidata._dateprec_to_date(time, prec)
        except ValueError:
            pass
    return point_in_time


def _get_interval(obj_dict: dict) -> Interval:
    pit = _get_point_in_time(obj_dict)
    if pit is not None:
        start_time, end_time = pit, pit
    else:
        start_time, end_time = _get_start_end_time(obj_dict)
    interval = Interval(start_time, end_time)
    return interval


class TempWikidata(Wikidata):
    def _find_batch(
        self, query: TimedTripleQuery | TripleQuery
    ) -> Iterable[TimedTriple]:
        subject, relation, object = query.subject, query.relation, query.object
        try:
            validity = query.valid_at
        except AttributeError:
            validity = None
        query_mdb, project = _build_mongo_query(query)

        object_not_none = object is not None
        if object_not_none:
            if isinstance(object, abc.Iterable):
                object = list(object)
            else:
                object = [object]
        validity_not_none = validity is not None
        if validity_not_none:
            if not isinstance(validity, (str, Interval)) and isinstance(validity, abc.Iterable):
                validity = list(validity)
            else:
                validity = [validity]
            # Replace "present" by the timestamp of the knowledge base
            for i, v in enumerate(validity):
                if v == "present":
                    validity[i] = self.time_date

        cursor = self.collection.find(query_mdb, project)
        for ent in cursor:
            subject = _mongo_dict_to_ent(ent)
            for prop_id, objects in ent["claims"].items():
                relation = Relation(prop_id)
                for obj_dict in objects:
                    try:
                        interval = _get_interval(obj_dict)
                    except ValueError:
                        continue
                    if validity_not_none:
                        for elt in validity:
                            success = (
                                wikidata_is_valid_at(
                                    interval.start, interval.end, elt, self.time_date
                                )
                                == "valid"
                            )
                            if success:
                                # Remainder: "break" command skips the "else" section in a for loop
                                break
                        else:
                            continue
                    mainsnak = obj_dict["mainsnak"]
                    datavalue = mainsnak.get("datavalue")
                    if datavalue is None:
                        continue
                    try:
                        obj = Wikidata._get_value(datavalue["value"])
                    except ValueError:
                        continue
                    if object_not_none and obj not in object:
                        continue
                    yield TimedTriple(subject, relation, obj, interval)

    def get_types_of_entity(
        self, entities: list[Entity], valid_at=None
    ) -> dict[Entity, list[Entity]]:
        ent2types = defaultdict(list)
        cursor = self.find(
            TimedTripleQuery(entities, INSTANCE_OF_RELATION, valid_at=valid_at)
        )
        for triple in cursor:
            ent2types[triple.subject].append(triple.object)
        return ent2types

    def contains(self, prop: Proposition) -> bool:
        return super().contains(prop)


class WikidataPopularity(Blueprint, Mongoable):
    def __init__(self, time: str, mongodb_url=None) -> None:
        super().__init__()
        assert date_is_correct(time), "time must be in format YYYYMMDD"
        self.time = time
        self.mongodb_url = mongodb_url
        self.client = get_mongodb_client(mongodb_url)
        self.db = self.client.get_database("wiki")

    def _get_build_tasks(self) -> list:
        tasks = [
            (
                "Create Wikipedia Popularity",
                "build_wd/wikidata_scripts/create_database_wikipedia_consultation.py --time %s"
                % self.time,
                [],
            ),
            (
                "Compute Importance",
                "build_wd/wikidata_scripts/compute_importance_time.py --time %s"
                % self.time,
                ["Create Wikipedia Popularity"],
            ),
        ]
        return tasks

    @property
    def collection_name(self) -> str:
        return "%s_entities_importance" % self.time

    @property
    def collection(self) -> Collection:
        return self.db.get_collection(self.collection_name)

    def _build(self) -> None:
        if self.time not in Wikidata.available_dumps():
            raise Exception(
                f"No Wikidata dump with the specified date '{self.time}' was found!\n"
                "List the available dumps using Wikidata.available_dumps() and choose one date from the dictionary.\n"
                "In case you did not found the dump you are looking for, use Wikidata.available_dumps(refresh=True) to get the most up-to-date Wikidata dumps list."
            )

        tasks = self._get_build_tasks()
        runner = TaskLauncher(tasks)

        os.makedirs(progress_folder, exist_ok=True)
        os.makedirs(osp.join(STORAGE_FOLDER, "errors"), exist_ok=True)
        runner.run()
        app = DockLayoutExample(runner, tasks)
        app.run()
        print("%s built successfully!!" % self)

    def iterate_subjects(
        self, pop_order="descending"
    ) -> Iterable[tuple[Entity, float]]:
        assert pop_order in ("ascending", "descending")
        pop_order = 1 if pop_order == "ascending" else -1
        cursor = self.collection.find().sort([("ent_imp", pop_order)])
        for ent in cursor:
            yield Entity(ent["_id"]), ent["ent_imp"]

    def get_popularity(self, entities: Iterable[Entity]) -> list[float]:
        entities = list(entities)
        list_ids = list(set(list(map(lambda x: x.id, entities))))
        cursors = (
            self.collection.find({"_id": {"$in": list_ids[i : i + 1000]}})
            for i in range(0, len(list_ids), 1000)
        )
        ents = {
            Entity(ent["_id"]): ent["ent_imp"] for cursor in cursors for ent in cursor
        }
        return [ents.get(e) for e in entities]

    def number_of_subjects(self) -> int:
        return self.collection.estimated_document_count()

    def _destroy(self) -> None:
        name, command, _ = self._get_build_tasks()[0]
        remove_from_progress_file(name, command)
        self.collection.drop()

    def built(self) -> bool:
        return self.collection_name in self.db.list_collection_names()

    def __repr__(self) -> str:
        return "%s(%s)" % (self.__class__.__name__, self.time)

    def _to_json(self) -> dict | list:
        return {"time": self.time, "mongodb_url": self.mongodb_url}

    @classmethod
    def _from_json(cls: type[SameType], d: dict | list) -> SameType:
        return cls(**d)

    def _identifier(self) -> json.Any:
        self._to_json()


@singleton_by_args
class WikidataTypeSearchEngine(Blueprint):
    SAVE_PATH = osp.join(STORAGE_FOLDER, "WikidataTypeSearchEngine__data")

    def __init__(self, wd: Wikidata, valid_at=None) -> None:
        super().__init__()
        assert not isinstance(
            valid_at, (str, Date, Interval)
        ), "valid_at argument as iterable is not yet supported."
        self.wd = wd
        self.valid_at = valid_at
        self.type2ents = None
        self.ent2types = None
        self.all_types = None

    def _build(self) -> None:
        type2ents = defaultdict(list)
        ent2types = defaultdict(list)
        it = self.wd.find(
            TimedTripleQuery(
                relation=[INSTANCE_OF_RELATION, SUBCLASS_OF_RELATION],
                valid_at=self.valid_at,
            )
        )

        for triple in tqdm(
            it,
            desc="Collecting types from Wikidata",
            total=self.wd.number_of_subjects(),
        ):
            ent = triple.subject.id
            type = triple.object.id
            type2ents[type].append(ent)
            ent2types[ent].append(type)

        os.makedirs(self.folder_save_path)
        with open(self.type2ents_path, "wb") as f:
            pickle.dump(type2ents, f)

        with open(self.ent2types_path, "wb") as f:
            pickle.dump(ent2types, f)

    def get_all_types_of_entity(self, ent: Entity) -> list[Entity]:
        return self.ent2types.get(ent, list())

    def get_all_entities_of_type(self, type_: Entity) -> list[Entity]:
        return self.type2ents.get(type_, list())

    def load(self) -> None:
        def to_entity(d: dict[str, list[str]]):
            return {Entity(k): [Entity(x) for x in v] for k, v in d.items()}

        with open(self.ent2types_path, "rb") as f:
            self.ent2types = to_entity(pickle.load(f))

        with open(self.type2ents_path, "rb") as f:
            self.type2ents = to_entity(pickle.load(f))

        self.all_types = list(self.type2ents.keys())

    @property
    def folder_save_path(self) -> str:
        return osp.join(
            self.SAVE_PATH,
            self.wd.time + "__" + self.wd.stage.name + "__" + str(self.valid_at) + self.wd._suffix_name,
        )

    @property
    def type2ents_path(self) -> str:
        return osp.join(self.folder_save_path, "type2ents.pkl")

    @property
    def ent2types_path(self) -> str:
        return osp.join(self.folder_save_path, "ent2types.pkl")

    def _destroy(self) -> None:
        shutil.rmtree(self.folder_save_path, ignore_errors=True)

    def built(self) -> bool:
        return osp.exists(self.ent2types_path) and osp.exists(self.type2ents_path)


# def wikidata_is_valid_at(st : Date, et : Date, t : Date | Interval, is_present : bool) -> str:

#     with enable_level_heterogeneous_comparison():
#         if not is_present and not (st is not None and t >= st or et is not None and t >= et):
#                 return "unk" # Validity : We cannot tell
#         res = ((st is None or (st <= t)) and et is None) or \
#         ((et is None or (et >= t)) and st is None) or \
#         (et is not None and st is not None and et >= t >= st)
#         if res:
#             # Validity : valid
#             return "valid"
#         return "invalid"
#         # Validity : Invalid


def wikidata_is_valid_at(
    st: Date | None, et: Date | None, t: Date | Interval, present: Date
) -> str:
    """Says if a triple is valid at the given temporal point or interval using its validity period [st, et]. Do not use this function outside of Wikidata

    Args:
        st (Date): The start of the validity period of the triple retrieved from Wikidata (could be None if Wikidata does not specifiy it)
        et (Date): The end of the validity period of the triple retrieved from Wikidata (could be None)
        t (Date | Interval): the temporal point or interval to test for
        present (Date): The position of the present of the Wikidata Knowledge base

    Returns:
        str: "valid" means this triple is totally valid on the whole time/interval t,
        "invalid" means it's partially or totally invalid on the time/interval t, and
        "unk" means we cannot tell totally or partially on the time/interval t.
    """
    if isinstance(t, Date):
        t = Interval(t, t)
    with enable_level_heterogeneous_comparison():
        if st is None and et is None:
            if t.is_point() and t.start == present:
                return "valid"
            else:
                return "unk"
        elif st is not None and et is not None:
            if t.intersection(Interval(st, et)) == t and t.start > st and t.end < et:
                return "valid"
            else:
                return "invalid"
        elif st is not None and et is None:
            if t.start > st:
                if t.end > present:
                    return "unk"
                elif t.end <= present:
                    return "valid"
                else:
                    return "invalid"
            else:
                return "invalid"
            
            # if st > t.end:
            #     return "invalid"
            # elif present >= t.end:
            #     return "valid"
            # else:
            #     return "unk"
        else:
            if et >= t.end:
                return "invalid"
            else:
                return "unk"
        return "unk"


time_stage = lambda wd: (wd.time, wd.stage)
InMemoryWikidataIT = instance_tracker_factory(time_stage)
class InMemoryWikidata(ABCWikidata, Mongoable, InMemoryWikidataIT):

    def __init__(self, triples: list[Triple] | None = None, aliases: dict[Entity, list[str]] | None= None, 
                 relation2example : dict[Relation, Triple] | None = None, use_temporal_triples=True, time: str=None, 
                 stage: WikidataPrepStage = None):
        """Initialize an in-memory Triple knowledge base

        Args:
            triples (list[Triple] | None, optional): List of triples to add to the knowledge base. Defaults to None (no triples are added).
            use_temporal_triples (bool, optional): Whether the return triples from this base should be TimedTriple objects (triples containing the validity period of the triple). Defaults to True.
            time (str, optional): The present time of this knowledge base (YYYY-MM-DD). Specify 'present' to take the current date. Defaults to None.
            stage (WikidataPrepStage, optional): Specifiy it only if you plan to use the function change_stage. Defaults to None.
        """
        if time == "present":
            import datetime
            time = str(datetime.now().date())
        self.time = time
        if date_is_correct(time):
            self.time_date = Date(
                np.datetime64("-".join((time[:-4], time[-4:-2], time[-2:])))
            )
        self.stage = stage
        ABCWikidata.__init__(self)
        Mongoable.__init__(self)
        InMemoryWikidataIT.__init__(self)
        self.con = db.connect(':memory:')
        self._first_query_add_triples = True
        self._first_query_add_aliases = True
        self._first_query_add_relation_examples = True
        if triples is not None:
            self.add_triples(triples)
        if aliases is not None:
            self.add_aliases(aliases)
        if relation2example is not None:
            self.add_relation_examples(relation2example)
        
        self.use_temporal_triples = use_temporal_triples
        if use_temporal_triples:
            self._build_triple = lambda tr : TimedTriple(*[pickle.loads(ent) for ent in tr])
        else:
            self._build_triple = lambda tr : Triple(*[pickle.loads(ent) for ent in tr[:-1]])
        

        
    
    def add_triples(self, triples : Iterable[Triple]) -> None:
        """Add triples to knowledge base

        Args:
            triples (Iterable[Triple]): Triples
        """
        triples_ = []
        for tr in tqdm(triples):
            infos = {}
            for name in ('subject', 'relation', 'object', 'valid_between'):
                x = getattr(tr, name, None)
                p = pickle.dumps(x)
                if name != "valid_between":
                    x = {"%s_hash" % name: x._hash(ret='int', include_class_id=False), '%s_pickle' % name: p}
                else:
                    x = {'%s_pickle' % name: p}
                infos.update(x)
            triples_.append(infos)
        df = pd.DataFrame(triples_)
        df.drop_duplicates(['subject_hash', 'relation_hash', 'object_hash'], inplace=True)
        s_df = df[['subject_hash', 'subject_pickle',]].drop_duplicates('subject_hash')
        r_df = df[['relation_hash', 'relation_pickle']].drop_duplicates('relation_hash')
        o_df = df[['object_hash', 'object_pickle']].drop_duplicates('object_hash')
        
        if self._first_query_add_triples:
            self.con.execute("CREATE TABLE triples AS SELECT subject_hash, relation_hash, object_hash, valid_between_pickle FROM df")
            self.con.execute("ALTER TABLE triples ADD PRIMARY KEY (subject_hash, relation_hash, object_hash)")

            self.con.execute("CREATE TABLE subjects AS SELECT subject_hash, subject_pickle FROM s_df")
            self.con.execute("ALTER TABLE subjects ADD PRIMARY KEY (subject_hash)")
            
            self.con.execute("CREATE TABLE relations AS SELECT relation_hash, relation_pickle FROM r_df")
            self.con.execute("ALTER TABLE relations ADD PRIMARY KEY (relation_hash)")

            self.con.execute("CREATE TABLE objects AS SELECT object_hash, object_pickle FROM o_df")
            self.con.execute("ALTER TABLE objects ADD PRIMARY KEY (object_hash)")
            self._first_query_add_triples = False
        else:
            self.con.execute("INSERT IGNORE INTO triples SELECT subject_hash, relation_hash, object_hash FROM df")
            self.con.execute("INSERT IGNORE INTO subjects SELECT subject_hash, subject_pickle FROM s_df")
            self.con.execute("INSERT IGNORE INTO relations SELECT relation_hash, relation_pickle FROM r_df")
            self.con.execute("INSERT IGNORE INTO objects SELECT object_hash, object_pickle FROM o_df")
        
    def add_aliases(self, aliases: dict[Entity, list[str]]) -> None:
        data = []
        for ent, aliases_ in tqdm(aliases.items()):
            ent_hash = ent._hash(ret='int', include_class_id=False)
            for i, alias in enumerate(aliases_):
                infos = {'entity_hash': ent_hash, 'entity_alias': alias, 'position' : i}
                data.append(infos)
        df = pd.DataFrame(data)
        df.drop_duplicates(['entity_hash', 'entity_alias'], inplace=True)

        if self._first_query_add_aliases:
            self.con.execute("CREATE TABLE aliases AS SELECT entity_hash, entity_alias, position FROM df")
            self.con.execute("ALTER TABLE aliases ADD PRIMARY KEY (entity_hash, entity_alias)")
            self._first_query_add_aliases = False
        else:
            self.con.execute("INSERT IGNORE INTO aliases SELECT entity_hash, entity_alias, position FROM df")

    def add_relation_examples(self, relation2example: dict[Relation, Triple]) -> None:
        data = []
        for rel, example in tqdm(relation2example.items()):
            rel_hash = rel._hash(ret='int', include_class_id=False)
            infos = {'relation_hash': rel_hash, 'example': pickle.dumps(example)}
            data.append(infos)
        df = pd.DataFrame(data)

        if self._first_query_add_relation_examples:
            self.con.execute("CREATE TABLE relation_examples AS SELECT relation_hash, example FROM df")
            self.con.execute("CREATE INDEX relation_examples_index ON relation_examples (relation_hash)")
            self._first_query_add_relation_examples = False
        else:
            self.con.execute("INSERT IGNORE INTO relation_examples SELECT relation_hash, example FROM df")

    def find(self, query: TripleQuery | TimedTripleQuery) -> Iterable[Triple]:
        # Create temporary tables for subject, relation, and object hashes
        self.con.execute("DROP TABLE IF EXISTS temp_subject_hashes")
        self.con.execute("DROP TABLE IF EXISTS temp_relation_hashes")
        self.con.execute("DROP TABLE IF EXISTS temp_object_hashes")

        # Handle subject
        if query.subject is None:
            sub_condition = "1=1"  # No filtering
        elif isinstance(query.subject, Iterable):
            subhash = [sub._hash(ret='int', include_class_id=False) for sub in query.subject]
            subhash = pd.DataFrame({'subject_hash': subhash})
            self.con.execute("CREATE TEMPORARY TABLE temp_subject_hashes AS SELECT * FROM subhash")
            sub_condition = "triples.subject_hash IN (SELECT subject_hash FROM temp_subject_hashes)"
        else:  # Single element
            subhash = query.subject._hash(ret='int', include_class_id=False)
            subhash = pd.DataFrame({'subject_hash': [subhash]})
            self.con.execute("CREATE TEMPORARY TABLE temp_subject_hashes AS SELECT * FROM subhash")
            sub_condition = "triples.subject_hash IN (SELECT subject_hash FROM temp_subject_hashes)"

        # Handle relation
        if query.relation is None:
            rel_condition = "1=1"  # No filtering
        elif isinstance(query.relation, Iterable):
            relhash = [rel._hash(ret='int', include_class_id=False) for rel in query.relation]
            relhash = pd.DataFrame({'relation_hash': relhash})
            self.con.execute("CREATE TEMPORARY TABLE temp_relation_hashes AS SELECT * FROM relhash")
            rel_condition = "triples.relation_hash IN (SELECT relation_hash FROM temp_relation_hashes)"
        else:  # Single element
            relhash = query.relation._hash(ret='int', include_class_id=False)
            relhash = pd.DataFrame({'relation_hash': [relhash]})
            self.con.execute("CREATE TEMPORARY TABLE temp_relation_hashes AS SELECT * FROM relhash")
            rel_condition = "triples.relation_hash IN (SELECT relation_hash FROM temp_relation_hashes)"

        # Handle object
        if query.object is None:
            obj_condition = "1=1"  # No filtering
        elif isinstance(query.object, Iterable):
            objhash = [obj._hash(ret='int', include_class_id=False) for obj in query.object]
            objhash = pd.DataFrame({'object_hash': objhash})
            self.con.execute("CREATE TEMPORARY TABLE temp_object_hashes AS SELECT * FROM objhash")
            obj_condition = "triples.object_hash IN (SELECT object_hash FROM temp_object_hashes)"
        else:  # Single element
            objhash = query.object._hash(ret='int', include_class_id=False)
            objhash = pd.DataFrame({'object_hash': [objhash]})
            self.con.execute("CREATE TEMPORARY TABLE temp_object_hashes AS SELECT * FROM objhash")
            obj_condition = "triples.object_hash IN (SELECT object_hash FROM temp_object_hashes)"

        # Build SQL query
        sql_query = f"""
        SELECT 
            subjects.subject_pickle, 
            relations.relation_pickle, 
            objects.object_pickle, 
            triples.valid_between_pickle 
        FROM 
            triples 
        JOIN 
            subjects ON triples.subject_hash = subjects.subject_hash 
        JOIN 
            relations ON triples.relation_hash = relations.relation_hash 
        JOIN 
            objects ON triples.object_hash = objects.object_hash 
        WHERE 
            {sub_condition} AND 
            {rel_condition} AND 
            {obj_condition}
        """

        cursor = self.con.execute(sql_query)

        # Handle validity filtering
        validity = getattr(query, "valid_at", None)
        if validity is None:
            while (tr := cursor.fetchone()) is not None:
                yield self._build_triple(tr)

        else:
            if not isinstance(validity, (str, Interval)) and isinstance(validity, Iterable):
                validity = list(validity)
            else:
                validity = [validity]
            # Replace "present" by the timestamp of the knowledge base
            for i, v in enumerate(validity):
                if v == "present":
                    validity[i] = self.time_date
            default_interval = Interval(None, None)
            while (tr := cursor.fetchone()) is not None:
                tr = self._build_triple(tr)
                interval = getattr(tr, "valid_between", default_interval)
                for elt in validity:
                    success = (
                        wikidata_is_valid_at(
                            interval.start, interval.end, elt, self.time_date
                        )
                        == "valid"
                    )
                    if success:
                        # Reminder: "break" command skips the "else" section in a for loop
                        break
                else:
                    continue
                yield tr

        
        # Drop temporary tables after use
        self.con.execute("DROP TABLE IF EXISTS temp_subject_hashes")
        self.con.execute("DROP TABLE IF EXISTS temp_relation_hashes")
        self.con.execute("DROP TABLE IF EXISTS temp_object_hashes")

    def _inject_info_batch(self, entities: list[Entity]) -> None:
        entities = [ent for ent in entities if not isinstance(ent, Literal)]
        hashes = tuple(ent._hash(ret='int', include_class_id=False) for ent in entities)
        entities_pickles = self.con.execute(f"""
        SELECT subject_pickle
        FROM subjects
        WHERE subject_hash IN {hashes}

        UNION ALL

        SELECT relation_pickle
        FROM relations
        WHERE relation_hash IN {hashes}

        UNION ALL

        SELECT object_pickle
        FROM objects
        WHERE object_hash IN {hashes}
        """).fetchall()
        entities_fetched = [pickle.loads(x[0]) for x in entities_pickles]
        id2info = {x.id: x for x in entities_fetched}
        for ent in entities:
            info = id2info.get(ent.id)
            if info is None:
                continue
            ent._label = info._label
            ent._description = info._description

    def sample_subjects(self, n: int) -> Iterable[Entity]:
        cursor = self.con.execute(f"""
        SELECT subject_pickle FROM subjects USING SAMPLE {n}
        """)
        while (sub := cursor.fetchone()) is not None:
            yield pickle.loads(sub[0])

    def number_of_subjects(self):
        count = self.con.execute(f"""
        SELECT COUNT(*) FROM subjects
        """).fetchall()[0][0]
        return count

    def sample_relations(self, n):
        cursor = self.con.execute(f"""
        SELECT relation_pickle FROM relations USING SAMPLE {n}
        """)
        while (sub := cursor.fetchone()) is not None:
            yield pickle.loads(sub[0])

    def find_from_label(self, labels: list[str], show_progress=False) -> dict[str, list[Entity]]:
        """Find entities based on their labels.

        Args:
            labels (list[str]): List of labels to search for.

        Returns:
            dict[str, list[Entity]]: Input labels with their matched entities (if no match, the label will not be present in the output dictionary).
        """
        df = pd.DataFrame()
        df['label'] = labels
        self.con.execute("DROP TABLE IF EXISTS temp_labels")
        self.con.execute("CREATE TEMPORARY TABLE temp_labels AS SELECT * FROM df")

        # Query using a join
        query = """
        SELECT a.entity_alias, 
            COALESCE(s.subject_pickle, r.relation_pickle, o.object_pickle) AS entity_pickle
        FROM aliases a
        INNER JOIN temp_labels t ON a.entity_alias = t.label
        LEFT JOIN subjects s ON a.entity_hash = s.subject_hash
        LEFT JOIN relations r ON a.entity_hash = r.relation_hash
        LEFT JOIN objects o ON a.entity_hash = o.object_hash
        ORDER BY a.position
        """

        cursor = self.con.execute(query)
        results = cursor.fetchall()
        
        # Drop the temporary table
        self.con.execute("DROP TABLE temp_labels")
        
        # Process results into the desired dictionary format
        label_to_entities = defaultdict(list)
        for alias, entity_pickle in results:
            if entity_pickle:
                entity = pickle.loads(entity_pickle)
                label_to_entities[alias].append(entity)

        return label_to_entities

    def iterate_subjects(self, show_progress=False, **kwargs):
        cursor = self.con.execute(f"""
        SELECT subject_pickle FROM subjects
        """)
        it = iterate_duckdb_cursor(cursor)
        if show_progress:
            it = tqdm(it, total=self.number_of_subjects(), desc="Iterating subjects")
        for sub_pickle in it:
            yield pickle.loads(sub_pickle[0])
            

    def get_all_names_of_entity(self, entities: Entity | tuple[Entity], show_progress=False) -> list[list[str]]:
        """Return all the aliases associated with the given entities."""
        # Compute entity hashes
        entity_hashes = [entity._hash(ret='int', include_class_id=False) for entity in entities]

        unique_entity_hashes = list(set(entity_hashes))

        # Create a temporary table for entity hashes
        df = pd.DataFrame()
        df['entity_hash'] = unique_entity_hashes
        
        self.con.execute("DROP TABLE IF EXISTS temp_entity_hashes")
        self.con.execute("CREATE TEMPORARY TABLE temp_entity_hashes as SELECT * FROM df")
        
        # Query using a join
        query = """
        SELECT a.entity_hash, a.entity_alias
        FROM aliases a
        INNER JOIN temp_entity_hashes t ON a.entity_hash = t.entity_hash
        ORDER BY a.position
        """
        cursor = iterate_duckdb_cursor(self.con.execute(query))

        # Optionally show progress
        if show_progress:
            cursor = tqdm(cursor, desc="Fetching aliases")

        # Process results into a dictionary
        hash_to_aliases = defaultdict(list)
        for entity_hash, alias in cursor:
            hash_to_aliases[entity_hash].append(alias)

        # Drop the temporary table
        self.con.execute("DROP TABLE temp_entity_hashes")

        # Return aliases in the same order as the input entity hashes
        return [hash_to_aliases.get(float(h), []) for h in entity_hashes]


    def save(self, folderpath: str) -> None:
        shutil.rmtree(folderpath, ignore_errors=True)
        os.makedirs(folderpath, exist_ok=True)
        data_path = osp.join(folderpath, 'data')
        self.con.execute(f"""EXPORT DATABASE '{data_path}' (FORMAT parquet);""")
        config_path = osp.join(folderpath, 'config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'use_temporal_triples': self.use_temporal_triples,
                'time': self.time,
                'stage': self.stage.name if self.stage else None
            }, f)

    @staticmethod
    def from_disk(folderpath: str) -> InMemoryWikidata:
        data_path = osp.join(folderpath, 'data')
        config_path = osp.join(folderpath, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            config['stage'] = None if config['stage'] is None else getattr(WikidataPrepStage, config['stage'])
        wd = InMemoryWikidata(**config)
        wd.con.execute(f"""IMPORT DATABASE '{data_path}';""")

        if table_exists(wd.con, 'aliases'):
            wd._first_query_add_aliases = False

        if table_exists(wd.con, 'triples'):
            wd._first_query_add_triples = False

        if table_exists(wd.con, 'relation_examples'):
            wd._first_query_add_relation_examples = False
        return wd

    def change_stage(self, stage: WikidataPrepStage) -> InMemoryWikidata:
        ts = (self.time, stage)
        instance = self.get_instance_by_key(ts)
        if instance is None:
            raise Exception("InMemoryWikidata instance with configuation time=%s and stage=%s was not initialized in the current process.")
        return instance

    # BUTCHER: I'm butchering my code here but I don't have time unfortunately
    # EDIT: I think this is resolved and can be removed (to test)
    def built(self) -> bool:
        return True
    
    @functools.lru_cache(1024)
    def retrieve_example(self, rel: Relation) -> Triple | None:
        h = rel._hash(ret="int", include_class_id=False)
        res = self.con.execute(f"SELECT example FROM relation_examples WHERE relation_hash = {h}").fetchall()
        if len(res) == 0:
            return None
        example = pickle.loads(res[0][0])
        return example


def iterate_duckdb_cursor(cursor) -> Iterable:
    while (obj := cursor.fetchone()) is not None:
        yield obj

def table_exists(con, table_name):
    """
    Check if a table exists in the DuckDB database.

    Args:
        con (duckdb.DuckDBPyConnection): The DuckDB connection object.
        table_name (str): The name of the table to check.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    query = f"""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_name = '{table_name}'
    """
    result = con.execute(query).fetchall()
    return len(result) > 0