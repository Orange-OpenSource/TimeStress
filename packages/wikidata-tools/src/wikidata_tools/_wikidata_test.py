# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from __future__ import annotations
import os.path as osp
import os
from ke_utils.general import mongodb_dump, mongodb_exists, mongodb_restore
from .wikidata import TempWikidata, Wikidata, WikidataPrepStage

# Small version of Wikidata for test purposes
class MiniWikidata(Wikidata):
    __test__ = False
    _TEST_ENV_TIME_LABEL = "test_env"
    _TEST_ENV_DUMP_DIRPATH = osp.join(osp.dirname(__file__), '../../tests')
    _TEST_TIME_REFERENCE = "20230227"
    QUERY_BATCHSIZE = 5

    def __init__(
        self,
        time: str,
        stage: WikidataPrepStage,
        with_triple_extension: bool = False,
        mongodb_url: str = None,
    ) -> None:
        super().__init__(
            time, stage, with_triple_extension, mongodb_url
        )
        self._suffix_name = '__test'

    def _get_build_tasks(self) -> list[tuple]:
        test_raw_name = MiniWikidata.generate_collection_name(
            self.time, WikidataPrepStage.ALMOST_RAW
        )
        test_prep_name = MiniWikidata.generate_collection_name(
            self.time, WikidataPrepStage.PREPROCESSED
        )
        real_raw_name = Wikidata.generate_collection_name(
            self.time, WikidataPrepStage.ALMOST_RAW
        )
        real_prep_name = Wikidata.generate_collection_name(
            self.time, WikidataPrepStage.PREPROCESSED
        )

        tasks = super()._get_build_tasks()
        for i in range(len(tasks)):
            if tasks[i][0] == "Push Wikidata":
                # For test purposes only
                t = tasks[i]
                tasks[i] = (
                    t[0],
                    (t[1] + " --limit 1").replace(real_raw_name, test_raw_name),
                    t[2],
                )
            elif tasks[i][0] == "Preprocess Wikidata":
                t = tasks[i]
                tasks[i] = (
                    t[0],
                    t[1]
                    .replace(real_raw_name, test_raw_name)
                    .replace(real_prep_name, test_prep_name),
                    t[2],
                )
            # elif tasks[i][0] == 'Download Wikidata':
            #     t = tasks[i]
            #     tasks[i] = (t[0], t[1].replace(TestWikidata._TEST_ENV_TIME_LABEL, TestWikidata._TEST_TIME_REFERENCE), t[2])
        return tasks

    def build(self, force=False, confirm=True) -> None:
        built = self.built()
        if not built or force:
            if not force:
                print(
                    "Test environment not built. Attempting to restore it from test dump folder (%s)..."
                    % MiniWikidata._TEST_ENV_DUMP_DIRPATH
                )
                try:
                    mongodb_restore(
                        path=MiniWikidata._TEST_ENV_DUMP_DIRPATH,
                        conn=self.client,
                        db_name=self.db.name,
                        coll_name=self.collection_name,
                    )
                    print("Test environment successfully restored!")
                    return
                except FileNotFoundError:
                    print(
                        "Test environment could not be restored because the dump was not found."
                    )

            os.makedirs(MiniWikidata._TEST_ENV_DUMP_DIRPATH, exist_ok=True)
            super().build(True, confirm)

        else:
            print("%s already exists" % self)
        if not mongodb_exists(
            MiniWikidata._TEST_ENV_DUMP_DIRPATH, self.collection_name
        ):
            print("Dumping test environment...")
            mongodb_dump(
                collections=[self.collection_name],
                conn=self.client,
                db_name=self.db.name,
                path=MiniWikidata._TEST_ENV_DUMP_DIRPATH,
            )
    
    @staticmethod
    def generate_collection_name(time : str, stage : WikidataPrepStage):
        return "wikidata__" + time + "__" + MiniWikidata._TEST_ENV_TIME_LABEL + "__" + str(stage.name)
    
    def change_stage(self, stage : WikidataPrepStage) -> MiniWikidata:
        return self.__class__(self.time, stage, self.with_triple_extension, self.mongodb_url)


class MiniTempWikidata(MiniWikidata, TempWikidata):
    __test__ = False