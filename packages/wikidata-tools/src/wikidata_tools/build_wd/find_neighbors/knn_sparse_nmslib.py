# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from typing import Iterable, Union
import numpy as np
from wikidata_tools.build_wd.find_neighbors.config import (
    TFIDF_WIKIPEDIA_INDEX_FOLDER,
    TFIDF_FULL_INDEX_FOLDER,
)
from wikidata_tools.build_wd.find_neighbors.scripts.index_tfidf_entities import (
    main as main_nmslib_index,
)

import os.path as osp
import nmslib
import scipy.sparse

from wikidata_tools.build_wd.utils.general import load_json


def _check_if_nmslib_index_exists(path):
    return osp.exists(osp.join(path, "index.bin"))


def read_list(path: str) -> list[str]:
    with open(path, "r") as f:
        l = [x.strip() for x in f]
    return l


class KNearestNeighbors:
    def __init__(
        self,
        full=False,
        time="old",
        load_vocab: bool = False,
        add_reverse_features=False,
    ) -> None:
        """Class to setup a neareset neighbor system using NMSlib on entity labels from wikidata.
        Each entity is encoded into a sparse TF-IDF vector

        full (bool, optional): Load saved KNN that contains all Wikidata entities. It False, loads the one containing only Wikipedia entities.
        time (str, optional): Load the KNN corresponding to the old or new wikidata. Defaults to 'old'
        load_vocab (bool, optional): Load vocabulary (dictionary token-to-index) that was used to compute TF-IDF vectors. Defaults to False.
        add_reverse_features (bool, optional): Load the NN system in which reverse information were incorporated in entities' vector representation. Adding reverse information is to add (s,r), s, and r to the bag of words of o when seeing (s,r,o)
        """
        assert time in ["old", "new"]
        self.index = None
        self.entities_in_index = None
        self.entities_other = None
        self.entities_in_index_name2idx = None
        self.entities_other_name2idx = None
        self.tfidf_old = None
        self.tfidf_new = None
        self.full = full
        self.time = time
        self.location = (
            TFIDF_WIKIPEDIA_INDEX_FOLDER if not self.full else TFIDF_FULL_INDEX_FOLDER
        )
        if self.time == "new":
            self.location += "_new"
        self.load_vocab = load_vocab
        self.vocab: dict = None
        self.add_reverse_features = add_reverse_features
        if add_reverse_features:
            self.location += "_with_revfeat"

    def _load_vocab(self):
        print("Loading Vocabulary...")
        self.vocab = load_json(osp.join(self.location, "tfidf_vectorizer.json"))

    def setup(self, reset=False):
        if reset or not _check_if_nmslib_index_exists():
            print("NMSlib index NOT detected.")
            print("NMSlib index creation launched.")
            main_nmslib_index()

        print("Environment for KNearestNeighbors setuped!")

    def load_index(self):
        if not _check_if_nmslib_index_exists(self.location):
            raise Exception(
                'Error : Index does not exist ("%s" folder does not exist or is incomplete).\nCall the setup() function to create it.'
                % self.location
            )
        self.index = nmslib.init(
            method="hnsw",
            space="cosinesimil_sparse",
            data_type=nmslib.DataType.SPARSE_VECTOR,
        )
        print("Loading Index...")
        self.index.loadIndex(osp.join(self.location, "index.bin"), load_data=True)
        self.index.setQueryTimeParams({"efSearch": 2000})

        print("Loading entity names...")
        order = ["old", "new"] if self.time == "old" else ["new", "old"]
        self.entities_in_index = read_list(
            osp.join(self.location, "entities_%s.txt" % order[0])
        )
        self.entities_other = read_list(
            osp.join(self.location, "entities_%s.txt" % order[1])
        )
        self.entities_in_index_name2idx = {
            k: i for i, k in enumerate(self.entities_in_index)
        }
        self.entities_other_name2idx = {k: i for i, k in enumerate(self.entities_other)}

        print("Loading TF-IDF sparse matrix...")
        self.tfidf_old = scipy.sparse.load_npz(
            osp.join(self.location, "features_sparses_old.npz")
        )
        self.tfidf_new = scipy.sparse.load_npz(
            osp.join(self.location, "features_sparses_new.npz")
        )

        self.order = (
            (
                ("old", self.entities_in_index_name2idx),
                ("new", self.entities_other_name2idx),
            )
            if self.time == "old"
            else (
                ("new", self.entities_in_index_name2idx),
                ("old", self.entities_other_name2idx),
            )
        )
        if self.load_vocab:
            self._load_vocab()

    def get_vectors(
        self, ids: Union[str, Iterable[str]]
    ) -> Union[np.ndarray, list[np.ndarray]]:
        if self.index is None:
            raise Exception(
                "Error : Index not loaded. Load the index using KNearestNeighbors.load_index() function."
            )
        if is_single := isinstance(ids, str):
            ids = [ids]
        vectors = []
        for ent_id in ids:
            for version, id2int in self.order:
                idx = id2int.get(ent_id, None)
                if idx is not None:
                    break
            else:
                print("WARNING : %s ID not found in the index" % ent_id)
                return None
            if version == "old":
                vector = self.tfidf_old[idx]
            else:
                vector = self.tfidf_new[idx]
            vectors.append(vector)
        if is_single:
            return vectors[0]
        return vectors

    def get_nearest(self, emb_vector: list[np.ndarray], k=10):
        if self.index is None:
            raise Exception(
                "Error : Index not loaded. Load the index using KNearestNeighbors.load_index() function."
            )
        ids, distances = self.index.knnQueryBatch(emb_vector, k=k)[0]
        ids = [self.entities_in_index[i] for i in ids]
        return ids, distances


if __name__ == "__main__":
    from wikidata_tools.build_wd.utils.wd import get_info_wikidata

    knn = KNearestNeighbors()
    knn.load_index()
    print(
        [
            get_info_wikidata(x.strip())["name"]
            for x in knn.get_nearest(knn.get_vectors("Q142"), k=100)[0]
        ]
    )
    print("hhhh")
