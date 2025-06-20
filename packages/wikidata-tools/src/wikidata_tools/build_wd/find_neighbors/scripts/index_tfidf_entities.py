# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from collections import defaultdict
import sys
from wikidata_tools.build_wd.utils.wd import db, ent_to_vec, wikidata_iterator
from wikidata_tools.build_wd.utils.general import dump_json
from wikidata_tools.build_wd.gpt3_5_verbalization.verbalize_wikidata import run_in_process
from sklearn.feature_extraction.text import TfidfVectorizer

import nmslib
import scipy.io
from wikidata_tools.build_wd.find_neighbors.config import (
    TFIDF_FULL_INDEX_FOLDER,
    TFIDF_WIKIPEDIA_INDEX_FOLDER,
)
import os.path as osp
import shutil
import os


def save_list(l: list[str], path: str):
    with open(path, "w") as f:
        for x in l:
            f.write(x + "\n")


def identity(x):
    return x


def update_dict_of_list(dict_of_list: dict[str, list], new_info: dict[str, list]):
    for k, l in new_info.items():
        dict_of_list[k].extend(l)


def main(full=False, time="old", add_rev_info=False):
    # Reset environment
    save_folder = TFIDF_WIKIPEDIA_INDEX_FOLDER if not full else TFIDF_FULL_INDEX_FOLDER
    if time == "new":
        save_folder += "_new"
    shutil.rmtree(save_folder, ignore_errors=True)
    os.mkdir(save_folder)

    seen_entities = set()
    list_seen_entities = []
    ent_vectors = []
    entid2idx = {}
    reverse_info = defaultdict(list)
    version_count = [0, 0]
    order = ["old", "new"] if time == "old" else ["new", "old"]
    print(
        "Collecting entity features (relation-object couples, relations, and objects)..."
    )
    for version_i, version in enumerate(order):
        print("From Wikidata %s" % version)
        for ent_id, claims in wikidata_iterator(version, list(seen_entities), not full):
            seen_entities.add(ent_id)
            list_seen_entities.append(ent_id)
            ent_vec, rev_info = ent_to_vec(ent_id, claims, add_singles=True)
            if add_rev_info:
                update_dict_of_list(reverse_info, rev_info)
                entid2idx[ent_id] = len(ent_vectors)
            ent_vectors.append(ent_vec)
            version_count[version_i] += 1

    if add_rev_info:
        # Add reverse information
        for ent_id, features in reverse_info.items():
            idx = entid2idx.get(ent_id)
            if idx is not None:
                ent_vectors[idx].extend(features)
        # Clear memory
        del entid2idx
        del reverse_info

    tfidf = TfidfVectorizer(analyzer=identity, norm=None)

    print("Building TF-IDF vectors...")
    features_sparses = tfidf.fit_transform(ent_vectors)
    print("TF-IDF Matrix Shape : %s" % str(features_sparses.shape))
    dump_json(osp.join(save_folder, "tfidf_vectorizer.json"), tfidf.vocabulary_)

    features_sparses_0, features_sparses_1 = (
        features_sparses[: version_count[0]],
        features_sparses[version_count[0] :],
    )
    entities_0, entities_1 = (
        list_seen_entities[: version_count[0]],
        list_seen_entities[version_count[0] :],
    )
    save_list(entities_0, osp.join(save_folder, "entities_%s.txt" % order[0]))
    save_list(entities_1, osp.join(save_folder, "entities_%s.txt" % order[1]))
    scipy.sparse.save_npz(
        osp.join(save_folder, "features_sparses_%s.npz" % order[0]), features_sparses_0
    )
    scipy.sparse.save_npz(
        osp.join(save_folder, "features_sparses_%s.npz" % order[1]), features_sparses_1
    )

    print("Index creation...")
    # Initialize NMSLib index
    index = nmslib.init(
        method="hnsw",
        space="cosinesimil_sparse",
        data_type=nmslib.DataType.SPARSE_VECTOR,
    )

    # Add data points to the index
    index.addDataPointBatch(features_sparses_0)

    # Create the index
    index.createIndex({"post": 2}, print_progress=True)
    print("\nSave index...")
    index.saveIndex(osp.join(save_folder, "index.bin"), save_data=True)
    print("Finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Compute embeddings for all Wikidata entities. WARNING : Do not select this option unless you have >400GB of RAM",
    )
    parser.add_argument(
        "--time",
        type=str,
        choices=["old", "new"],
        help="It can take two values : 'old' which builds a KNN at time=old and 'new' which builds a KNN at time=which builds a KNN at time=new. Defaults to 'old'",
        default="old",
    )
    parser.add_argument(
        "--add_rev_info",
        action="store_true",
        help="Add reverse information in entity vectors, i.e., if the script finds the triple (s,r,o), it adds s,r, and (s,r) to the features of o. It is deactivated by default.",
    )
    args = parser.parse_args()
    main(full=args.full, time=args.time, add_rev_info=args.add_rev_info)
