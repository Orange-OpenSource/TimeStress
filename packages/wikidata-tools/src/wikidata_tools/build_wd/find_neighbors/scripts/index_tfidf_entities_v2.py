# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

import sys
from typing import Iterable

import pandas as pd
from wikidata_tools.build_wd.utils.wd import adapt_name
from _old_codebase.utils.wd import db, ent_to_vec
from _old_codebase.gpt3_5_verbalization.verbalize_wikidata import run_in_process
from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm
import nmslib
import scipy.io
from _old_codebase.find_neighbors.config import (
    TFIDF_FULL_INDEX_FOLDER,
    TFIDF_WIKIPEDIA_INDEX_FOLDER,
)
import os.path as osp
import shutil
import os
from joblib import dump
from dask import bag, array
from dask_ml.feature_extraction.text import CountVectorizer
import h5py


@run_in_process
def generator_wikipedia(version: str, seen_entities: set[str]):
    # idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1
    total = db["wikidata_%s_prep" % version].estimated_document_count()
    seen_entities = set(seen_entities)

    for x in tqdm.tqdm(
        db["wikidata_%s_prep" % version].aggregate(
            [
                {"$project": {"_id": 1}},
                {
                    "$lookup": {
                        "from": adapt_name(version),
                        "localField": "_id",
                        "foreignField": "_id",
                        "as": "wd",
                    }
                },
                {"$replaceRoot": {"newRoot": {"$first": "$wd"}}},
                {"$project": {"claims": 1}},
                # {"$limit" : 1000}
            ]
        ),
        total=total,
        mininterval=5,
        file=sys.stdout,
    ):
        if x["_id"] in seen_entities:
            continue
        yield x["_id"], x["claims"]


@run_in_process
def generator_full(version: str, seen_entities: set[str]):
    # total = db['wikidata_%s_json' % version].estimated_document_count()
    total = 100
    seen_entities = set(seen_entities)

    for x in tqdm.tqdm(
        db[adapt_name(version)].find({}, {"claims": 1}).limit(100),
        total=total,
        mininterval=5,
        file=sys.stdout,
    ):
        if x["_id"] in seen_entities:
            continue
        yield x["_id"], x["claims"]


def save_list(l: list[str], path: str):
    with open(path, "w") as f:
        for x in l:
            f.write(x + "\n")


def identity(x: str) -> str:
    return x


def process_one_entity(ent_id, claims):
    ent_vec = ent_to_vec(claims, add_singles=True)
    return ent_vec


# @delayed
# def save_ent_vec_to_file(file, ent_id : str, ent_vec : list[str]):
#     file.write("%s, %s\n" % (ent_id, ent_vec))

# def process_one_instance(x, seen_entities, list_seen_entities, version_count, version_i, file):
#     ent_id, claims = x
#     seen_entities.add(ent_id)
#     list_seen_entities.append(ent_id)
#     ent_vec = process_one_entity(ent_id, claims)
#     version_count[version_i] += 1


def entire_generator(full, vocab: set):
    seen_entities = set()
    generator = generator_full if full else generator_wikipedia

    for version_i, version in enumerate(["old", "new"]):
        print("From Wikidata %s" % version)
        for ent_id, claims in generator(version, list(seen_entities)):
            ent_vec = process_one_entity(ent_id, claims)
            yield ent_id, ent_vec, version_i
            vocab.update(ent_vec)


def compute_bag_of_words(full: bool):
    print(
        "Collecting entity features (relation-object couples, relations, and objects)..."
    )
    vocab = set()
    for i, x in enumerate(generator_full("old", set())):
        if i == 20:
            break
        print(x[0])
    it = bag.from_sequence(seq=entire_generator(full, vocab), partition_size=2)
    df = it.to_dataframe(meta={"ID": str, "Features": str, "Version": int})
    return df, list(vocab)


def main(full=False):
    # Reset environment
    save_folder = TFIDF_WIKIPEDIA_INDEX_FOLDER if not full else TFIDF_FULL_INDEX_FOLDER
    shutil.rmtree(save_folder, ignore_errors=True)
    os.mkdir(save_folder)

    df, vocabulary = compute_bag_of_words(full)
    count_vect = CountVectorizer(
        analyzer=identity, vocabulary={k: i for i, k in enumerate(vocabulary)}
    )
    features = count_vect.fit_transform(df["Features"].to_bag())
    idf = (
        array.log((features.shape[0] + 1) / (features.sum(axis=1, keepdims=True) + 1))
        + 1
    )
    tf_idf = features * idf

    f = h5py.File("myfile.hdf5")
    d = f.require_dataset("/data", shape=tf_idf.shape, dtype=tf_idf.dtype)
    array.store(tf_idf, d)
    exit(0)

    tf_idf
    tfidf = TfidfVectorizer(analyzer=identity, norm="l2")

    print("Building TF-IDF vectors...")
    features_sparses = tfidf.fit_transform(ent_vectors)
    print("TF-IDF Matrix Shape : %s" % str(features_sparses.shape))
    dump(tfidf, osp.join(save_folder, "tfidf_vectorizer.json"))

    features_sparses_old, features_sparses_new = (
        features_sparses[: version_count[0]],
        features_sparses[version_count[0] :],
    )
    entities_old, entities_new = (
        list_seen_entities[: version_count[0]],
        list_seen_entities[version_count[0] :],
    )
    save_list(entities_old, osp.join(save_folder, "entities_old.txt"))
    save_list(entities_new, osp.join(save_folder, "entities_new.txt"))
    scipy.sparse.save_npz(
        osp.join(save_folder, "features_sparses_old.npz"), features_sparses_old
    )
    scipy.sparse.save_npz(
        osp.join(save_folder, "features_sparses_new.npz"), features_sparses_new
    )

    print("Index creation...")
    # Initialize NMSLib index
    index = nmslib.init(
        method="hnsw",
        space="cosinesimil_sparse",
        data_type=nmslib.DataType.SPARSE_VECTOR,
    )

    # Add data points to the index
    index.addDataPointBatch(features_sparses_old)

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
    args = parser.parse_args()
    main(full=args.full)
