# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from wikidata_tools.build_wd.config import STORAGE_FOLDER
import os.path as osp


EMBEDDINGS_FOLDER = osp.join(STORAGE_FOLDER, "gpt_encode_entity_labels")
TFIDF_WIKIPEDIA_INDEX_FOLDER = osp.join(STORAGE_FOLDER, "tfidf_index")
TFIDF_FULL_INDEX_FOLDER = osp.join(STORAGE_FOLDER, "tfidf_full_index")
