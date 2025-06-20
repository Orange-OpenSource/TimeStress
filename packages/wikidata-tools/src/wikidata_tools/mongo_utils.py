# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

import os
import os.path as osp

from ke_utils.globals import STORAGE_FOLDER


def remove_from_progress_file(name: str, command: str) -> None:
    # Code specific to old code base
    try:
        success_file = osp.join(
            STORAGE_FOLDER, "progress", name.replace(" ", "_") + ".success"
        )
        with open(success_file) as f:
            commands = [x.strip() for x in f]
            try:
                commands.remove(command)
                with open(success_file, "w") as f:
                    f.write("\n".join(commands))
            except ValueError:
                pass
    except FileNotFoundError:
        pass


def handle_mongodb_url(mongodb_url):
    if mongodb_url is None:
        mongodb_url = os.getenv("MONGO_URL")
        if mongodb_url is None:
            mongodb_url = "mongodb://127.0.0.1"
    return mongodb_url
