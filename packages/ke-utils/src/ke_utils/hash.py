# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models
"""
Generate stable hashes for Python data objects.
Contains no business logic.

The hashes should be stable across interpreter implementations and versions.

Supports dataclass instances, datetimes, and JSON-serializable objects.

Empty dataclass fields are ignored, to allow adding new fields without
the hash changing. Empty means one of: None, '', (), [], or {}.

The dataclass type is ignored: two instances of different types
will have the same hash if they have the same attribute/value pairs.

CREDITS: This code is heavily based lemon24's provided under BSD-3-Clause License in:
https://github.com/lemon24/reader/blob/1efcd38c78f70dcc4e0d279e0fa2a0276749111e/src/reader/_hash_utils.py 
"""
import dataclasses
import datetime
import hashlib
import json
from collections.abc import Collection
from typing import Any
from typing import Dict
from xxhash import xxh128


# Implemented for https://github.com/lemon24/reader/issues/179


# The first byte of the hash contains its version,
# to allow upgrading the implementation without changing existing hashes.
# (In practice, it's likely we'll just let the hash change and update
# the affected objects again; nevertheless, it's good to have the option.)
#
# A previous version recommended using a check_hash(thing, hash) -> bool
# function instead of direct equality checking; it was removed because
# it did not allow objects to cache the hash.

# _VERSION = 0

def consistent_hash(thing: object) -> int:
    # prefix = _VERSION.to_bytes(1, 'big')
    digest = xxh128(_json_dumps(thing).encode('utf-8')).intdigest()
    return digest

def _json_dumps(thing: object) -> str:
    to_serialize = getattr(thing, "_identifier", thing)
    if to_serialize is not thing:
        to_serialize = to_serialize()
    return json.dumps(
        to_serialize,
        default=None,
        ensure_ascii=False,
        sort_keys=True,
        indent=None,
        separators=(',', ':'),
    )
