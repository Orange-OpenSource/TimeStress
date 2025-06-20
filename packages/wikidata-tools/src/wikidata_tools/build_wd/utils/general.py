# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from typing import Any, Iterable
import json
from functools import wraps
import tqdm
from multiprocessing import Process, Queue


def uniquifier(seq: Iterable) -> list:
    """Remove duplicates while keeping order (last one is filtered)

    Source : https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order

    Args:
        seq (Iterable): Iterable

    Returns:
        list: Iterable without duplicates
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def dump_json(filepath: str, obj):
    with open(filepath, "w") as f:
        json.dump(obj, f)


def load_json(filepath: str) -> Any:
    with open(filepath, "r") as f:
        content = json.load(f)
    return content


def subrel2str(subject: str, relation: str):
    return "%s_%s" % (subject, relation)


def neg_ent(entity: str) -> str:
    return "-" + entity


def run_in_process(gen_func):
    """Run the given generator in another process (useful to parallelization)

    Args:
        gen_func (function): A python generator

    Returns:
        Iterable
    """

    @wraps(gen_func)
    def wrapper(*args, **kwargs):
        queue = Queue(100)

        def target_func(queue: Queue, *args, **kwargs):
            gen = gen_func(*args, **kwargs)
            for item in gen:
                queue.put(item)
            queue.put(StopIteration)

        p = Process(target=target_func, args=(queue,) + args, kwargs=kwargs)
        p.start()

        while True:
            item = queue.get()
            if item is StopIteration:
                break
            yield item

        p.join()

    return wrapper
