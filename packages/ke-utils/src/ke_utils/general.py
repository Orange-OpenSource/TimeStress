# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
import datetime
from functools import wraps
import hashlib
from itertools import groupby
import itertools
import json
import math
import os
import pickle
import random
import re
import sys
import time
from typing import Callable, ContextManager, Iterable, Union
import bson
import os.path as osp

import numpy as np
import pandas as pd
from pymongo import MongoClient
import torch
from scipy.stats import pearsonr
import hashlib

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry
import zipfile
import os
from io import BytesIO


class PrintableException(Exception):
    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return self.__class__.__name__ + ": " + self.args[0]


def all_equal(iterable: Iterable) -> bool:
    """Checks if all elements of iterable are equal.

    Args:
        iterable (Iterable)

    Returns:
        bool
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def str2number(number_string: str) -> Union[int, float]:
    """Convert a string representing a number to either an integer or a float (starting with integer conversion).

    Args:
        number_string (str): The number as a string

    Raises:
        ValueError: when number_string is not a number

    Returns:
        Union[int, float]: Number
    """
    try:
        return int(number_string)
    except ValueError:
        pass

    try:
        return float(number_string)
    except ValueError:
        raise ValueError('"%s" is neither an integer nor a float!' % number_string)


def get_print_verbose(verbose=False):
    def nothing(*args, **kwargs) -> None:
        pass

    if verbose:
        return print
    return nothing


DATE_PAT = re.compile(r"^[0-9]{8}$")


def date_is_correct(date: str) -> bool:
    """Check if a date in this format YYYYMMDD is correct and exists

    Args:
        date (str): Text representing a date

    Returns:
        bool
    """
    if date is None:
        return False
    m = re.match(DATE_PAT, date)
    if not m:
        return False

    try:
        datetime.datetime.strptime(date, "%Y%m%d")
        return True
    except ValueError:
        raise ValueError(
            "Incorrect data format, should be YYYYMMDD. Date found : %s" % date
        )


def mongodb_dump(collections, conn, db_name, path):
    """
    MongoDB Dump

    :param collections: Database collections name
    :param conn: MongoDB client connection
    :param db_name: Database name
    :param path:
    :return:

    >>> DB_BACKUP_DIR = '/path/backups/'
    >>> conn = MongoClient("mongodb://admin:admin@127.0.0.1:27017", authSource="admin")
    >>> db_name = 'my_db'
    >>> collections = ['collection_name', 'collection_name1', 'collection_name2']
    >>> dump(collections, conn, db_name, DB_BACKUP_DIR)
    """

    db = conn[db_name]
    for coll in collections:
        with open(os.path.join(path, f"{coll}.bson"), "wb+") as f:
            for doc in db[coll].find():
                f.write(bson.BSON.encode(doc))


def mongodb_exists(path, coll_name):
    return os.path.exists(os.path.join(path, coll_name + ".bson"))


def mongodb_restore(path, conn, db_name, coll_name):
    """
    MongoDB Restore

    :param path: Database dumped path
    :param conn: MongoDB client connection
    :param db_name: Database name
    :param coll: Collection name
    :return:

    >>> DB_BACKUP_DIR = '/path/backups/'
    >>> conn = MongoClient("mongodb://admin:admin@127.0.0.1:27017", authSource="admin")
    >>> db_name = 'my_db'
    >>> restore(DB_BACKUP_DIR, conn, db_name)

    """
    found = False
    db = conn[db_name]
    for coll in os.listdir(path):
        if coll.endswith(".bson") and coll.startswith(coll_name):
            with open(os.path.join(path, coll), "rb") as f:
                db[coll.split(".")[0]].insert_many(bson.decode_all(f.read()))
            found = True
    if not found:
        raise FileNotFoundError


def dump_json(filepath: str, obj):
    if filepath.endswith(".json"):
        data = json.dumps(obj)
    elif filepath.endswith(".jsonl"):
        data = "".join(json.dumps(x) + "\n" for x in obj)
    with open(filepath, "w") as f:
        f.write(data)


def load_json(filepath: str) -> Union[dict, list]:
    with open(filepath) as f:
        if filepath.endswith(".json"):
            content = json.load(f)
        elif filepath.endswith(".jsonl"):
            content = [json.loads(x) for x in f]
    return content


def load_many_pickle(filepath: str) -> list:
    objects = []
    with open(filepath, 'rb') as file:
        while True:
            try:
                obj = pickle.load(file)
                objects.append(obj)
            except EOFError:
                break
    
    return objects


def dump_one_pickle(filepath : str, obj)-> None:
    with open(filepath, 'ab') as file:
        pickle.dump(obj, file)


def read_list(path: str) -> list[str]:
    with open(path, "r") as f:
        l = [x.strip() for x in f]
    return l


def save_list(l: list[str], path: str):
    with open(path, "w") as f:
        for x in l:
            f.write(x + "\n")


def inf_gen(value=None):
    """Infinite generatore of "None" values"""
    while True:
        yield value


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_class(path: str) -> type:
    """Get class specified in the path , e.g. "os.path"

    The module is imported if needed

    Args:
        path (str): Path to the module just like you were doing a regular import

    Raises:
        NameError: If the module or the class inside the module could not be found

    Returns:
        type: The class
    """
    module_name, class_name = osp.splitext(path)
    class_name = class_name.lstrip(".")
    module = sys.modules.get(module_name)
    if module is None:
        try:
            module = __import__(module_name)
        except ImportError:
            raise NameError("The module %s could not be imported" % module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise NameError("The class %s could not be imported" % path)
    return cls


MONGODB_CLIENTS = {}


def get_mongodb_client(mongodb_url: str = None) -> MongoClient:
    """Get MongoDB client given the MongoDB URL.
    This functions caches the mongodb clients to avoid creating a new connection each time a Wikidata object is instanciated for example.
    This is the recommended way of instantiating a MongoDB client.

    Args:
        mongodb_url (str): The URL that points to a Mongo database

    Returns:
        MongoClient
    """
    if mongodb_url is None:
        mongodb_url = os.getenv("MONGO_URL", "mongodb://127.0.0.1/wiki")
    client = MONGODB_CLIENTS.get(mongodb_url, None)
    if client is not None:
        return client

    client = MongoClient(mongodb_url)
    MONGODB_CLIENTS[mongodb_url] = client
    return client


def create_switch_contextmanager(cls: type, var_name: str) -> ContextManager:
    @contextmanager
    def manager():
        """Enables caching of credibility results by LMs on text"""
        # Set the class variable to True to indicate we're inside the context
        cp = getattr(cls, var_name)
        setattr(cls, var_name, True)
        try:
            yield
        finally:
            # Recover the class variable when exiting the context
            setattr(cls, var_name, cp)

    return manager


def is_subseq(subseq: list, seq: list) -> bool:
    for i in range(len(seq) - len(subseq) + 1):
        for j in range(len(subseq)):
            if subseq[j] != seq[i + j]:
                break
        else:
            return True
    return False


def concat(l: list[list | np.ndarray | torch.Tensor]):
    if len(l) == 0:
        return []
    first = l[0]
    if isinstance(first, np.ndarray):
        return np.concatenate(l, axis=0)
    elif isinstance(first, torch.Tensor):
        # NOTE: EXPERIMENTAL
        return torch.stack(l, axis=0)
    elif isinstance(first, list):
        return [y for x in l for y in x]
    else:
        return l


def pearson_r(x, y) -> tuple[float, float]:
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    x_isnan = np.isnan(x) | np.isinf(x)
    y_isnan = np.isnan(y) | np.isinf(y)
    mask = ~x_isnan & ~y_isnan
    return pearsonr(x[mask], y[mask])


def uniquifier(seq: Iterable, key=None, return_index=False) -> list:
    """Remove duplicates while keeping order (after first one is filtered)

    Source : https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order

    Args:
        seq (Iterable): Iterable

    Returns:
        list: Iterable without duplicates
    """
    if key is None:
        key = lambda x: x
    seen = set()
    seen_add = seen.add
    out = [
        (i, x) for i, x in enumerate(seq) if not (key(x) in seen or seen_add(key(x)))
    ]
    if len(out):
        index, uniques = list(zip(*out))
    else:
        uniques, index = [], []
    if return_index:
        return list(uniques), list(index)
    return list(uniques)


def hash_dict(d: dict):
    return commutative_hash(pickle.dumps(x) for x in d.items())


def commutative_hash(*args):
    # Initialize an integer to store the combined hash
    combined_hash = 0

    # Hash each argument and combine the result using XOR
    for arg in args:
        # Create a new hash object for each argument
        hash_obj = hashlib.sha256()
        hash_obj.update(arg)

        # Convert the hash digest to an integer and XOR it with the combined hash
        current_hash = int(hash_obj.hexdigest(), 16)
        combined_hash ^= current_hash

    # Convert the combined integer hash back to hexadecimal
    combined_hash_hex = format(combined_hash, "x").zfill(64)
    return combined_hash_hex


def confidence_interval_prec(prec: int = 1):
    def f(frame: pd.DataFrame):
        m = frame.mean(axis=0, skipna=True)
        s = frame.std(axis=0, skipna=True)
        c = len(frame) - frame.isna().sum(axis=0)
        d = 1.96 * s / math.sqrt(c)
        return f"%.{prec}f ± %.{prec}f" % (m, d)

    return f


class DownloadError(Exception):
    """Custom exception for download errors."""

    pass


# Function to download and unzip a file from a URL with retry logic
def download_and_unzip(url, extract_to, retries=3, backoff_factor=0.3):
    def is_zip_file(response, url):
        """
        Determine if the response is a zip file either by the Content-Disposition header
        or by the file extension in the URL.
        """
        content_disposition = response.headers.get("Content-Disposition", "")
        if "zip" in content_disposition:
            return True
        if url.lower().endswith(".zip"):
            return True
        return False

    # Create a session object
    session = requests.Session()
    # Define the retry parameters
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        # method_whitelist=["HEAD", "GET", "OPTIONS"]
    )
    # Mount it for both http and https usage
    session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

    try:
        # Send a HTTP request to the URL
        print(f"Downloading file from {url}")
        with session.get(url, stream=True) as response:
            # Raise an exception if the download failed
            response.raise_for_status()

            # Check if the response content is a zip file
            if is_zip_file(response, url):
                # Create a BytesIO object to hold the chunks of data
                zip_file_bytes = BytesIO()
                total_size = int(response.headers.get("content-length", 0))
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="Downloading"
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        # Filter out keep-alive new chunks
                        if chunk:
                            zip_file_bytes.write(chunk)
                            progress_bar.update(len(chunk))

                # Extract the zip file
                zip_file_bytes.seek(0)  # Move to the beginning of the BytesIO object
                with zipfile.ZipFile(zip_file_bytes, "r") as zip_ref:
                    # Extract the zip file to the specified directory
                    print(f"Extracting files to '{extract_to}' folder")
                    zip_ref.extractall(extract_to)
                    print("Extraction complete.")
            else:
                raise DownloadError("The URL does not contain a zip file.")
    except requests.exceptions.HTTPError as http_err:
        raise DownloadError(f"HTTP error occurred: {http_err}") from http_err
    except Exception as err:
        raise DownloadError(f"An error occurred: {err}") from err
    finally:
        # Close the session
        session.close()


class TimeItContextManager:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        print(f"{self.name}", end=" ")
        self.time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.execution_time = time.time() - self.time
        print(f": {self.execution_time} sec")


def sample_from_list_of_collections(list_of_lists, k):
    """Memory-and-compute-efficient sampling from a list of collections,
    i.e., without creating a big intermediate collection to sample from.
    This sampling is with replacement and is equivalent (if I'm not wrong) to uniform random sampling with replacement
    from the concatenation of all collections.
    """
    # Step 1: Calculate the total number of elements
    total_elements = sum(len(sublist) for sublist in list_of_lists)

    if k > total_elements:
        raise ValueError("k cannot be greater than the total number of elements")

    # Step 2: Randomly select k unique positions
    sampled_positions = random.sample(range(total_elements), k)
    sampled_positions.sort()  # Sorting helps in efficiently locating elements

    # Step 3: Locate the elements
    sampled_elements = []
    current_position = 0
    pos_index = 0

    for sublist in list_of_lists:
        sublist_length = len(sublist)

        while (
            pos_index < k
            and sampled_positions[pos_index] < current_position + sublist_length
        ):
            element_index = sampled_positions[pos_index] - current_position
            sampled_elements.append(sublist[element_index])
            pos_index += 1

        current_position += sublist_length

        if pos_index >= k:
            break

    return sampled_elements


def singleton(cls):
    """Singleton class decorator"""
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def singleton_by_args(cls):
    """Singleton by arguments decorator.
    It ensures the same instance is returned for the same __init__ arguments"""
    instances = {}

    @wraps(cls)
    def get_instance(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in instances:
            instances[key] = cls(*args, **kwargs)
        return instances[key]

    return get_instance


def topk_with_indices(arr: np.ndarray, k: int) -> tuple[np.ndarray]:
    """Find the top-k values in array and returns these values + the indices where these values appear"""
    # Avoid index issues
    k = min(arr.shape[0], k)
    # Get the indices of the top k elements
    indices = np.argpartition(arr, -k)[-k:]
    # Get the top k elements
    topk_elements = arr[indices]
    # Sort the top k elements and their indices
    sorted_indices = indices[np.argsort(topk_elements)[::-1]]
    topk_elements_sorted = arr[sorted_indices]
    return topk_elements_sorted, sorted_indices


def logsumexp_by_group(x, b):
    """
    Compute the logsumexp of elements in x indexed by counts in b in a numerically stable way.

    Args:
    x (torch.Tensor): Input tensor of size N.
    b (list or torch.Tensor): tensor of counts for each group.

    Returns:
    torch.Tensor: Resulting tensor of size equal to the length of b.
    """
    # Convert b to a tensor if it's a list

    # Generate the indices from the counts using repeat_interleave
    indices = torch.repeat_interleave(torch.arange(len(b)), b)

    # Number of unique indices in b
    num_classes = len(b)

    # Step 1: Compute the maximum value for each group
    max_vals = torch.zeros(num_classes, dtype=x.dtype).scatter_reduce(
        0, indices, x, reduce="amax"
    )

    # Step 2: Subtract the maximum value from each element in x
    x_stable = x - max_vals[indices]

    # Step 3: Compute the exponentials of the stabilized x
    x_exp = torch.exp(x_stable)

    # Step 4: Initialize a tensor to hold the sums of exponentials
    sum_exp = torch.zeros(num_classes, dtype=x.dtype).scatter_add(0, indices, x_exp)

    # Step 5: Compute the log of the summed exponentials and add the max values back
    logsumexp = torch.log(sum_exp) + max_vals

    return logsumexp


try:
    from openai import AzureOpenAI

    # Set your Azure OpenAI API key and other configurations here
    endpoint = os.getenv(
        "ENDPOINT_URL", "https://open-ai-dai3-nepal-fra.openai.azure.com/"
    )
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

    # Initialize Azure OpenAI client with key-based authentication
    if subscription_key is not None:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=subscription_key,
            api_version="2024-05-01-preview",
        )
    else:
        print(
            "To use OpenAI API you must provide AZURE_OPENAI_API_KEY as an environment variable"
        )
        clinet = None

except ImportError:
    print('"pip install openai" to use OpenAI ChatGPT API')


def get_chatgpt_response(
    prompt,
    model="gpt-35-turbo-16k-0613",
    max_retries=3,
    retry_delay=5,
    temperature=0,
    top_n=1.0,
):
    """
    Get a response from ChatGPT for a given prompt with error handling.

    Args:
    - prompt (str): The input prompt to send to ChatGPT.
    - model (str): The model to use (default is the deployment name).
    - max_retries (int): The maximum number of retries in case of failure (default is 3).
    - retry_delay (int): The delay between retries in seconds (default is 5).

    Returns:
    - str: The response from ChatGPT.
    """
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperature=temperature,  # No randomness
                top_p=top_n,  # No randomness
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")

        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    raise Exception("ERROR: Azure OpenAI API is not working as intended (maybe down?)")


def convert_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    return d


def ramer_douglas_peucker(points, epsilon):
    """
    Simplify a polyline using the Ramer-Douglas-Peucker algorithm.

    :param points: List of points (x, y) to simplify
    :param epsilon: Maximum distance from the original line to simplify
    :return: Simplified list of points
    """
    # Find the point with the maximum distance
    dmax = 0.0
    index = 0
    end = len(points)
    for i in range(1, end - 1):
        d = perpendicular_distance(points[i], points[0], points[end - 1])
        if d > dmax:
            index = i
            dmax = d

    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        # Recursive call
        rec_results1 = ramer_douglas_peucker(points[: index + 1], epsilon)
        rec_results2 = ramer_douglas_peucker(points[index:], epsilon)

        # Build the result list
        result = rec_results1[:-1] + rec_results2
    else:
        result = [points[0], points[end - 1]]

    return result


def perpendicular_distance(point, line_start, line_end):
    """
    Calculate the perpendicular distance from a point to a line.

    :param point: The point (x, y)
    :param line_start: The start point of the line (x, y)
    :param line_end: The end point of the line (x, y)
    :return: Perpendicular distance
    """
    if line_start == line_end:
        return np.linalg.norm(np.array(point) - np.array(line_start))

    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)
    t = max(0.0, min(1.0, t))
    nearest = line_vec * t
    dist = np.linalg.norm(nearest - point_vec)
    return dist

def search_str_in_tokens(text : str, tokens : list[str], subchar_tokens_count : list[int] = None, ignore_case=False) -> list[tuple[int,int]]:
    assert len(text) and len(tokens), "text and tokens must be non-empty"
    if subchar_tokens_count is None:
        subchar_tokens_count = [1]*len(tokens)
    subchar_tokens_count = np.array([0] + subchar_tokens_count)
    cum_sum = np.cumsum(subchar_tokens_count)
    tokens_length = [len(x) for x in tokens]
    pos2token = np.arange(len(tokens)).repeat(tokens_length)
    tokens_cat = ''.join(tokens)
    flags = re.IGNORECASE if ignore_case else 0
    spans = []
    for m in re.finditer(re.escape(text), tokens_cat, flags):
        s,e = m.start(), m.end()-1
        ts, te = pos2token[[s, e]]
        spans.append((cum_sum[ts].item(), cum_sum[te+1].item()))
    return spans


def rep_sample(df : pd.DataFrame, col, n, random_seed=421):
    if n > len(df):
        return df
    a = df.groupby(col, observed=True)[col].count().astype(float)
    b = (a*len(a))**-1
    b = b.to_frame()
    b.columns = ['p']
    b = b.reset_index()
    df2 = df.merge(b, on=col)
    df2.set_index(df.index, inplace=True)
    df2 = df2.sample(n, weights=df2['p'], random_state=random_seed)
    return df.loc[df2.index]

def instance_tracker_factory(key_function):
    """
    A factory function to create a class that tracks its instances.

    Args:
        key_function (callable): A function that takes an instance and returns a unique key for it.

    Returns:
        type: A dynamically created class with instance tracking.
    """
    _instances = {}
    class InstanceTracker:
        # Class-level dictionary to store all instances
        def __init__(self):
            # Use the key_function to determine the key for this instance
            key = key_function(self)
            _instances[key] = self
            self.DELETE = True

        def __del__(self):
            """Remove the instance from the _instances dictionary when it is deleted."""
            if not self.DELETE:
                return
            key = key_function(self)
            if key in _instances:
                del _instances[key]

        @classmethod
        def get_instances(cls):
            """Return all existing instances as a dictionary."""
            return _instances

        @classmethod
        def get_instance_by_key(cls, key):
            """Retrieve a specific instance by its key."""
            return _instances.get(key)

        @classmethod
        def clear_instances(cls):
            """Clear the dictionary of tracked instances."""
            _instances.clear()

        def register(self) -> None:
            _instances[key_function(self)] = self

    return InstanceTracker

def flatten(nested):
    flat_list = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            flat_list.extend(flatten(item))  # Recursively flatten
        else:
            flat_list.append(item)  # Add the element to the flat list
    return flat_list

def func2dict(func: Callable) -> Mapping:
    """Convert a function to a dictionary-like interface

    Args:
        func (Callable): Function

    Returns:
        Mapping: Dictionary-like version of func
    """
    class DictLike(Mapping):
        def __getitem__(self, *args, **kwargs):
            return func(*args, **kwargs)
        
        def __len__(self):
            pass

        def __iter__(self):
            pass
    
    return DictLike()


def select_index(array: list, index: list[int]) -> list:
    return [array[i] for i in index]