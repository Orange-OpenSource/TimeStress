# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from __future__ import annotations
from abc import abstractmethod
import base64
from contextlib import contextmanager
import datetime

import inspect
import io
import json
import pickle
from typing import Any, Callable, Hashable, Iterable
from enum import Enum
from functools import total_ordering, wraps
from abc import ABCMeta
import os
import os.path as osp

import bson
import numpy as np
import torch
from hashlib import sha256
from sortedcontainers import SortedDict
from xxhash import xxh128

from .globals import STORAGE_FOLDER
from .general import (
    concat,
    dotdict,
    dump_json,
    get_class,
    load_json,
)
from typing import TypeVar, Type

import dataclasses


def meta_bypass_overridence(function_name: str, add_abc=False) -> type:
    class BypassOverridence(type):
        base_cls = None

        def __new__(mcs, name, bases, namespace):
            cls = super().__new__(mcs, name, bases, namespace)
            if BypassOverridence.base_cls is None:
                BypassOverridence.base_cls = cls
            # If 'f' is not defined in the namespace, use base_cls's 'f'
            if function_name not in namespace:
                setattr(
                    cls,
                    function_name,
                    getattr(BypassOverridence.base_cls, function_name),
                )
            return cls

    if add_abc:

        class Combined(ABCMeta, BypassOverridence):
            pass

        ret = Combined
    else:
        ret = BypassOverridence

    return ret


class ParentTracking(type):
    def __new__(metacls, name, bases, namespace, **kwargs):
        # Create the new class
        new_class = super().__new__(metacls, name, bases, namespace, **kwargs)
        # Set the _parents attribute to the bases that are created by this metaclass
        new_class._tracking = True
        new_class._parents = []
        new_class._childs = []
        for base in bases:
            try:
                if getattr(base, "_tracking"):
                    new_class._parents.append(base)
                    base._childs.append(new_class)
            except AttributeError:
                pass
        return new_class


# class Building(metaclass=meta_bypass_overridence('get_buildings', add_abc=True)):
class Blueprint(metaclass=ABCMeta):
    """An object that can be built and saved in the hard drive"""

    def build(self, force=False, confirm=True) -> None:
        """Build the object and save it to the hard drive"""
        if not force and self.built():
            raise Exception("This building object already exists!")
        if confirm:
            while True:
                ans = input(
                    "This process will possibly take a long time to finish. Are you sure you want to continue? [Y/n]"
                )
                ans = ans.lower().strip()
                if ans in ("y", "yes") or ans.strip() == "":
                    break
                elif ans in ("n", "no"):
                    raise Exception("Build process aborted!")
        if force:
            self._destroy()
        self._build()

    @abstractmethod
    def _build(self) -> None:
        pass

    @abstractmethod
    def _destroy(self) -> None:
        pass

    @abstractmethod
    def built(self) -> bool:
        """Was this object built ?

        Returns:
            bool
        """
        pass

    def destroy(self, force=False, confirm=True) -> None:
        """Delete this object from the hard drive if it exists"""
        if not force and not self.built():
            raise Exception("This building object does not exist!")
        if confirm:
            while True:
                ans = input(
                    "You are about to delete this object : %s. Are you sure you want to continue? [Y/n]"
                    % self
                )
                ans = ans.lower().strip()
                if ans in ("y", "yes") or ans.strip() == "":
                    break
                elif ans in ("n", "no"):
                    raise Exception("Destroy process aborted!")

        self._destroy()

    def load(self) -> None:
        """Loading procedure to make the building usable"""
        pass

    def build_and_load(self) -> None:
        """Build if this building was not already built and then load it"""
        if not self.built():
            self.build(confirm=False)
        self.load()

    @classmethod
    def get_buildings(cls) -> Iterable[Blueprint]:
        """Get the list of objects from the current class that were built (i.e. present in the hard drive)"""
        # if cls is Building:
        #     return
        # else:
        #     for b in super(cls, cls).get_buildings():
        #         if isinstance(b, cls):
        #             yield b
        raise NotImplementedError

    @classmethod
    def _building_bases(cls) -> Iterable[type]:
        yield cls
        for base in cls._parents:
            yield from base._building_bases()

    @classmethod
    def _building_childs(cls) -> Iterable[type]:
        yield cls
        for base in cls._childs:
            yield from base._building_childs()

    def storage_used(self) -> int:
        """Get the hard drive storage used by this object in bytes if it is already built (i.e. present in the hard drive)

        Returns:
            int: Size in bytes
        """
        raise NotImplementedError


SameType = TypeVar("SameType", bound="JSONable")
JSON_ID_INDICATOR = "ee1421cd"
JSONABLE_TYPES = (np.ndarray, torch.Tensor, torch.dtype)


def replace_json_buildings(data, replace_with="id") -> tuple[dict, list[JSONable]]:
    to_build = []

    def _replace_json_buildings(data):
        """
        Recursively walk through the data and replace JSONBuilding objects with either their json representation or an identifier
        """
        if isinstance(data, dict):
            return {k: _replace_json_buildings(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [_replace_json_buildings(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(_replace_json_buildings(item) for item in data)
        elif isinstance(data, JSONable):
            if replace_with == "json":
                return data._to_full_json(single_json=True, include_id=False, include_time=False)[0]
            else:
                to_build.append(data)
                return JSON_ID_INDICATOR + ":" + data._hash()
        elif issubclass((t := type(data)), JSONABLE_TYPES):
            d, _ = cls2saveablecls(t)(data)._to_full_json(
                single_json=True, include_id=False, include_time=False
            )
            d["resolve"] = True
            return d
        else:
            return data

    return _replace_json_buildings(data), to_build


class SaveableNotFoundError(Exception):
    pass


# TODO: Implement recursive building


class JSONable:
    SAVE_FOLDER = osp.join(STORAGE_FOLDER, "json_buildings")

    def __init__(self) -> None:
        self._save_in_progress = False

    def _to_json(self) -> dict | list:
        # Special case for dataclasses
        if dataclasses.is_dataclass(self):
            d = {}
            for attr in self.__annotations__.keys():
                value = getattr(self, attr)
                d[attr] = value
            return d

        raise NotImplementedError(
            "Please implement _to_json for the class %s" % self.__class__
        )

    def infos(self) -> dict:
        return dict(cls=self.class_id())

    @classmethod
    def _from_json(cls: Type[SameType], d: dict | list) -> SameType:
        if issubclass(cls, MongoableEnum):
            return cls(d)
        return cls(**d)

    def _identifier(self) -> Any:
        return self._to_json()

    @classmethod
    def _hash_class(cls, id: Any, ret='hex', include_class_id=True) -> str | int:
        id, _ = replace_json_buildings(id)
        json_ = json.dumps(
            (id, cls.class_id()) if include_class_id else id,
            default=None,
            ensure_ascii=False,
            sort_keys=True,
            indent=None,
            separators=(',', ':'),
        )

        h = xxh128(json_)
        if ret == "hex":
            return h.hexdigest()
        elif ret == 'int':
            return h.intdigest()
        

    def _hash(self, ret='hex', include_class_id=True) -> str | int:
        cls = self.__class__
        return cls._hash_class(self._identifier(), ret, include_class_id)

    def _to_full_json(
        self, single_json=True, include_id=True, include_time=True
    ) -> tuple[list | dict, list[JSONable]]:
        d = self._to_json()

        d, to_build = replace_json_buildings(d, "json" if single_json else "id")

        cls = self.__class__
        cls_id = cls.class_id()
        full_json = {"class": cls_id, "content": d, "_id": ""}
        if include_id:
            full_json["_id"] = self._hash()
        if include_time:
            full_json["time"] = str(datetime.datetime.now())
        return full_json, to_build

    def save(self, single_json=True) -> None:
        self._save_in_progress = True
        to_save, to_build = self._to_full_json(single_json)
        for b in to_build:
            if not b._save_in_progress and not b.saved():
                b.save()
        self.__class__._save_full_json(to_save)
        self._save_in_progress = False

    @classmethod
    def resolve_json_building_ids(cls, data):
        """
        Recursively walk through the data and replace JSONBuilding Hash IDs with their object
        """
        if isinstance(data, dict):
            if isinstance(id_ := data.get("_id"), str):
                return cls._from_full_json(data)
            return {k: cls.resolve_json_building_ids(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.resolve_json_building_ids(item) for item in data]
        elif isinstance(data, str) and data.startswith(JSON_ID_INDICATOR + ":"):
            id_ = data[len(JSON_ID_INDICATOR) + 1 :]
            obj = cls._from_id_hash(id_)
            return obj
        else:
            return data

    @classmethod
    def _save_full_json(cls, full_json: dict) -> None:
        os.makedirs(JSONable.SAVE_FOLDER, exist_ok=True)
        dump_json(osp.join(JSONable.SAVE_FOLDER, full_json["_id"] + ".json"), full_json)

    @classmethod
    def _from_json_with_resolve(cls: Type[SameType], d: dict | list) -> SameType:
        d = cls.resolve_json_building_ids(d)
        return cls._from_json(d)

    @classmethod
    def _from_full_json(cls: Type[SameType], d_full: dict | list) -> SameType:
        obj_cls = get_class(d_full["class"])
        resolve = d_full.get("resolve", False)
        res = obj_cls._from_json_with_resolve(d_full["content"])
        if resolve:
            return res.get_value()
        return res

    @classmethod
    def _from_path(
        cls: Type[SameType], path: str | None, ignore_not_found=False
    ) -> SameType:
        if path is not None:
            json_dict = load_json(path)
            return cls._from_full_json(json_dict)
        if not ignore_not_found:
            cls_id = cls.class_id()
            raise SaveableNotFoundError(
                "Object instance of %s with path=%s could not be found!"
                % (cls_id, path)
            )
        return None

    @classmethod
    def from_id(
        cls: Type[SameType], id: Hashable, many=False, ignore_not_found=False
    ) -> SameType | list[SameType]:
        if many:
            res = []
            for id_ in id:
                path = cls._get_path(id=id_)
                res.append(cls._from_path(path))
            return res
        path = cls._get_path(id=id)
        return cls._from_path(path, ignore_not_found)

    @classmethod
    def class_id(cls) -> str:
        return cls.__module__ + "." + cls.__qualname__

    @classmethod
    def _from_id_hash(cls: Type[SameType], id_hash: str) -> SameType:
        path = cls._get_path(id_hash=id_hash)
        return cls._from_path(path)

    @classmethod
    def _get_path(cls: Type[SameType], id: Hashable = None, id_hash: str = None) -> str:
        if id is not None:
            id_ = cls._hash_class(id)
        else:
            id_ = id_hash
        path = osp.join(STORAGE_FOLDER, JSONable.SAVE_FOLDER, id_ + ".json")
        if osp.exists(path):
            return path
        return None

    def delete(self) -> None:
        cls = self.__class__
        cls.delete_from_id(self._identifier())

    @classmethod
    def delete_from_id(cls, id) -> None:
        path = cls._get_path(id)
        if path is None:
            return
        try:
            os.remove(path)
        except OSError:
            pass

    def saved(self) -> bool:
        return self.__class__._get_path(self._identifier()) is not None

    def size_in_bytes(self) -> int:
        path = self._get_path(self._identifier())
        if path is None:
            raise SaveableNotFoundError(
                "This JSON building was not found on the hard drive!"
            )
        return osp.getsize(path)


class Mongoable(JSONable):
    # Specify custom MongoDB URL in the next line if you want
    # MONGO_COLLECTION = (
    #     get_mongodb_client(mongodb_url=None)
    #     .get_database("buildings")
    #     .get_collection("mongobuildings")
    # )
    # MONGO_COLLECTION.create_index("class")

    @classmethod
    def _save_full_json(cls, full_json: dict) -> None:
        cls.MONGO_COLLECTION.replace_one(
            {"_id": full_json["_id"]}, full_json, upsert=True
        )

    @classmethod
    def from_id(
        cls: Type[SameType],
        id: Hashable | Iterable[Hashable],
        many=False,
        ignore_not_found=False,
    ) -> SameType | list[SameType]:
        if many:
            res = cls._from_id_hash(
                [cls._hash_class(id_) for id_ in id], ignore_not_found
            )
            return res
        return cls._from_id_hash(cls._hash_class(id), ignore_not_found)

    @classmethod
    def delete_from_id(cls, id) -> None:
        cls.MONGO_COLLECTION.delete_one({"_id": cls._hash_class(id)})

    @classmethod
    def _from_id_hash(
        cls: Type[SameType], id_hash: str | list[str], ignore_not_found=False
    ) -> SameType | list[SameType]:
        if isinstance(id_hash, list):
            documents = {
                x["_id"]: x
                for x in cls.MONGO_COLLECTION.find({"_id": {"$in": id_hash}})
            }
            results = []
            for id in id_hash:
                document = documents.get(id)
                if document is None:
                    if not ignore_not_found:
                        raise SaveableNotFoundError
                    results.append(None)
                    continue
                results.append(Mongoable._from_full_json(document))
            return results

        full_json = cls.MONGO_COLLECTION.find_one({"_id": id_hash})
        if full_json is None:
            if ignore_not_found:
                return None
            raise SaveableNotFoundError
        return cls._from_full_json(full_json)

    def saved(self) -> bool:
        return self.MONGO_COLLECTION.find_one({"_id": self._hash()}) is not None

    def size_in_bytes(self) -> int:
        a = self.MONGO_COLLECTION.find_one({"_id": self._hash()})
        return len(bson.encode(a))


class MongoableEnum(Mongoable, Enum):
    def __init__(self, *args, **kwargs) -> None:
        Mongoable.__init__(self)
        Enum.__init__(self)

    def _to_json(self) -> dict | list:
        return self.value

    @classmethod
    def _from_json(cls: type[SameType], d: dict | list) -> SameType:
        return cls(d)


class Precision(MongoableEnum):
    FLOAT32 = 0
    FLOAT16 = 1
    BFLOAT16 = 2

    def to_torch_dtype(self) -> torch.dtype:
        return PREC2TORCH[self]

    @staticmethod
    def to_precision(dtype: torch.dtype) -> Precision:
        return TORCH2PREC[dtype]


PREC2TORCH = {
    Precision.FLOAT32: torch.float32,
    Precision.BFLOAT16: torch.bfloat16,
    Precision.FLOAT16: torch.float16,
}
TORCH2PREC = {v: k for k, v in PREC2TORCH.items()}


@total_ordering
class TimeUnit(MongoableEnum):
    DAY = 0
    WEEK = 1
    MONTH = 2
    YEAR = 3

    def __lt__(self, other):
        if isinstance(other, TimeUnit):
            return self.value < other.value
        raise ValueError


@dataclasses.dataclass
class Cache(Mongoable):
    # MONGO_COLLECTION = (
    #     get_mongodb_client(mongodb_url=None)
    #     .get_database("buildings")
    #     .get_collection("cache")
    # )
    # MONGO_COLLECTION.create_index("class")
    data: Any
    identifier: Any
    function: str

    def _to_json(self) -> dict | list:
        d = super()._to_json()
        d.pop("identifier")
        return d

    def _identifier(self) -> Any:
        return {"id": self.identifier, "func": self.function}

    @classmethod
    def _from_json(cls: type[SameType], d: dict | list) -> SameType:
        d["identifier"] = None
        return cls(**d)


class JSONableData(Mongoable):
    def get_value(self):
        raise NotImplementedError


class Tensor(JSONableData):
    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()
        self.tensor = tensor

    def _to_json(self) -> dict | list:
        buffer = io.BytesIO()
        torch.save(self.tensor, buffer)
        buffer.seek(0)
        return {"tensor": base64.b64encode(buffer.read())}

    @classmethod
    def _from_json(cls: type[SameType], d: dict | list) -> SameType:
        d["tensor"] = torch.load(io.BytesIO(base64.b64decode(d["tensor"])))
        return cls(**d)

    def get_value(self):
        return self.tensor


class TorchDType(JSONableData):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def _to_json(self) -> dict | list:
        return {"dtype": str(self.dtype).split(".")[-1]}

    @classmethod
    def _from_json(cls: type[SameType], d: dict | list) -> SameType:
        return cls(getattr(torch, d["dtype"]))

    def get_value(self):
        return self.dtype


class NumpyArray(JSONableData):
    def __init__(self, array: np.ndarray) -> None:
        super().__init__()
        self.array = array

    def _to_json(self) -> dict | list:
        b = self.array.tobytes()
        return {
            "bytes": base64.b64encode(b),
            "shape": self.array.shape,
            "dtype": self.array.dtype.name,
        }

    @classmethod
    def _from_json(cls: type[SameType], d: dict | list) -> SameType:
        array = np.frombuffer(base64.b64decode(d["bytes"]), d["dtype"]).reshape(
            d["shape"]
        )
        return cls(array)

    def get_value(self):
        return self.array


def cls2saveablecls(cls: type) -> Type[JSONable]:
    if issubclass(cls, torch.Tensor):
        return Tensor
    elif issubclass(cls, np.ndarray):
        return NumpyArray
    elif issubclass(cls, torch.dtype):
        return TorchDType
    raise Exception("Saveable class not found for class %s" % cls.__name__)


_cache_function_activated = dotdict()


def is_cache_activated(func: Callable) -> bool:
    return _cache_function_activated.get(func, False)


def delete_cache() -> None:
    Cache.MONGO_COLLECTION.delete_many({"class": Cache.class_id()})


def get_cache_by_class(cls: type[JSONable]) -> list[Cache]:
    l = [
        Cache._from_full_json(x).get_value()
        for x in Cache.MONGO_COLLECTION.find({"class": Cache.class_id()})
    ]


@contextmanager
def activate_cache(func: Callable):
    _cache_function_activated[func] = True
    try:
        yield
    finally:
        _cache_function_activated.pop(func)


def cache_return(
    func: Callable,
    args_identifier: tuple[str],
    loop_args: list[str] = [],
    additional_kwargs: dict = {},
    postprocess: Callable = lambda x: x,
    verbose=False,
) -> Callable:
    @wraps(func)
    def new_func(*args, **kwargs):
        if is_cache_activated(func):
            bound_arguments = inspect.signature(func).bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            vars = bound_arguments.arguments
            identifier = {k: vars.pop(k) for k in args_identifier}
            identifier_short = SortedDict(
                {
                    k: (
                        v._to_full_json(include_idtime=False)
                        if isinstance(v, JSONable)
                        else v
                    )
                    for k, v in identifier.items()
                }
            )
            loops = [vars.pop(arg) for arg in loop_args]

            tmp = identifier.copy()
            # for arg in loop_args:
            #     tmp.pop(arg)

            identifier_short = [
                identifier_short.copy()
                for _ in range(len(loops[0]) if len(loops) else 1)
            ]
            for j, arg in enumerate(loop_args):
                for i, id_ in enumerate(identifier_short):
                    id_[arg] = loops[j][i]
            cache = list(
                map(
                    postprocess,
                    Cache.from_id(
                        (
                            {"id": x, "func": func.__qualname__}
                            for x in identifier_short
                        ),
                        many=True,
                        ignore_not_found=True,
                    ),
                )
            )
            results = [None if x is None else x.data for x in cache]
            if verbose:
                for c in results:
                    if c is None:
                        continue
                    print("cache_return::%s retrieved!" % c.infos())
            results = [
                (
                    None
                    if x is None
                    else x.get_value() if isinstance(x, JSONableData) else x
                )
                for x in results
            ]
            missing = [i for i in range(len(results)) if results[i] is None]
            for arg, loop in zip(loop_args, loops):
                tmp[arg] = concat([loop[i] for i in missing])
            if len(missing):
                missing_res = func(**tmp, **additional_kwargs, **vars)
                if len(loop_args) == 0:
                    missing_res = [missing_res]
            else:
                missing_res = []

            for i, res in zip(missing, missing_res):
                results[i] = res
                if type(res) in JSONABLE_TYPES:
                    res = cls2saveablecls(res.__class__)(res)
                cache = Cache(res, identifier_short[i], func.__qualname__)
                cache.save(single_json=True)
                if verbose:
                    print("cache_return::%s saved!" % cache.infos())
            return concat(results) if len(loop_args) else results[0]
        else:
            return func(*args, **kwargs)

    return new_func
