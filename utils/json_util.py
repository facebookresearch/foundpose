"""
Helpers for working with JSON.
"""

import collections.abc as abc
import dataclasses
import json
import re
from enum import Enum
import inspect
from typing import (
    Any,
    Dict,
    FrozenSet,
    get_type_hints,
    List,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

# type of json objects.
#
# The last two union elements should be `List["JsonValue"]` and
# `Dict[str,"JsonValue"]`, but none of the Python type checkers
# are able to handle recursive  types.

JsonValue = Union[None, str, bool, int, float, List[Any], Dict[str, Any]]

T = TypeVar("T")


# Return the json value resulting from merging b into a.
def merge(a: JsonValue, b: JsonValue) -> JsonValue:
    # dicts merge any common elements
    if isinstance(a, dict) and isinstance(b, dict):
        a = dict(a)
        for k in b:
            if k in a:
                a[k] = merge(a[k], b[k])
            else:
                a[k] = b[k]
        return a

    # anything else (including lists) just replaces the old value
    return b


def merge_at(obj: JsonValue, path: str, rhs: JsonValue) -> JsonValue:
    """
    Return the result of merging `rhs` into `obj` at path `path`. Obj
    itself is not modified.

    Arguments:
    obj -- a json object
    path -- dotted field specifier, e.g. 'a.b.c'; empty to replace whole obj
    rhs -- new json value to merge at path

    >>> merge_at({'x': 1, 'y': {'a': 2, 'b': 3}}, 'x.b', 4)
    {'x': 1, 'y': {'a': 2, 'b': 4}}

    >>> merge_at({'x': 1, 'y': 2}, '', {'z': 3})
    {'x': 1, 'y': 2, 'z': 3}
    """
    keys = [k for k in path.split(".") if k]
    while keys:
        key = keys.pop()
        rhs = {key: rhs}
    return merge(obj, rhs)


# Parse s as a json value, but allow non-json strings to be given
# without quotes for convenience on command line.
def parse_json_or_str(s: str) -> JsonValue:
    s = s.strip()
    if re.match("""null$|true$|false$|^[0-9."'[{-]""", s):
        return json.loads(s)
    else:
        return s


# merge an assignment like 'a.b.c={some json}' into a json object
# if the rhs looks
def merge_from_arg(obj: JsonValue, arg: str):
    # look for '=' before any quotes
    if re.match("[^\"']+=", arg):
        path, rhs = arg.split("=", 1)
    else:
        path = ""
        rhs = arg
    rhs_obj = parse_json_or_str(rhs)
    return merge_at(obj, path, rhs_obj)


def from_any(x: Any) -> JsonValue:
    """
    Convert a Python object to json. This does not require type hints,
    since we have the actual object available.
    """

    # primitives
    if type(x) in (str, bool, int, float, type(None)):
        return x

    if isinstance(x, Enum):
        return from_any(x.value)

    if isinstance(x, abc.Mapping):
        return {str_from_any(k): from_any(v) for (k, v) in x.items()}

    if dataclasses.is_dataclass(x):
        return from_any(dataclasses.asdict(x))

    # NamedTuple
    if hasattr(x, "_asdict"):
        return from_any(x._asdict())

    if isinstance(x, (abc.Sequence, abc.Set)):
        return [from_any(x) for x in x]

    # NumPy floating-point types
    if isinstance(x, np.floating):
        return float(x)

    # NumPy integer types
    if isinstance(x, np.integer):
        return int(x)

    # Numpy array
    if isinstance(x, np.ndarray):
        return x.tolist()

    raise ValueError(f"conversion of {x!r} to json not supported")


def str_from_any(x: Any) -> str:
    k = from_any(x)
    if not isinstance(k, str):
        raise TypeError(f"use of {x!r} as a json key is not supported")
    return k


def save_json(path: str, content: Any) -> None:
    """Saves a content to a JSON file.

    Args:
        path: The path to the output JSON file.
        content: The content to save (typically a dictionary).
    """

    with open(path, "w", encoding="utf-8") as f:
        content_json = from_any(content)
        json.dump(content_json, f, indent=2)


def load_json(path: str, keys_to_int: bool = False) -> Any:
    """Loads the content of a JSON file.

    Args:
        path: The path to the input JSON file.
        keys_to_int: Whether to convert keys to integers.
    Returns:
        The loaded content (typically a dictionary).
    """

    def convert_keys_to_int(x):
        return {int(k) if k.lstrip("-").isdigit() else k: v for k, v in x.items()}

    with open(path, "r") as f:
        if keys_to_int:
            return json.load(f, object_hook=convert_keys_to_int)
        else:
            return json.load(f)



def _is_optional(t: Type) -> bool:
    # type: ignore
    return getattr(t, "__origin__", None) is Union and t.__args__[1:] == (type(None),)


def _is_sequence(t: Type) -> bool:
    """Returns true if T is a sequence-like type, e.g. List[int] or
    Set[int] or Tuple[int,...].
    """

    origin = getattr(t, "__origin__", None)
    # Python 3.7 has __origin__==List; python 3.8 changes to list
    if origin in (list, set, frozenset, List, Set, FrozenSet):
        return True
    if origin in (tuple, Tuple) and t.__args__[1:] == (Ellipsis,):
        return True
    return False


def get_real_type(t):
    """
    Get the actual type associated with a typing.Type.
    You can't do List[int]([1,2,3]); you have to do list([1,2,3]).

    __orig_bases__ lets us find the latter from the former.
    >>> get_real_type(List[int])
    <class 'list'>
    >>> get_real_type(Tuple[int, float])
    <class 'tuple'>
    """
    if hasattr(t, "__orig_bases__"):
        return t.__orig_bases__[0]
    else:
        return t.__origin__


def _append_path(path: str, key: Union[str, int]) -> str:
    "build up dotted path for error messages"
    if path:
        return f"{path[:-1]}.{key}'"
    else:
        return f" at '{key}'"


def validate_json(j: JsonValue, t: Type[T], at: str = "") -> T:
    """
    Convert a json object (one of None, bool, int, float, str, [json], {str: json})
    to an object of type t.

    t must be compatible type. Compatible types are:

    - NoneType, bool, int, float, src

    - typing.List/Set/FrozenSet/MutableSet[T] where T is a compatible type.

    - typing.Dict[str,V] where V is a compatible type

    - typing.Tuple[T1, T2, ...] where T1, T2, ... are compatible types. The
      tuple can end with an actual ellipsis to indicate a variable number of
      same-typed elements.

    - typing.NamedTuple(_, f1=T1, f2=T2, ...) where T1, T2, ... are compatible
      types.

    - Optional[T] where T is a compatible type.

    Raises TypeError if `t` isn't a compatible type.

    Raises ValueError if the json object doesn't match type t.
    """
    # type checkers just have no idea what's going on here, so delegate
    # to an unchecked function.
    return _validate_json(j, t, at)


def _validate_json(j, t, at: str):
    # unwrap optional types
    if _is_optional(t):
        if j is None:
            return None
        else:
            t = t.__args__[0]

    tname = getattr(t, "__name__", str(t))

    # bool, int and float interconvert as long as the value is unchanged.
    # So "3.0" is valid for an int and "4" is valid for a float, and "0"
    # and "1" are valid for bools.
    if t in (bool, int, float):
        if type(j) not in (bool, int, float) or t(j) != j:
            raise ValueError(f"{j!r} is not a valid value of type {tname}{at}")
        return t(j)

    # str and None must be exactly represented in json
    if t is str:
        if not isinstance(j, str):
            raise ValueError(f"expected json string but got {j}{at}")
        return j

    if t in (None, type(None)):
        if j is not None:
            raise ValueError(f"expected json null but got {j}{at}")
        return None

    origin = getattr(t, "__origin__", None)

    # sequence types
    if _is_sequence(t):
        return _json_to_seq(j, t, at)

    # tuples
    if origin in (tuple, Tuple):
        return _json_to_tuple(j, t, at)

    # mapping types
    if origin in (dict, Dict):
        return _json_to_dict(j, t, at)

    if origin is Union:
        for tt in t.__args__:
            try:
                return _validate_json(j, tt, at)
            except ValueError:
                pass
        raise ValueError(f"expected json Union but got {j}{at}")

    if inspect.isclass(t) and issubclass(t, Enum):
        return t(j)

    if hasattr(t, "__annotations__"):
        return _json_to_struct(j, t, at)

    # NumPy types
    if "numpy" in str(t):
        if np.issubdtype(t, np.ndarray):
            return np.array(j)
        else:
            return t(j)

    # If the passed-in type is not supported, that's a TypeError bug, not a ValueError
    # in the input.
    raise TypeError(f"don't know how to validate {t}{at}")


def _json_to_struct(j, t, at):
    """
    Convert json to a dataclass or namedtuple type.

    >>> class Named(NamedTuple):
    ...    a : int
    ...    b : float
    ...    c : Optional[int]
    ...    d : int = 4

    >>> _json_to_struct({'a': 1, 'b': 2, 'c': 3, 'd': 5}, Named, '')
    Named(a=1, b=2.0, c=3, d=5)
    >>> _json_to_struct({'a': 1, 'b': 2}, Named, '')
    Named(a=1, b=2.0, c=None, d=4)
    """
    if not isinstance(j, dict):
        raise ValueError(f"expected json dict but got {j!r}{at}")

    types = get_type_hints(t)
    args = {}
    for k, v in j.items():
        at_k = _append_path(at, k)

        if k not in types:
            raise ValueError(f"unknown field{at_k} for {t.__name__}")

        args[k] = _validate_json(v, types[k], at_k)

    try:
        return t(**args)
    except TypeError as e:
        # convert to ValueError and add context
        raise ValueError(str(e) + at) from e


def _json_to_tuple(j, tuple_type, at: str):
    """
    Convert json to a tuple type (like Tuple[int,float,str]).

    >>> _json_to_tuple([1, 2, 'x'], Tuple[int,float,str], '')
    (1, 2.0, 'x')
    >>> _json_to_tuple([1, 2, 3], Tuple[int,...], '')
    (1, 2.0, None)
    """
    elem_types = tuple_type.__args__

    if elem_types[1:] == [Ellipsis]:
        # sequence-like tuple
        return tuple(_json_to_seq(j, List[elem_types[0]], at))

    if not isinstance(j, (list, tuple)):
        raise ValueError(f"expected json list/tuple but got {j!r}{at}")

    n = len(elem_types)
    if len(j) != n:
        # fixed-size tuple
        raise ValueError(f"expected {n}-tuple but got {len(j)}{at}")

    return tuple(
        _validate_json(j[i], elem_types[i], _append_path(at, i)) for i in range(n)
    )


def _json_to_seq(j, seq_type, at):
    """
    Convert json to a Sequence type (like List[int], Set[str], etc.)

    >>> _json_to_seq([1, 2], List[float])
    [1.0, 2.0]
    >>> _json_to_seq([1, 2], Set[float])
    {1.0, 2.0}
    >>> _json_to_seq([1, 2], FrozenSet[float])
    frozenset({1.0, 2.0})
    >>> _json_to_seq([1, 2, 3], Tuple[float,...])
    (1.0, 2.0, 3.0)

    >>> _json_to_seq(1, List[float])
    Traceback (most recent call last):
        ...
    ValueError: expected json list but got <class 'int'> for typing.List[float]
    """
    if not isinstance(j, list):
        raise ValueError(f"expected json list but got {j!r} for {seq_type}")
    elem_type = seq_type.__args__[0]
    elems = [
        _validate_json(elem, elem_type, _append_path(at, i))
        for (i, elem) in enumerate(j)
    ]

    return get_real_type(seq_type)(elems)


def _json_to_dict(j, t, at: str):
    """
    Convert a json dict to a typed Python dict, by just mapping validate_json() over
    the values.

    >>> _json_to_dict({'x':1, 'y': 2}, Dict[str, float], '')
    {'x': 1.0, 'y': 2.0}

    >>> _json_to_dict({'x':1, 'y': 2}, Dict[int, float], '')
    Traceback (most recent call last):
        ...
    TypeError: Dict types must have string keys

    >>> _json_to_dict([], Dict[str, float], '')
    Traceback (most recent call last):
        ...
    ValueError: expected json object but got <class 'list'> for typing.Dict[str, float]

    >>> _json_to_dict({'x':1}, Dict[str, str], '')
    Traceback (most recent call last):
        ...
    ValueError: expected <class 'str'> but got <class 'int'> at 'x'
    """
    kt, vt = t.__args__
    if not isinstance(j, dict):
        raise ValueError(f"expected json object but got {j!r} for {t}{at}")

    d = {}
    for k, v in j.items():
        at_k = _append_path(at, k)
        d[_validate_json(k, kt, at_k)] = _validate_json(v, vt, at_k)
    return d
