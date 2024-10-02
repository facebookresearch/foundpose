#!/usr/bin/env python3

"""Utility functions for managing configuration options."""


import argparse
import logging
import re
import typing

from typing import (
    Any,
    Dict,
    Mapping,
    NamedTuple,
    NamedTupleMeta,
    Optional,
    Tuple,
    Union,
)

from . import json_util

logger = logging.getLogger(__name__)


def print_opts(opts: NamedTuple) -> None:
    """Prints options.

    Args:
        opts: Options to be printed.
    """

    separator = "-" * 80
    logger.info(separator)
    logger.info(f"Options {opts.__class__.__name__}:")
    logger.info(separator)
    for name, value in opts._asdict().items():
        logger.info(f"- {name}: {value}")


def load_opts_from_raw_dict(
    opts_raw: Dict[str, Any],
    opts_types: Mapping[str, Any],
    optional_opts_types: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    opts = {}
    for name, type_def in opts_types.items():
        opts[name] = json_util.validate_json(opts_raw[name], type_def)
    if optional_opts_types is not None:
        for name, type_def in optional_opts_types.items():
            if name in opts_raw:
                opts[name] = json_util.validate_json(opts_raw[name], type_def)
            else:
                opts[name] = type_def()
    return opts


def load_opts_from_json(
    path: str,
    opts_types: Mapping[str, Any],
    optional_opts_types: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Loads options from a JSON file.

    Args:
        path: The path to the input JSON file.
        opts_types: A mapping from names of the expected option sets to
            types of the option sets. Example:
            opts_types = {
                "model_opts": config.ModelOpts,
                "train_opts": config.TrainOpts,
            }
            optional_opts_types: A mapping from names of the optional options sets to
            types of the option sets. If not given, set to the type defaults. Example:
            optional_opts_types = {
                "data_opts": config.DataOpts,
            }
    Returns:
        A dictionary mapping names of option sets to validated option sets.
    """

    opts_raw = json_util.load_json(path)
    return load_opts_from_raw_dict(opts_raw, opts_types, optional_opts_types)


def load_from_file(
    path: str,
    opts_types: Mapping[str, Any],
    optional_opts_types: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Loads options from a JSON or YAML file.
    Args:
        path: The path to the input JSON or YAML file.
        opts_types: A mapping from names of the expected option sets to
            types of the option sets.
        optional_opts_types: A mapping from names of the optional options sets to
            types of the option sets. If not given, set to the type defaults.
    Returns:
        A dictionary mapping names of option sets to validated option sets.
    """
    if path.endswith(".json"):
        return load_opts_from_json(path, opts_types, optional_opts_types)
    elif path.endswith(".yaml"):
        return load_opts_from_yaml(path, opts_types, optional_opts_types)
    else:
        raise ValueError(f"File {path} must be a .json or .yaml file")

def convert_to_parser_type(data_type: Any) -> Dict[str, Any]:
    """Converts a data type to a type description understood by ArgumentParser.

    Args:
        data_type: A data type.
    Returns:
        A dictionary with a description of the data type for ArgumentParser.
    """

    # List or Tuple (e.g., List[str], List[int]).
    if typing.get_origin(data_type) in [list, tuple]:
        return {"type": typing.get_args(data_type)[0], "nargs": "+"}

    # Boolean.
    elif data_type == bool:
        return {"type": lambda x: (str(x).lower() in ["true", "1"])}

    # Other types (see https://docs.python.org/3/library/argparse.html#type for
    # a list of supported types).
    else:
        return {"type": data_type}


def add_opts_to_parser(
    opts_type: NamedTuple, parser: argparse.ArgumentParser 
) -> None:
    """Adds options of specified types and defaults to an ArgumentParser.

    Args:
        opts_type: A NamedTuple definition describing the options.
        parser: A parser to which the options are added.
    """

    # Iterate over fields of the NamedTuple definition, collect their types and
    # default values, and add them as options (aka arguments) to the parser.
    for field in opts_type._fields:
        field_info = {}

        # The default value.
        # pylint: disable=W0212, needs to access protected field _field_defaults
        if field in opts_type._field_defaults:
            # pylint: disable=W0212, needs to access protected field _field_defaults
            field_info["default"] = opts_type._field_defaults[field]

        # The data type.
        # pylint: disable=W0212, needs to access protected field _field_types
        if field in opts_type.__annotations__:
            # pylint: disable=W0212, needs to access protected field _field_types
            field_type = opts_type.__annotations__[field]

            # Optional fields (e.g., Optional[int], Optional[List[str]]) act as
            # Union[type, None] and need a special treatment.
            if typing.get_origin(field_type) == Union:
                type_args = typing.get_args(field_type)
                if len(type_args) > 2 or type_args[1] is not None.__class__:
                    raise ValueError(
                        "Only unions of a form Union[type, None] are supported."
                    )
                field_info.update(convert_to_parser_type(type_args[0]))
            else:
                field_info.update(convert_to_parser_type(field_type))

        # Add the option to the parser.
        field_name = field.replace("_", "-")
        parser.add_argument(f"--{field_name}", **field_info)


def parse_opts_from_command_line(
    opts_type: Union[NamedTuple, Mapping[str, NamedTuple]]
) -> Tuple[NamedTuple, Optional[str]]:
    """Parses options from the command line.

    Args:
        opts_type: A data structure defining the options to parse, or a dictionary
            of data structures. In the latter case, each dictionary item defines
            a sub-command and its options (see, e.g., assets.py for an example).
    Returns:
        A tuple with the parsed options and the subcommand name (None if a
        subcommand was specified).
    """

    # Create a parser of command-line arguments.
    parser = argparse.ArgumentParser()

    # Options specific to the selected sub-command.
    if isinstance(opts_type, Mapping):
        # Parse the options (a special parser is created for each sub-command).
        subparsers = parser.add_subparsers(dest="subcmd")
        for subcmd, subcmd_opts_type in opts_type.items():
            subcmd_parser = subparsers.add_parser(subcmd)
            add_opts_to_parser(subcmd_opts_type, subcmd_parser)
        args = parser.parse_args()

        # Check that a valid sub-command was provided.
        if args.subcmd not in opts_type.keys():
            raise ValueError(f"A subcommand required, one of: {opts_type.keys()}")

        # Convert the parsed options to a NamedTuple.
        args_items = args.__dict__.items()
        opts = opts_type[args.subcmd](
            **{k: v for k, v in args_items if v is not None and k != "subcmd"}
        )

        return opts, args.subcmd

    # A single set of options (no sub-commands).
    else:
        # Parse the options.
        add_opts_to_parser(opts_type, parser)
        args = parser.parse_args()

        # Convert the parsed options to a NamedTuple.
        args_items = args.__dict__.items()
        opts = opts_type(**{k: v for k, v in args_items if v is not None})

        return opts, None


def camel_to_snake_name(name: str) -> str:
    """Convert a camel case name to a snake case name.

    Args:
        name: A camel case name (e.g. "InferOpts").
    Returns:
        A snake case name (e.g. "infer_opts").
    """

    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def load_opts_from_json_or_command_line(
    opts_type: Union[NamedTuple, Mapping[str, NamedTuple]]
) -> Tuple[NamedTuple, Optional[str]]:
    """Loads options from a JSON file or the command line.

    The options are loaded from a JSON file specified via `--opts-path`
    command line argument. If this argument is not provided, then the
    options are read directly from the command line.

    Returns:
        A tuple with the parsed options and the subcommand name (None if a
        subcommand was specified).
    """

    # Try to parse argument `--opts-path`.
    parser = argparse.ArgumentParser()
    parser.add_argument("--opts-path", type=str, default=None)
    args = parser.parse_known_args()[0]

    # Load options from a JSON file if `--opts-path` is specified.
    if args.opts_path is not None:
        if isinstance(opts_type, Mapping):
            # See function `parse_opts_from_command_line` for more details
            # about subcommands mentioned in the exception message below.
            raise ValueError(
                "Subcommands are not supported when loading from a JSON "
                "file. Please provide a single definition of options."
            )

        # Get a snake-case version of the options name (e.g. "InferOpts"
        # is converted to "infer_opts").
        opts_name = camel_to_snake_name(opts_type.__name__)

        # Load the options from a JSON file specified by `--opts-path`.
        opts = load_opts_from_json(
            path=args.opts_path, opts_types={opts_name: opts_type}
        )[opts_name]

        return opts, None

    # Otherwise parse options from the command line.
    else:
        return parse_opts_from_command_line(opts_type)
