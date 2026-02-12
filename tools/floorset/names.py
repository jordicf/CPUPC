"""Module to manage names between floorset and netlists"""

import re
from typing import Optional, TypeAlias
from enum import StrEnum
from cpupc.utils.utils import (
    read_json_yaml_file,
    write_json_yaml,
    file_type_from_suffix,
    FileType,
)


class FsNames(StrEnum):
    MODULES = "modules"
    PINS = "pins"
    MIB = "mib"
    CLUSTER = "cluster"


DictNames: TypeAlias = Optional[dict[str, list[str]]]


def read_names(names_file: Optional[str]) -> DictNames:
    """Reads the names file if it is provided, otherwise returns None."""
    if names_file is not None:
        names = read_json_yaml_file(names_file)
        assert isinstance(names, dict), "Names file does not contain a dictionary"
        assert FsNames.MODULES in names, "Names file does not contain modules"
        assert FsNames.PINS in names, "Names file does not contain pins"
        assert FsNames.MIB in names, "Names file does not contain MIBs"
        assert FsNames.CLUSTER in names, "Names file does not contain adjacent clusters"
        return names
    else:
        return None


def write_names(names: DictNames, names_file: Optional[str]) -> None:
    """Writes the names file if it is provided."""
    if names_file is not None and names is not None:
        file_type = file_type_from_suffix(names_file)
        assert file_type != FileType.UNKNOWN, "Unknown suffix for names file"
        write_json_yaml(names, file_type == FileType.JSON, names_file)


def rename_msg(msg: str, names: DictNames) -> str:
    """Renames the module names in the error message according to the names file."""
    assert names is not None, "Names file is required to rename error messages"
    modules = set(re.findall(r"M\$\d+", msg))
    for module in modules:
        i = int(module[2:])
        msg = msg.replace(module, module_name(i, names))
    return msg


def module_name(i: int, names: DictNames) -> str:
    return names[str(FsNames.MODULES)][i] if names is not None else f"M_{i}"


def pin_name(i: int, names: DictNames) -> str:
    return names[str(FsNames.PINS)][i] if names is not None else f"P_{i}"


def mib_name(i: int, names: DictNames) -> str:
    return names[str(FsNames.MIB)][i - 1] if names is not None else f"MIB_{i}"


def cluster_name(i: int, names: DictNames) -> str:
    return names[str(FsNames.CLUSTER)][i - 1] if names is not None else f"CL_{i}"


def name2idx(names: list[str], offset: int = 0) -> dict[str, int]:
    """Helper function to convert a list of names into a dictionary of name to index.
    The names are sorted by their numerical suffix (if it exists) and then by their name.
    The indices are assigned according to this order, starting from the given offset.
    """
    name_idx = list({(name, _str_suffix(name)) for name in names})
    name_idx.sort(key=lambda x: (x[1], x[0]))
    return {name: i + offset for i, (name, _) in enumerate(name_idx)}


def _str_suffix(s: str) -> int:
    """Helper function to extract the numerical suffix of a string.
    In case there is no numerical suffix, -1 is returned.
    For example, for "M_10" it returns 10, for "P_5" it returns 5, and for "M_fixed" it returns -1.
    """
    temp_idx = s.rfind(next(filter(lambda x: not x.isdigit(), s[::-1])))
    suffix = s[temp_idx + 1 :]
    return int(suffix) if suffix.isdigit() else -1
