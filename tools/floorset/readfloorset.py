# (c) Jordi Cortadella 2026
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

"""Tool to read a netlist in the floorset format and convert it to FPEF."""

import argparse
from typing import Any, Optional
from .readnetlist import read_floorset_netlist
from cpupc.utils.keywords import KW
from cpupc.utils.utils import (
    write_json_yaml,
    file_type_from_suffix,
    FileType,
    Python_object,
)


def parse_options(
    prog: Optional[str] = None, args: Optional[list[str]] = None
) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        usage="%(prog)s [options]",
        description="Converts a netlist from floorset into FPEF."
    )
    parser.add_argument("--data", required=True, help="input file (data)")
    parser.add_argument("--label", required=True, help="input file (label)")
    parser.add_argument("--netlist", required=True, help="output netlist")
    parser.add_argument("--die", help="output die file (optional)")
    parser.add_argument("--pins", action="store_true", help="include pins in the die boundary")
    return vars(parser.parse_args(args))


def main(prog: Optional[str] = None, args: Optional[list[str]] = None) -> None:
    """Main function."""
    options = parse_options(prog, args)

    # Before doing anything, file suffixes are checked
    netlist_type = file_type_from_suffix(options["netlist"])
    assert netlist_type != FileType.UNKNOWN, "Unknown suffix for netlist file"
    
    die_type = FileType.UNKNOWN
    if options["die"] is not None:
        die_type = file_type_from_suffix(options["die"])
        assert die_type != FileType.UNKNOWN, "Unknown suffix for die file"
    
    netlist, width, height = read_floorset_netlist(
        options["data"], options["label"], options["pins"]
    )

    write_json_yaml(netlist, netlist_type == FileType.JSON, options["netlist"])

    if options["die"] is not None:
        die: Python_object = {
            str(KW.WIDTH): width,
            str(KW.HEIGHT): height
        }
        die_type = file_type_from_suffix(options["die"])
        write_json_yaml(die, die_type == FileType.JSON, options["die"])


if __name__ == "__main__":
    main()
