# (c) Jordi Cortadella 2026
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

"""Tool to read a netlist in the floorset format"""

import argparse
import torch
from typing import Optional, Any
from cpupc.netlist.netlist import Netlist
from cpupc.utils.utils import file_type_from_suffix, FileType
from .writenetlist import write_netlist
from .names import write_names


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
        description="Converts a netlist from FPEF to floorset",
    )
    parser.add_argument("netlist", help="input file (netlist)")
    parser.add_argument("--data", required=True, help="output file (data)")
    parser.add_argument("--label", required=True, help="output file (label)")
    parser.add_argument("--names", help="optional output file (names)")
    return vars(parser.parse_args(args))


def main(prog: Optional[str] = None, args: Optional[list[str]] = None) -> None:
    """Main function."""
    options = parse_options(prog, args)

    # Before doing anything, check output file suffix
    netlist_type = file_type_from_suffix(options["netlist"])
    assert netlist_type != FileType.UNKNOWN, "Unknown suffix for netlist file"

    names_type = FileType.UNKNOWN
    if options["names"]:
        names_type = file_type_from_suffix(options["names"])
        assert names_type != FileType.UNKNOWN, "Unknown suffix for names file"

    data, label, names = write_netlist(Netlist(options["netlist"]))

    torch.save([data], options["data"])
    torch.save([label], options["label"])
    write_names(names, options["names"])


if __name__ == "__main__":
    main()
