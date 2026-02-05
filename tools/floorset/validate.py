# (c) Jordi Cortadella 2026
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

"""Tool to validate the constraints of a netlist in the floorset format."""

import sys
import argparse
from typing import Any, Optional

from tools.floorset.validnetlist import check_constraints


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
        description="Validates the constraints of a floorplan solution against a reference solution.",
    )
    parser.add_argument("--data", required=True, help="floorplan data file")
    parser.add_argument("--label", required=True, help="floorplan label file")
    parser.add_argument("--ref", required=True, help="reference label file")
    parser.add_argument("-err", help="file to write errors to (optional)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print validation errors")
    parser.add_argument("--area", action="store_true", help="validate area requirements")
    parser.add_argument("--overlap", action="store_true", help="validate overlaps")
    parser.add_argument("--mib", action="store_true", help="validate MIB constraints")
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="validate cluster constraints",
    )
    parser.add_argument(
        "--boundary", action="store_true", help="validate boundary constraints"
    )
    parser.add_argument("--hard", action="store_true", help="validate hard modules")
    parser.add_argument("--fixed", action="store_true", help="validate fixed modules")
    return vars(parser.parse_args(args))


def main(prog: Optional[str] = None, args: Optional[list[str]] = None) -> None:
    """Main function."""
    options = parse_options(prog, args)
    options_checks = ["area", "overlap", "mib", "cluster", "boundary", "hard", "fixed"]
    
    # If none is specified, all are checked
    if not any(options[k] for k in options_checks):
        for opt in options_checks:
            options[opt] = True
            
    errors = check_constraints(
        options["data"],
        options["label"],
        options["ref"],
        options["area"],
        options["overlap"],
        options["mib"],
        options["cluster"],
        options["boundary"],
        options["hard"],
        options["fixed"],
    )
    
    if options["verbose"]:
        if errors:
            print(f"Validation failed with {len(errors)} errors:")
            for error in errors:
                print(f"- {error}")
        else:
            print("Validation passed without any errors.")
            
    if options["err"] is not None:
        with open(options["err"], "w") as f:
            for error in errors:
                f.write(f"{error}\n")
                
    sys.exit(len(errors))    


if __name__ == "__main__":
    main()
