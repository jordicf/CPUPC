# (c) Jordi Cortadella 2026
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

"""Tool to validate the constraints of a netlist in the floorset format."""

import sys
import argparse
from typing import Any, Optional
from .validnetlist import check_constraints
from .names import read_names, rename_msg


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
        epilog="If no specific constraint is selected, all constraints are validated (except aspect ratio). "
        "The aspect ratio constraint can be used together with specific constraints or alone "
        "(in which case it is the only constraint validated). To check all constraints and aspect ratio, use --all together with --ar.",
    )
    parser.add_argument("--data", required=True, help="floorplan data file")
    parser.add_argument("--label", required=True, help="floorplan label file")
    parser.add_argument("--ref", help="reference label file (optional, default: same as label)")
    parser.add_argument("--names", help="optional names file")
    parser.add_argument("-err", help="file to write errors to (optional)")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print validation errors"
    )
    parser.add_argument(
        "--area", action="store_true", help="validate area requirements"
    )
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
    parser.add_argument(
        "--ar", type=float, help="validate aspect ratio within the bounds AR and 1/AR"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="validate all constraints (overrides individual constraint options)",
    )
    return vars(parser.parse_args(args))


def main(prog: Optional[str] = None, args: Optional[list[str]] = None) -> None:
    """Main function."""
    options = parse_options(prog, args)
    options_checks = ["area", "overlap", "mib", "cluster", "boundary", "hard", "fixed"]

    # If none is specified, all are checked
    check_all = options["all"] or (
        not any(options[k] for k in options_checks) and not options["ar"]
    )
    if check_all:
        for opt in options_checks:
            options[opt] = True

    if options["ar"]:
        assert options["ar"] > 0, "Aspect ratio bound must be greater than 0."
        ar = options["ar"]
        aspect_ratio = max(
            ar, 1 / ar
        )  # Store the larger of ar and 1/ar to simplify checks
    else:
        aspect_ratio = 0  # No aspect ratio constraint if not specified

    if not options["ref"]:
        options["ref"] = options[
            "label"
        ]  # If no reference is provided, use the solution as reference to check internal consistency

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
        aspect_ratio,
    )

    if options["names"]:  # Recover module names in the error messages
        names = read_names(options["names"]) if options["names"] else None
        errors = [rename_msg(error, names) for error in errors]

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
