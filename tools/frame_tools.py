# (c) Jordi Cortadella, 2025
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

"""CPUPC command-line utility."""

import argparse
import importlib

# Modules that must be imported dynamically when invoking a tool.
# The keys are the tool names, and the values are the module paths.
# The main function of each module must be called "main" and must
# accept two parameters: prog (str) and args (list[str]).

PROJECT = "cpupc"

TOOLS = {
    "hello": "tools.hello.hello",
    "draw": "tools.draw.draw",
    "legalizer": "tools.legalrect.legalizer",
    "glb_legalizer": "tools.legalrect.glb_legalizer",
    "gp_area_legalizer": "tools.legalrect.gp_area_legalizer",
    "gp_wl_legalizer": "tools.legalrect.gp_wl_legalizer",
    "parse_floorset": "tools.floorset_parser.floorset_handler",
    "uscs_parser": "tools.uscs_parser.uscs_parser",
    "early_router": "tools.early_router.main_router",
    "grdraw": "tools.grdraw.grdraw",
    "attrrep": "tools.attrrep.attrrep",
    "fastpswap": "tools.fastpswap.fastpswap",
    "inipoints": "tools.inipoints.inipoints",
    "inirects": "tools.inirects.inirects",
    "all": "tools.all.all",
}


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(prog="frame")
    parser.add_argument(
        "tool", choices=TOOLS.keys(), nargs=argparse.REMAINDER, help="tool to execute"
    )
    args = parser.parse_args()
    
    if args.tool:
        tool_name, tool_args = args.tool[0], args.tool[1:]
        if tool_name in TOOLS:
            try:
                importlib.import_module(
                    TOOLS[tool_name]).main(f"{PROJECT} {tool_name}",tool_args)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error ({tool_name}): {e}")
        else:
            print("Unknown frame tool:", tool_name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
