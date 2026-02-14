import argparse
import numpy as np
from copy import deepcopy
from .contraction import contract
from .expansion import expand

from cpupc.die.die import Die
from cpupc.netlist.netlist import Netlist

from typing import Any


def _get_centers_array(netlist: Netlist) -> np.ndarray:
    centers = []
    for m in netlist.modules:
        if m.center is not None:
           centers.append(np.array([m.center.x, m.center.y]))
        else:
           centers.append(np.array([0., 0.]))
    return np.array(centers)


def _run_contract_expand(
    netlist: Netlist,
    die: Die,
    hyperparams: dict,
) -> Netlist:
    """
    Runs the whole contraction - expansion algorithm

    Params:
        netlist: netlist object
        die: Die object with dimensions
        hyperparams: hyperparameters dictionary
    """
    
    H: float = die.height
    W: float = die.width
    
    iter: int = 1
    best_hpwl: float = float('inf')
    patience: int = hyperparams.get("patience", 1)
    no_improvement_count: int = 0

    while True:
        print(f"Starting big iteration {iter}")
        

        contract(netlist, hyperparams=hyperparams)
        expand(netlist, H, W, hyperparams)

        # stopping condition
        current_hpwl: float = netlist.wire_length
        
        if current_hpwl < best_hpwl:
            best_hpwl = current_hpwl
            best_netlist = deepcopy(netlist)
            no_improvement_count = 0
            print(f"New best HPWL: {best_hpwl:.2f}")
        else:
            no_improvement_count += 1
            print(f"Current HPWL: {current_hpwl:.2f}")
            if no_improvement_count >= patience:
                print(f"Stopping: HPWL did not improve for {patience} iterations.")
                break
        
        iter += 1

    return best_netlist


def parse_options(
    prog: str | None = None, args: list[str] | None = None
) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Relocate and reshape the modules of the netlist using "
        "the attraction-repulsion algorithm.",
    )
    parser.add_argument("--netlist", required=True, help="input netlist filename")
    parser.add_argument(
        "-d",
        "--die",
        metavar="<WIDTH>x<HEIGHT> or FILENAME",
        required=True,
        help="size of the die (width x height) or name of the file",
    )
    parser.add_argument("--output", required=True, help="output netlist file")
    parser.add_argument(
        "--overlap_tolerance",
        default=1e-3,
        type=float,
        help="tolerance to stop the expansion phase (the lower, the less overlap but more cpu time)",
    )
    parser.add_argument(
        "--patience",
        default=1,
        type=int,
        help="Number of iterations without HPWL improvement to stop.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Seed for simulated annealing.",
    )
    parser.add_argument(
        "--swaps",
        default=100,
        type=int,
        help="Number of swaps per iteration in simulated annealing.",
    )
    parser.add_argument(
        "--split_threshold",
        default=0.5,
        type=float,
        help="Area threshold for splitting modules in simulated annealing (0.0-1.0).",
    )
    parser.add_argument(
        "--star",
        default=1,
        type=int,
        help="Use star model for split nets in simulated annealing (1: True, 0: False).",
    )

    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> None:
    options: dict[str, str] = parse_options(prog, args)

    netlist_path: str = options["netlist"]
    die_path: str = options["die"]
    out_path: str = options["output"]
    out_file_type: str = out_path[out_path.rfind(".") :]
    # expansion hyperparams
    overlap_tolerance: float = options["overlap_tolerance"]
    # contraction hyperparams
    swaps: int = options["swaps"]
    split_threshold: float = options["split_threshold"]
    star: bool = bool(options["star"])
    patience: int = options["patience"]
    seed: int | None = options["seed"]

    if out_file_type not in [".json", ".yaml"]:
        raise ValueError(f"Invalid output file type {out_path}, must be json or yaml")

    netlist: Netlist = Netlist(netlist_path)
    die: Die = Die(die_path)

    hyperparams = {
        "patience": patience,

        "contraction": {
            "swaps": swaps,
            "split_threshold": split_threshold,
            "star": star,
            "seed": seed,
            "tfactor": 0.95,
            "accept": 0.5,
            "accept_decay": 0.9,
        },

        "expansion": {
            "epsilon": overlap_tolerance,  # improvement to stop expanding
            "epsilon_decay": 0.8,
            "epsilon_min": 1e-4,
            "repel_rectangles": {
                "epsilon": 1e-2,  # improvement to stop iterating
                "epsion_decay": 0.9,
                "min_epsilon": 1e-4,
                "resolution": 1e-3,  # minimum relative distance between hanan grid cells
                "resolution_decay": 1,
            },
            "pseudo_solver": {
                "epsilon": 1e-5,  # improvement to stop iterating
                "lr": 1,  # initial learning rate
                "lr_decay": 0.9,
            },
        },
    }

    # Check that centers are initialized
    for m in netlist.modules:
        if m.center is None:
             raise ValueError(f"Module {m.name} has no center initialized. Use 'grdraw' or 'force' first.")

    netlist = _run_contract_expand(
        netlist=netlist,
        die=die,
        hyperparams=hyperparams,
    )

    if out_file_type == ".json":
        netlist.write_json(out_path)

    elif out_file_type == ".yaml":
        netlist.write_yaml(out_path)


if __name__ == "__main__":
    main()
