import argparse
import numpy as np
from copy import deepcopy
from .contraction import contract
from .expansion import expand

from cpupc.die.die import Die
from cpupc.netlist.netlist import Netlist
from cpupc.netlist.module import Module, Boundary
from cpupc.utils.keywords import KW
from cpupc.geometry.geometry import Point, Rectangle, Shape
from cpupc.netlist.netlist import HyperEdge

from typing import Any


def _add_floorset_nets(netlist: Netlist, die: Die, penalty: float) -> int:
    """
    Add artificial nets and modules to penalize floorset constraint
    violations (Boundary and Cluster constraints).

    Params:
        netlist:
        die:
        penalty: weight of the artificial nets added, as a relative
                 value of the max net weight in the netlist.

    Boundary penalty:
    At each corner in the die a module is added. For each module
    that has a boundary constraint, artificial nets are added to the
    corresponding corner module(s) (e.g for a TOP-LEFT constraint, nets
    to the top-left corner module; for a TOP constraint, nets to the
    top-left and top-right corner modules)

    Cluster penalty:
    An artificial net is added for each cluster, with a high weight.
    """
    added_nets: int = 0

    # TODO: clean this mess up

    W, H = die.width, die.height

    # add modules at each corner
    TOP_LEFT: str = Boundary.TOP_LEFT
    TOP_RIGHT: str = Boundary.TOP_RIGHT
    BOTTOM_LEFT: str = Boundary.BOTTOM_LEFT
    BOTTOM_RIGHT: str = Boundary.BOTTOM_RIGHT


    kwargs = {KW.FIXED: True, KW.IO_PIN: True}
    corner_modules: dict[str, Module] = {}
    
    kwargs[KW.RECTANGLES] = [Rectangle(**{KW.CENTER: Point(0, H), KW.SHAPE: Shape(w=0, h=0)})]
    corner_modules[TOP_LEFT] = Module(TOP_LEFT, **kwargs)
    
    kwargs[KW.RECTANGLES] = [Rectangle(**{KW.CENTER: Point(W, H), KW.SHAPE: Shape(w=0, h=0)})]
    corner_modules[TOP_RIGHT] = Module(TOP_RIGHT, **kwargs)

    kwargs[KW.RECTANGLES] = [Rectangle(**{KW.CENTER: Point(0, 0), KW.SHAPE: Shape(w=0, h=0)})]
    corner_modules[BOTTOM_LEFT] = Module(BOTTOM_LEFT, **kwargs)

    kwargs[KW.RECTANGLES] = [Rectangle(**{KW.CENTER: Point(W, 0), KW.SHAPE: Shape(w=0, h=0)})]
    corner_modules[BOTTOM_RIGHT] = Module(BOTTOM_RIGHT, **kwargs)

    for module in corner_modules.values():
        netlist.modules.append(module)
        netlist._name2module[module.name] = module

    # fake net weight
    weight: float = penalty * max(net.weight for net in netlist.edges)

    # add nets for each boundary constraint
    for module in netlist.modules:
        if not module.boundary or module.name in corner_modules:
            continue

        new_net: list[Module] = [module]
        match module.boundary:
            case Boundary.TOP_LEFT:
                new_net.append(corner_modules[TOP_LEFT])
            
            case Boundary.TOP_RIGHT:
                new_net.append(corner_modules[TOP_RIGHT])

            case Boundary.BOTTOM_LEFT:
                new_net.append(corner_modules[BOTTOM_LEFT])

            case Boundary.BOTTOM_RIGHT:
                new_net.append(corner_modules[BOTTOM_RIGHT])
            
            case Boundary.TOP:
                new_net.append(corner_modules[TOP_LEFT])
                new_net.append(corner_modules[TOP_RIGHT])

            case Boundary.BOTTOM:
                new_net.append(corner_modules[BOTTOM_LEFT])
                new_net.append(corner_modules[BOTTOM_RIGHT])

            case Boundary.LEFT:
                new_net.append(corner_modules[TOP_LEFT])
                new_net.append(corner_modules[BOTTOM_LEFT])

            case Boundary.RIGHT:
                new_net.append(corner_modules[TOP_RIGHT])
                new_net.append(corner_modules[BOTTOM_RIGHT])

            case _:
                raise Exception(f"Unknown boundary constraint {module.boundary}")
        
        netlist.edges.append(HyperEdge(new_net, weight))
        added_nets += 1

    # add fake nets for each cluster
    for cluster in netlist.adjacency_clusters.get_clusters():
        netlist.edges.append(HyperEdge(list(cluster), weight))
        added_nets += 1

    return added_nets

def _cleanup_floorset_nets(netlist: Netlist, added_nets: int) -> None:
    """
    Remove the artificial nets added to enforce floorset constraints.
    """
    for _ in range(added_nets):
        netlist.edges.pop()
    
    # delete the 4 fake corner modules
    for _ in range(4):
        netlist.modules.pop()


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
    # add fake nets to enforce floorset constraints
    added_nets: int = _add_floorset_nets(netlist, die, hyperparams.get('penalty', 10))
    
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
        current_hpwl: float = sum([net.hpwl for net in netlist.edges])
        
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

    _cleanup_floorset_nets(best_netlist, added_nets)

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
