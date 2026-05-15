from cpupc.die.die import Die
from cpupc.netlist.netlist import Netlist
from tools.fastpswap.netlist import swapNetlist
from tools.fastpswap.anneal_cached import simulated_annealing
from typing import Any, Optional

def contract(
    netlist: Netlist,
    hyperparams: dict[str, Any] = {},
    verbose: bool = False,
    die: Optional[Die] = None,
) -> None:
    """
    Optimize module placement using FastPSwap (simulated annealing).
    """        
    local_hyperparams: dict[str, Any] = hyperparams.get("contraction", {})

    swaps: int = local_hyperparams.get("swaps", 100)
    split_threshold: float = local_hyperparams.get("split_threshold", 0.5)
    star: bool = local_hyperparams.get("star", True)
    tfactor: float = local_hyperparams.get("tfactor", 0.95)
    accept: float = local_hyperparams.get("accept", 0.2)
    accept_decay: float = local_hyperparams.get("accept_decay", 1)
    seed: Optional[int] = local_hyperparams.get("seed", None)

    swap_net = swapNetlist(
        netlist, 
        split_threshold=split_threshold,
        star_model=star,
        verbose=verbose
    )
    
    simulated_annealing(
        swap_net, 
        n_swaps=swaps,
        verbose=verbose,
        temp_factor=tfactor,
        target_acceptance=accept,
        seed=seed,
        die=die,
    )
    
    swap_net.remove_subblocks()
    swap_net.netlist.update_centers(
       {swap_net.idx2name(i): (p.x, p.y) for i, p in enumerate(swap_net.points)}
    )
    
    if accept_decay < 1 and "accept" in local_hyperparams:
        local_hyperparams["accept"] *= accept_decay
