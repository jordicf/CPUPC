from cpupc.netlist.netlist import Netlist
from tools.fastpswap.netlist import swapNetlist
from tools.fastpswap.anneal_cached import simulated_annealing

def contract(
    netlist: Netlist, 
    hyperparams: dict = {},
    verbose: bool = False
) -> None:
    """
    Optimize module placement using FastPSwap (simulated annealing).
    """        
    hyperparams = hyperparams.get("contraction", {})

    swaps = hyperparams.get("swaps", 100)
    split_threshold = hyperparams.get("split_threshold", 0.5)
    star = hyperparams.get("star", True)
    tfactor = hyperparams.get("tfactor", 0.95)
    accept = hyperparams.get("accept", 0.2)
    accept_decay = hyperparams.get("accept_decay", 1)
    seed = hyperparams.get("seed", None)

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
        seed=seed
    )
    
    swap_net.remove_subblocks()
    swap_net.netlist.update_centers(
       {swap_net.idx2name(i): (p.x, p.y) for i, p in enumerate(swap_net.points)}
    )

    if accept_decay < 1 and "accept" in hyperparams:
        hyperparams["accept"] *= accept_decay
