# (c) Jordi Cortadella 2025
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).
"""Simulated annealing optimization for module centroid swapping."""

import math
import numpy as np
from numba import njit
from .netlist import swapNetlist



def simulated_annealing(
    net: swapNetlist,
    n_swaps: int = 100,
    patience: int = 20,
    target_acceptance: float = 0.5,
    temp_factor: float = 0.95,
    net_factor: float | None = None,
    seed: int = None,
    verbose: bool = False,
) -> None:

    if verbose:
        print("Creating JIT-compiled simulated annealing...")

    if net_factor is None:
        net_factor = 1 / temp_factor

    assert 0 < temp_factor < 1 and net_factor > 1

    # Flatten the netlist structure into NumPy arrays
    n_points = len(net.points)
    point_x = np.array([p.x for p in net.points], dtype=np.float64)
    point_y = np.array([p.y for p in net.points], dtype=np.float64)

    # Point to nets connectivity
    p_nets_indices: list[int] = []
    p_nets_offsets: list[int] = [0]
    for p in net.points:
        p_nets_indices.extend(p.nets)
        p_nets_offsets.append(len(p_nets_indices))
    
    point_nets_indices = np.array(p_nets_indices, dtype=np.int32)
    point_nets_offsets = np.array(p_nets_offsets, dtype=np.int32)

    assert n_points == len(point_nets_offsets) - 1

    # Net properties and connectivity
    n_nets = len(net.nets)
    n_external_nets: int = net._external_nets
    net_weights = np.array([n.weight for n in net.nets], dtype=np.float64)
    net_hpwls = np.zeros(n_nets, dtype=np.float64)

    net_points_indices: list[int] = []
    net_points_offsets: list[int] = [0]
    for n in net.nets:
        net_points_indices.extend(n.points)
        net_points_offsets.append(len(net_points_indices))
        
    net_points_indices = np.array(net_points_indices, dtype=np.int32)
    net_points_offsets = np.array(net_points_offsets, dtype=np.int32)

    assert n_nets == len(net_points_offsets) - 1

    movable = np.array(net.movable, dtype=np.int32)

    # initialize array of net hpwl
    _compute_netlist_hpwl(
        point_x, point_y,
        net_weights, net_hpwls,
        net_points_indices, net_points_offsets
    )

    # Fast simulated annealing using Numba JIT compilation
    jit_simulated_annealing(
        point_x, point_y, point_nets_indices, point_nets_offsets,
        net_weights, net_hpwls, n_external_nets, net_points_indices, 
        net_points_offsets, movable, n_swaps, patience, 
        target_acceptance, temp_factor, net_factor, 
        seed, verbose
    )
            
    # Recover the optimized positions
    central_idxs = movable[movable < (n_points - net._num_subblocks)]
    n_orig_movable = len(central_idxs) # number of original movable modules
    central_id2pos = {i: idx for idx, i in enumerate(central_idxs)} # maps central module id => position in central_idx

    final_positions = np.zeros((n_orig_movable, 2), dtype=np.float64)
    submodule_count = np.zeros(n_orig_movable, dtype=np.uint32) # num of submodules for each movable module

    for i in range(len(movable)):
        parent: int = net.points[movable[i]].parent # central module id of submodule i
        assert parent <= movable[i]

        parent_pos: int = central_id2pos[parent]
        submodule_count[parent_pos] += 1
        final_positions[parent_pos, 0] += point_x[movable[i]]
        final_positions[parent_pos, 1] += point_y[movable[i]]
    
    assert sum(submodule_count) == len(movable)
    assert min(submodule_count) > 0
    
    final_positions /= submodule_count[:, None]
    
    for i in range(n_orig_movable):
        module = net.idx2module(central_idxs[i])
        min_x = module.rectangles[0].shape.w / 2
        min_y = module.rectangles[0].shape.h / 2

        net.points[central_idxs[i]].x = max(min_x, float(final_positions[i, 0]))
        net.points[central_idxs[i]].y = max(min_y, float(final_positions[i, 1]))


@njit(cache=True)
def jit_simulated_annealing(
    point_x: np.ndarray,
    point_y: np.ndarray,
    point_nets_indices: np.ndarray,
    point_nets_offsets: np.ndarray,
    net_weights: np.ndarray,
    net_hpwls: np.ndarray,
    external_nets: int,
    net_points_indices: np.ndarray,
    net_points_offsets: np.ndarray,
    movable: np.ndarray,
    n_swaps: int,
    patience: int,
    target_acceptance: float,
    temp_factor: float,
    net_factor: float,
    seed: int | None,
    verbose: bool,
) -> None:
    """Optimize the netlist using simulated annealing.
    (point_x, point_y, point_nets_indices, point_nets_offsets, net_weights, 
    net_hpwls, net_points_indices, net_points_offsets, movable) all represent 
    the netlist to optimize

    target_acceptance is the desired initial acceptance ratio (value in (0,0.95]),
    temp_factor is the factor by which the temperature decreases,
    patience is the number of iterations to perform without improvement,
    n_swaps is the number of swaps to perform per iteration and per movable point.
    Note that the total number of swaps per iteration is multiplied by the
    number of movable points"""

    if seed is not None:
        np.random.seed(seed)

    # Compute total hpwl of the netlist
    total_hpwl: float = np.sum(net_hpwls)
    total_internal_hpwl: float = np.sum(net_hpwls[external_nets:])

    if verbose:
        print("Running JIT-compiled simulated annealing...")
        print("Initial HPWL:", total_hpwl)

    net_prev_hpwls = net_hpwls.copy()
    
    n_swaps = n_swaps * len(movable)
    
    # Compute the initial temperature
    temp: float = _find_best_temperature(
        n_swaps, target_acceptance,
        point_x, point_y, point_nets_indices, point_nets_offsets,
        net_weights, net_hpwls, external_nets, net_points_indices, 
        net_points_offsets, movable
    )

    # Initial solution
    best_x = point_x.copy()
    best_y = point_y.copy()
    best_hpwl = current_hpwl = total_hpwl
    best_hpwl_internal = current_hpwl_internal = total_internal_hpwl
    best_dispersion = current_dispersion = _compute_internal_unweighted_hpwl(
        external_nets, point_x, point_y, net_points_indices, net_points_offsets
    )

    if verbose:
        print("Initially: Temperature", temp, "HPWL", best_hpwl)

    no_improvement = 0 # Number of iteration without improvement
    iter_count = 0 # Iteration counter
    best_avg = math.inf # Conservative best average HPWL in one iteration
    
    while no_improvement < patience:
        iter_count += 1
        avg = 0.0
        # Perform n_swaps
        for _ in range(n_swaps):
            idx1, idx2 = _pick_two_randomly(movable)
            
            delta_hpwl, delta_hpwl_internal = _swap_points(
                idx1, idx2,
                point_x, point_y, point_nets_indices, point_nets_offsets,
                net_weights, net_hpwls, net_prev_hpwls, external_nets, 
                net_points_indices, net_points_offsets
            )      

            if delta_hpwl < 0 or np.random.random() < np.exp(-delta_hpwl / temp):
                current_hpwl += delta_hpwl
                current_hpwl_internal += delta_hpwl_internal

                if current_hpwl < best_hpwl:
                    no_improvement = -1
                    best_hpwl = current_hpwl
                    best_hpwl_internal = current_hpwl_internal
                    best_x = point_x.copy()
                    best_y = point_y.copy()
            else:
                # Swap back
                _undo_swap(
                    idx1, idx2, point_x, point_y, 
                    point_nets_indices, point_nets_offsets, 
                    net_hpwls, net_prev_hpwls
                )
                
            avg += current_hpwl
        
        current_dispersion = _compute_internal_unweighted_hpwl(
            external_nets, point_x, point_y, net_points_indices, net_points_offsets
        )

        if no_improvement < 0:
            best_dispersion = _compute_internal_unweighted_hpwl(
                external_nets, best_x, best_y, net_points_indices, net_points_offsets
            )

        avg /= n_swaps
        if avg >= best_avg and current_dispersion >= best_dispersion:
            no_improvement += 1
        else:
            no_improvement = 0
            best_avg = avg
            best_dispersion = current_dispersion
            
        if verbose:
            print(
                "Iter.", iter_count,
                "Temp.", temp,
                "HPWL: Avg", avg,
                "Best Avg", best_avg,
                "Best", best_hpwl,
            )
        # decrease temperature and increase internal net weights
        temp = temp * temp_factor
        net_weights[external_nets:] *= net_factor
        net_hpwls[external_nets:] *= net_factor

        # update cost of best and current solutions
        current_hpwl_external = current_hpwl - current_hpwl_internal
        current_hpwl_internal *= net_factor
        current_hpwl = current_hpwl_external + current_hpwl_internal

        best_hpwl_external = best_hpwl - best_hpwl_internal
        best_hpwl_internal *= net_factor
        best_hpwl = best_hpwl_external + best_hpwl_internal

    # Restore best solution
    for i in range(len(point_x)):
        point_x[i] = best_x[i]
        point_y[i] = best_y[i]


@njit(cache=True)
def _find_best_temperature(
    nswaps: int,
    target_acceptance: float,
    point_x: np.ndarray,
    point_y: np.ndarray,
    point_nets_indices: np.ndarray,
    point_nets_offsets: np.ndarray,
    net_weights: np.ndarray,
    net_hpwls: np.ndarray,
    external_nets: int,
    net_points_indices: np.ndarray,
    net_points_offsets: np.ndarray,
    movable: np.ndarray,
) -> float:
    """Find the best temperature for simulated annealing.
    nswaps is the number of swaps performed to generate cost samples,
    target_acceptance is the desired acceptance ratio (value in (0,0.95])."""
    assert 0 < target_acceptance <= 0.95, "Target acceptance must be in (0, 0.95]"
    
    cost: list[float] = [] # incremental costs from the original location
    
    net_prev_hpwls = np.empty_like(net_hpwls)
    
    for _ in range(nswaps):
        idx1, idx2 = _pick_two_randomly(movable)
        
        delta, _ = _swap_points(
            idx1, idx2,
            point_x, point_y, point_nets_indices, point_nets_offsets,
            net_weights, net_hpwls, net_prev_hpwls, external_nets, 
            net_points_indices, net_points_offsets
        )
        cost.append(abs(delta))
        
        # Return to the original location
        _undo_swap(
            idx1, idx2, point_x, point_y,
            point_nets_indices, point_nets_offsets,
            net_hpwls, net_prev_hpwls
        )

    # Compute target temperature
    nonzero_cost = [c for c in cost if c > 0]
    if not nonzero_cost:
        raise ValueError("No valid cost samples found") # should never happen   
    nonzero_cost.sort()
    idx = min(int(len(nonzero_cost) * target_acceptance), len(nonzero_cost) - 1)
    return -nonzero_cost[idx] / math.log(target_acceptance)


@njit(cache=True)
def _compute_net_hpwl(
    net_idx: int,
    px: np.ndarray,
    py: np.ndarray,
    net_points_indices: np.ndarray,
    net_points_offsets: np.ndarray,
    net_weights: np.ndarray,
) -> float:
    """Compute the half-perimeter wire length (HPWL) of a net.
    It returns the computed HPWL."""

    start = net_points_offsets[net_idx]
    end = net_points_offsets[net_idx+1]
    
    if start == end: return 0.0

    p0 = net_points_indices[start]
    min_x = max_x = px[p0]
    min_y = max_y = py[p0]

    for i in range(start + 1, end):
        p = net_points_indices[i]
        nx = px[p]
        ny = py[p]
        if nx < min_x: min_x = nx
        if nx > max_x: max_x = nx
        if ny < min_y: min_y = ny
        if ny > max_y: max_y = ny
        
    return (max_x - min_x + max_y - min_y) * net_weights[net_idx]


@njit(cache=True)
def _compute_internal_unweighted_hpwl(
    external_nets: int,
    px: np.ndarray,
    py: np.ndarray,
    net_points_indices: np.ndarray,
    net_points_offsets: np.ndarray
) -> float:
    """Compute the half-perimeter wire length (HPWL) of all internal nets
    ignoring weights."""

    total_hpwl: float = 0.0
    n_nets: int = len(net_points_offsets) - 1

    for net_idx in range(external_nets, n_nets):        
        start = net_points_offsets[net_idx]
        end = net_points_offsets[net_idx+1]
        
        if start >= end: continue

        p0 = net_points_indices[start]
        min_x = max_x = px[p0]
        min_y = max_y = py[p0]

        for i in range(start + 1, end):
            p = net_points_indices[i]
            nx = px[p]
            ny = py[p]
            if nx < min_x: min_x = nx
            if nx > max_x: max_x = nx
            if ny < min_y: min_y = ny
            if ny > max_y: max_y = ny
        
        total_hpwl += (max_x - min_x) + (max_y - min_y)

    return total_hpwl


@njit(cache=True)
def _compute_netlist_hpwl(
    point_x: np.ndarray,
    point_y: np.ndarray,
    net_weights: np.ndarray,
    net_hpwls: np.ndarray,
    net_points_indices: np.ndarray,
    net_points_offsets: np.ndarray
) -> None:
    for n in range(len(net_hpwls)):
        net_hpwls[n] = _compute_net_hpwl(n, point_x, point_y, 
                                         net_points_indices, net_points_offsets, 
                                         net_weights)

@njit(cache=True)
def _merge_remove_common(list1: np.ndarray, list2: np.ndarray) -> np.ndarray:
    """Merge two sorted lists into a single sorted list without duplicates."""
    merged = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            merged.append(list1[i])
            i += 1
        elif list1[i] > list2[j]:
            merged.append(list2[j])
            j += 1
        else:
            i += 1
            j += 1
    while i < len(list1):
        merged.append(list1[i])
        i += 1
    while j < len(list2):
        merged.append(list2[j])
        j += 1
    return np.array(merged, dtype=np.int32)
    

@njit(cache=True)
def _swap_points(
    idx1: int,
    idx2: int,
    point_x: np.ndarray,
    point_y: np.ndarray,
    point_nets_indices: np.ndarray,
    point_nets_offsets: np.ndarray,
    net_weights: np.ndarray,
    net_hpwls: np.ndarray,
    net_prev_hpwls: np.ndarray,
    n_external_nets: int,
    net_points_indices: np.ndarray,
    net_points_offsets: np.ndarray,
) -> tuple[float, float]:
    """Swap two points and return the change in total HPWL,
    both total and of internal nets
    """
    # swap coordinates
    point_x[idx1], point_x[idx2] = point_x[idx2], point_x[idx1]
    point_y[idx1], point_y[idx2] = point_y[idx2], point_y[idx1]
    
    # Find affected nets
    s1, e1 = point_nets_offsets[idx1], point_nets_offsets[idx1+1]
    nets1 = point_nets_indices[s1:e1]

    s2, e2 = point_nets_offsets[idx2], point_nets_offsets[idx2+1]
    nets2 = point_nets_indices[s2:e2]

    affected_nets = _merge_remove_common(nets1, nets2)

    # Update hpwl
    delta_hpwl = 0.0
    delta_hpwl_internal = 0.0
    for n in affected_nets:
        net_prev_hpwls[n] = net_hpwls[n]
        delta_hpwl -= net_prev_hpwls[n]

        new_h = _compute_net_hpwl(n, point_x, point_y, net_points_indices, net_points_offsets, net_weights)
        net_hpwls[n] = new_h
        delta_hpwl += new_h

        if n >= n_external_nets: # if n is internal
            delta_hpwl_internal += new_h - net_prev_hpwls[n]

    return delta_hpwl, delta_hpwl_internal

@njit(cache=True)
def _undo_swap(
    idx1: int,
    idx2: int,
    point_x: np.ndarray,
    point_y: np.ndarray,
    point_nets_indices: np.ndarray,
    point_nets_offsets: np.ndarray,
    net_hpwls: np.ndarray,
    net_prev_hpwls: np.ndarray
) -> None:
    """Undo the swap of 2 points"""
    
    # swap coordinates
    point_x[idx1], point_x[idx2] = point_x[idx2], point_x[idx1]
    point_y[idx1], point_y[idx2] = point_y[idx2], point_y[idx1]
    
    # Find affected nets
    s1, e1 = point_nets_offsets[idx1], point_nets_offsets[idx1+1]
    nets1 = point_nets_indices[s1:e1]

    s2, e2 = point_nets_offsets[idx2], point_nets_offsets[idx2+1]
    nets2 = point_nets_indices[s2:e2]

    affected_nets = _merge_remove_common(nets1, nets2)

    # Update hpwl
    for n in affected_nets:
        net_hpwls[n] = net_prev_hpwls[n]

@njit(cache=True)
def _pick_two_randomly(choices: np.ndarray) -> tuple[int, int]:
    """Pick two different elements randomly from choices."""
    idx1 = idx2 = np.random.randint(0, len(choices))
    while idx2 == idx1:
        idx2 = np.random.randint(0, len(choices))
    return choices[idx1], choices[idx2]
