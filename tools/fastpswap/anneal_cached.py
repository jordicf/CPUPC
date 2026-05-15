# (c) Jordi Cortadella 2025
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).
"""Simulated annealing optimization for module centroid swapping."""

import math
import numpy as np
from typing import Optional
from numpy import float64
from numpy.typing import NDArray 
from numba import njit
from tools.fastpswap.netlist import swapNetlist, swapPoint
from cpupc.netlist.module import Module, Boundary
from cpupc.die.die import Die



def simulated_annealing(
    netlist: swapNetlist,
    n_swaps: int = 100,
    patience: int = 20,
    target_acceptance: float = 0.5,
    temp_factor: float = 0.95,
    net_factor: float | None = None,
    seed: Optional[int] = None,
    verbose: bool = False,
    die: Optional[Die] = None,
) -> None:

    if verbose:
        print("Creating JIT-compiled simulated annealing...")

    if net_factor is None:
        net_factor = 1 / temp_factor

    assert 0 < temp_factor < 1 and net_factor > 1

    # Flatten the netlist structure into NumPy arrays
    n_points: int = len(netlist.points)
    point_x: NDArray[float64] = np.array([p.x for p in netlist.points], dtype=np.float64)
    point_y: NDArray[float64] = np.array([p.y for p in netlist.points], dtype=np.float64)

    # Point to nets connectivity
    p_nets_indices: list[int] = []
    p_nets_offsets: list[int] = [0]
    for p in netlist.points:
        p_nets_indices.extend(p.nets)
        p_nets_offsets.append(len(p_nets_indices))
    
    point_nets_indices = np.array(p_nets_indices, dtype=np.int32)
    point_nets_offsets = np.array(p_nets_offsets, dtype=np.int32)

    assert n_points == len(point_nets_offsets) - 1

    # Net properties and connectivity
    n_nets = len(netlist.nets)
    n_external_nets: int = netlist._external_nets
    net_weights = np.array([n.weight for n in netlist.nets], dtype=np.float64)
    net_hpwls = np.zeros(n_nets, dtype=np.float64)

    _net_point_indices: list[int] = []
    _net_points_offsets: list[int] = [0]
    for n in netlist.nets:
        _net_point_indices.extend(n.points)
        _net_points_offsets.append(len(_net_point_indices))

    net_points_indices: NDArray[np.int32] = np.array(_net_point_indices, dtype=np.int32)
    net_points_offsets: NDArray[np.int32] = np.array(_net_points_offsets, dtype=np.int32)

    assert n_nets == len(net_points_offsets) - 1

    movable = np.array(netlist.movable, dtype=np.int32)

    # Project corner modules and exclude them from simulated annealing swaps
    sa_movable_list = list[int]()
    if die is not None:
        corner_constraints = {
            Boundary.TOP_LEFT,
            Boundary.TOP_RIGHT,
            Boundary.BOTTOM_LEFT,
            Boundary.BOTTOM_RIGHT,
        }
        for idx in movable:
            parent_idx: int = netlist.points[idx].parent
            mod: Module = netlist.idx2module(parent_idx)
            if mod.boundary in corner_constraints:
                pt: swapPoint = netlist.points[idx]
                w: float = mod.rectangles[0].shape.w
                h: float = mod.rectangles[0].shape.h

                if mod.boundary == Boundary.TOP_LEFT:
                    pt.x = w / 2
                    pt.y = die.height - h / 2
                elif mod.boundary == Boundary.TOP_RIGHT:
                    pt.x = die.width - w / 2
                    pt.y = die.height - h / 2
                elif mod.boundary == Boundary.BOTTOM_LEFT:
                    pt.x = w / 2
                    pt.y = h / 2
                elif mod.boundary == Boundary.BOTTOM_RIGHT:
                    pt.x = die.width - w / 2
                    pt.y = h / 2

                pt.x = max(w / 2, min(die.width - w / 2, pt.x))
                pt.y = max(h / 2, min(die.height - h / 2, pt.y))

                point_x[idx] = pt.x
                point_y[idx] = pt.y
            else:
                sa_movable_list.append(int(idx))
        sa_movable: NDArray[np.int32] = np.array(sa_movable_list, dtype=np.int32)
    else:
        sa_movable = movable

    # initialize array of net hpwl
    _compute_netlist_hpwl(
        point_x, point_y,
        net_weights, net_hpwls,
        net_points_indices, net_points_offsets
    )

    # Precompute constraints for each point
    point_constraints: NDArray[np.int32] = np.zeros(n_points, dtype=np.int32)
    for i in range(n_points):
        parent_idx = netlist.points[i].parent
        mod = netlist.idx2module(parent_idx)
        if mod.boundary is not None:
            point_constraints[i] = Boundary._constraints.get(mod.boundary, 0)

    # Map each constraint code (0 to 10) to the list of movable indices with that constraint
    same_constraint_lists: list[list[int]] = [[] for _ in range(11)]
    for idx in sa_movable:
        c = point_constraints[idx]
        if 0 <= c <= 10:
            same_constraint_lists[c].append(int(idx))

    same_constraint_indices_l = []
    same_constraint_offsets_l = [0]
    for c in range(11):
        same_constraint_indices_l.extend(same_constraint_lists[c])
        same_constraint_offsets_l.append(len(same_constraint_indices_l))

    same_constraint_indices = np.array(same_constraint_indices_l, dtype=np.int32)
    same_constraint_offsets = np.array(same_constraint_offsets_l, dtype=np.int32)

    # Fast simulated annealing using Numba JIT compilation
    jit_simulated_annealing(
        point_x,
        point_y,
        point_nets_indices,
        point_nets_offsets,
        net_weights,
        net_hpwls,
        n_external_nets,
        net_points_indices,
        net_points_offsets,
        sa_movable,
        point_constraints,
        same_constraint_indices,
        same_constraint_offsets,
        n_swaps,
        patience,
        target_acceptance,
        temp_factor,
        net_factor,
        seed,
        verbose,
    )
            
    # Recover the optimized positions
    central_idxs: NDArray[np.int32] = sa_movable[sa_movable < (n_points - netlist._num_subblocks)]
    n_orig_movable = len(central_idxs)  # number of original movable modules
    central_id2pos = {
        int(i): idx for idx, i in enumerate(central_idxs)
    }  # maps central module id => position in central_idx

    final_positions = np.zeros((n_orig_movable, 2), dtype=np.float64)
    submodule_count = np.zeros(n_orig_movable, dtype=np.uint32) # num of submodules for each movable module

    for i in range(len(sa_movable)):
        parent: int = netlist.points[sa_movable[i]].parent  # central module id of submodule i
        assert parent <= sa_movable[i]

        parent_pos: int = central_id2pos[parent]
        submodule_count[parent_pos] += 1
        final_positions[parent_pos, 0] += point_x[sa_movable[i]]
        final_positions[parent_pos, 1] += point_y[sa_movable[i]]

    assert sum(submodule_count) == len(sa_movable)
    assert np.min(submodule_count) > 0

    final_positions /= submodule_count[:, None]
    
    for i in range(n_orig_movable):
        module = netlist.idx2module(central_idxs[i])
        min_x = module.rectangles[0].shape.w / 2
        min_y = module.rectangles[0].shape.h / 2

        netlist.points[central_idxs[i]].x = max(min_x, float(final_positions[i, 0]))
        netlist.points[central_idxs[i]].y = max(min_y, float(final_positions[i, 1]))

    # project modules to their boundaries if they are somehow not there yet
    if die is not None:
        project_netlist(netlist, die)


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
    point_constraints: np.ndarray,
    same_constraint_indices: np.ndarray,
    same_constraint_offsets: np.ndarray,
    n_swaps: int,
    patience: int,
    target_acceptance: float,
    temp_factor: float,
    net_factor: float,
    seed: Optional[int],
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
        n_swaps,
        target_acceptance,
        point_x,
        point_y,
        point_nets_indices,
        point_nets_offsets,
        net_weights,
        net_hpwls,
        external_nets,
        net_points_indices,
        net_points_offsets,
        movable,
        point_constraints,
        same_constraint_indices,
        same_constraint_offsets,
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
            idx1, idx2 = _pick_feasible_swap(
                movable,
                point_constraints,
                same_constraint_indices,
                same_constraint_offsets,
                point_x,
                point_y,
            )
            if idx1 == idx2:
                continue

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
    point_constraints: np.ndarray,
    same_constraint_indices: np.ndarray,
    same_constraint_offsets: np.ndarray,
) -> float:
    """Find the best temperature for simulated annealing.
    nswaps is the number of swaps performed to generate cost samples,
    target_acceptance is the desired acceptance ratio (value in (0,0.95])."""
    assert 0 < target_acceptance <= 0.95, "Target acceptance must be in (0, 0.95]"
    
    cost: list[float] = [] # incremental costs from the original location
    
    net_prev_hpwls = np.empty_like(net_hpwls)
    
    for _ in range(nswaps):
        idx1, idx2 = _pick_feasible_swap(
            movable,
            point_constraints,
            same_constraint_indices,
            same_constraint_offsets,
            point_x,
            point_y,
        )
        if idx1 == idx2:
            continue

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
def _is_closer(c: int, x1: float, y1: float, x2: float, y2: float) -> bool:
    if c == 0:
        return True
    
    # Check X condition
    if (c & 1): # LEFT
        if x2 >= x1: return False
    elif (c & 2): # RIGHT
        if x2 <= x1: return False
        
    # Check Y condition
    if (c & 4): # TOP
        if y2 <= y1: return False
    elif (c & 8): # BOTTOM
        if y2 >= y1: return False
        
    return True


@njit(cache=True)
def _is_valid_swap(c1: int, c2: int, x1: float, y1: float, x2: float, y2: float) -> bool:
    if c1 == c2:
        return True
    if c1 != 0 and not _is_closer(c1, x1, y1, x2, y2):
        return False
    if c2 != 0 and not _is_closer(c2, x2, y2, x1, y1):
        return False
    return True


@njit(cache=True)
def _pick_feasible_swap(
    movable: np.ndarray,
    point_constraints: np.ndarray,
    same_constraint_indices: np.ndarray,
    same_constraint_offsets: np.ndarray,
    point_x: np.ndarray,
    point_y: np.ndarray,
) -> tuple[int, int]:
    n_movable = len(movable)
    idx1 = movable[np.random.randint(0, n_movable)]
    c1 = point_constraints[idx1]

    # Try bounded rejection sampling from all movable modules
    for _ in range(20):
        idx2 = movable[np.random.randint(0, n_movable)]
        if idx1 == idx2:
            continue
        c2 = point_constraints[idx2]
        if _is_valid_swap(c1, c2, point_x[idx1], point_y[idx1], point_x[idx2], point_y[idx2]):
            return idx1, idx2

    # Fall back to sampling from the precomputed set of the exact same constraint
    start = same_constraint_offsets[c1]
    end = same_constraint_offsets[c1 + 1]
    count = end - start
    if count > 1:
        idx2 = same_constraint_indices[start + np.random.randint(0, count)]
        while idx1 == idx2:
            idx2 = same_constraint_indices[start + np.random.randint(0, count)]
        return idx1, idx2

    return idx1, idx1


def project_netlist(swap_netlist: swapNetlist, die: Die) -> None:
    """Projects movable modules to satisfy their boundary constraints,
    handling individual modules and module clusters as a post-processing step.
    """
    # 1) Identify movable modules
    # Original movable modules are those before any split sub-blocks.
    orig_num_points = len(swap_netlist.points) - swap_netlist._num_subblocks
    movable_indices = [idx for idx in swap_netlist.movable if idx < orig_num_points]

    # 2) Identify module clusters
    # Map cluster name -> list of movable module indices belonging to that cluster
    clusters: dict[str, list[int]] = {}
    non_cluster_indices: list[int] = []

    for idx in movable_indices:
        mod = swap_netlist.idx2module(idx)
        if mod.cluster is not None:
            if mod.cluster not in clusters:
                clusters[mod.cluster] = []
            clusters[mod.cluster].append(idx)
        else:
            non_cluster_indices.append(idx)

    # Helper function to project a single module index to its boundary constraint
    def project_module(idx: int) -> None:
        mod = swap_netlist.idx2module(idx)
        if mod.boundary is None:
            return

        pt = swap_netlist.points[idx]
        w = mod.rectangles[0].shape.w
        h = mod.rectangles[0].shape.h

        if "left" in mod.boundary:
            pt.x = w / 2
        elif "right" in mod.boundary:
            pt.x = die.width - w / 2

        if "top" in mod.boundary:
            pt.y = die.height - h / 2
        elif "bottom" in mod.boundary:
            pt.y = h / 2

        # Ensure the module remains strictly within die bounds
        pt.x = max(w / 2, min(die.width - w / 2, pt.x))
        pt.y = max(h / 2, min(die.height - h / 2, pt.y))

    # 3) If a module not in a cluster has a boundary constraint, shift that module accordingly
    for idx in non_cluster_indices:
        project_module(idx)

    # project clusters
    for cluster_indices in clusters.values():
        constrained_indices = [
            idx for idx in cluster_indices if swap_netlist.idx2module(idx).boundary is not None
        ]
        unconstrained_indices = [
            idx for idx in cluster_indices if swap_netlist.idx2module(idx).boundary is None
        ]

        # only projcet clusters that have at least one module with a boundary constraint
        if not constrained_indices:
            continue

        # 4.2) Compute center of mass of constrained modules (area-weighted average)
        total_area: float = sum(swap_netlist._areas[idx] for idx in constrained_indices)
        orig_cm_x: float = sum(swap_netlist._areas[idx] * swap_netlist.points[idx].x for idx in constrained_indices) / total_area
        orig_cm_y: float = sum(swap_netlist._areas[idx] * swap_netlist.points[idx].y for idx in constrained_indices) / total_area

        # project constrained modules
        for idx in constrained_indices:
            project_module(idx)

        # 4.4) Compute the new center of mass of constrained modules
        new_cm_x = sum(swap_netlist._areas[idx] * swap_netlist.points[idx].x for idx in constrained_indices) / total_area
        new_cm_y = sum(swap_netlist._areas[idx] * swap_netlist.points[idx].y for idx in constrained_indices) / total_area

        delta_cm_x = new_cm_x - orig_cm_x
        delta_cm_y = new_cm_y - orig_cm_y

        # 4.5) Compute the new center for each unconstrained module in the cluster
        for idx in unconstrained_indices:
            mod = swap_netlist.idx2module(idx)
            pt = swap_netlist.points[idx]
            w = mod.rectangles[0].shape.w
            h = mod.rectangles[0].shape.h

            pt.x += delta_cm_x
            pt.y += delta_cm_y

            # clip coordinates
            pt.x = max(w / 2, min(die.width - w / 2, pt.x))
            pt.y = max(h / 2, min(die.height - h / 2, pt.y))

    # 5) Perform a random jitter post-processing if two modules have the exact same shape and position.
    n_movable = len(movable_indices)
    for i in range(n_movable):
        for j in range(i + 1, n_movable):
            idx1 = movable_indices[i]
            idx2 = movable_indices[j]
            pt1 = swap_netlist.points[idx1]
            pt2 = swap_netlist.points[idx2]

            mod1 = swap_netlist.idx2module(idx1)
            mod2 = swap_netlist.idx2module(idx2)
            w1, h1 = mod1.rectangles[0].shape.w, mod1.rectangles[0].shape.h
            w2, h2 = mod2.rectangles[0].shape.w, mod2.rectangles[0].shape.h

            if abs(pt1.x - pt2.x) < 0.2 * max(w1,w2) and abs(pt1.y - pt2.y) < 0.2 * max(h1,h2):
                if abs(w1 - w2) < 1e-3 and abs(h1 - h2) < 1e-3:
                    pt1.x += w1 * np.random.uniform(-0.2, 0.2)
                    pt1.y += h1 * np.random.uniform(-0.2, 0.2)
                    pt2.x += w2 * np.random.uniform(-0.2, 0.2)
                    pt2.y += h2 * np.random.uniform(-0.2, 0.2)

                    pt1.x = max(w1 / 2, min(die.width - w1 / 2, pt1.x))
                    pt1.y = max(h1 / 2, min(die.height - h1 / 2, pt1.y))
                    pt2.x = max(w2 / 2, min(die.width - w2 / 2, pt2.x))
                    pt2.y = max(h2 / 2, min(die.height - h2 / 2, pt2.y))

