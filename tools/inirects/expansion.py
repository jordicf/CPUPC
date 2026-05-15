import numpy as np
from tools.inirects.repel_rectangles import repel_rectangles
from tools.inirects.pseudo_solver import solve_widths
from cpupc.netlist.netlist import Netlist
from cpupc.geometry.geometry import Shape, Rectangle, AspectRatio
from collections import defaultdict
from numpy.typing import NDArray
from numpy import float64
from typing import Any

# weighted center of mass aproach

def set_heights_widths(netlist: Netlist, h: NDArray[float64], 
                       w: NDArray[float64], original_idx: list[int]) -> None:
    N: int = len(h)

    for i in range(N):
        idx = original_idx[i]
        module = netlist.modules[idx]

        assert not module.is_iopin # sanity check
        if not module.is_fixed and not module.is_hard:
            center = module.center
            module._rectangles = [
                Rectangle(center=center, shape=Shape(float(w[i]), float(h[i])), 
                          fixed=module.is_fixed, hard=module.is_hard)
            ]

def set_centers(netlist: Netlist, centers: NDArray,
                original_idx: list[int]) -> None:
    mod_centers = {}
    N: int = len(centers)
    for i in range(N):
        idx = original_idx[i]
        module = netlist.modules[idx]
        mod_centers[module.name] = (float(centers[i][0]), float(centers[i][1]))
    
    netlist.update_centers(mod_centers)

def pairwise_overlap(centers: NDArray, 
                     h: NDArray[float64], w: NDArray[float64],
                     i: int, j: int) -> float:
    """
    Returns overlap between rectangles "i" and "j"
    """
    xi, yi, wi, hi = centers[i][0], centers[i][1], w[i], h[i]
    xj, yj, wj, hj = centers[j][0], centers[j][1], w[j], h[j]
    

    h_overlap: float = max(0, (wi + wj)/2 - abs(xj - xi))
    v_overlap: float = max(0, (hi + hj)/2 - abs(yj - yi))
    
    return h_overlap * v_overlap

def total_overlap(centers: NDArray, h: NDArray, w: NDArray) -> float:
    """
    Calculates total overlap between all pairs of rectangles
    """
    
    N: int = len(w)
    accum_overlap: float = 0
    
    for i in range(N):
        for j in range(i + 1, N):
            accum_overlap += pairwise_overlap(centers, h, w, i, j)

    return accum_overlap

def ar_bounds(centers: NDArray, areas: NDArray[float64], 
               ar_min: NDArray[float64], ar_max: NDArray[float64], 
               H: float, W: float) -> tuple[NDArray, NDArray]:
    """
    Finds the minimum and maximum aspect ratio each rectangle can
    have, given its center, so as not to exceed the die bound.
    The aspect ratio is also bounded by hard limits ar_max, ar_min.

    Params:
        centers: list of (x,y) coordinates of the rectangle centers
        areas: array of rectangle areas
        ar_min, ar_max: list with hard aspect ratio limits
        H, W: die dimensions

    Returns:
        min_AR, max_AR: arrays with the min and max possible aspect
                        ratio for each rectangle
    """
    N: int = len(centers)

    min_AR = list[float]()
    max_AR = list[float]()

    for i in range(N):
        x, y = centers[i]
        A: float = areas[i]
        
        w_bound: float = min(x, W - x) # largest possible half-width
        h_bound: float = min(y, H - y) # largest possible half-height

        if w_bound == 0 or h_bound == 0: # only happens with terminals
            min_AR.append(1)
            max_AR.append(1)
            continue
                
        ar_min_i: float = max(A / (4 * h_bound**2), ar_min[i])
        ar_max_i: float = min((4 * w_bound**2) / A, ar_max[i])
        min_AR.append(ar_min_i)
        max_AR.append(ar_max_i)
    
    return np.array(min_AR), np.array(max_AR)

def expand(netlist: Netlist, H: float, W: float,
           hyperparams: dict[str, Any] = {}) -> None:
    """
    Expands the rectangles in the netlist to reduce overlap, and
    reshapes them. Pins are ignored in this stage.
    """
    hyperparams = hyperparams.get('expansion', {})
    
    # extract data from netlist to arrays for speed
    center_list: list[NDArray[float64]] = []
    height_list: list[float] = []
    width_list: list[float] = []
    area_list: list[float] = []
    ar_min_list: list[float] = []
    ar_max_list: list[float] = []
    original_idx: list[int] = [] # maps 0..N-1 to original module indices
    fixed = set[int]()
    # maps MIB name to list of module indices, ONLY for soft modules
    # soft modules not in an MIB are given their own exclusive MIB
    mib_dict: dict[str, list[int]] = defaultdict(list)

    for i, m in enumerate(netlist.modules):
        if m.is_iopin: # filter out pins
            continue
        
        r = m.rectangles[0]
        center_list.append(np.array([r.center.x, r.center.y]))
        height_list.append(r.shape.h)
        width_list.append(r.shape.w)
        area_list.append(m.area())
        
        min_ar: float = 1 / 3 # default values
        max_ar: float = 3.0
        if m.aspect_ratio:
            min_ar = m.aspect_ratio.min_wh
            max_ar = m.aspect_ratio.max_wh
        else:
            m.aspect_ratio = AspectRatio(min_wh=1/3, max_wh=3)

        ar_min_list.append(min_ar)
        ar_max_list.append(max_ar)

        original_idx.append(i)

        if m.is_fixed:
            fixed.add(len(center_list) - 1)
        
        if m.mib and not m.is_hard:
            mib_dict[m.mib].append(len(center_list) - 1)

        elif not m.is_hard: 
            # soft modules not in an MIB are given their own exclusive MIB
            assert m.name not in mib_dict
            mib_dict[m.name].append(len(center_list) - 1)

    centers: NDArray = np.array(center_list)
    heights: NDArray[float64] = np.array(height_list)
    widths: NDArray[float64] = np.array(width_list)
    areas: NDArray[float64] = np.array(area_list)
    ar_min: NDArray[float64] = np.array(ar_min_list)
    ar_max: NDArray[float64] = np.array(ar_max_list)
    # partition of soft module indices in [0..N-1] into MIB's
    mib_clusters: list[list[int]] = list(mib_dict.values())

    epsilon: float = hyperparams.get('epsilon', 1e-5) * H * W

    overlap: float = total_overlap(centers, heights, widths)

    while True:
        old_centers, old_h, old_w = centers, heights, widths
        prev_overlap = overlap

        centers = repel_rectangles(centers, heights, widths, 
                                   H, W, fixed, hyperparams)

        min_AR, max_AR = ar_bounds(centers, areas, ar_min, ar_max, H, W)

        heights, widths = solve_widths(centers, heights, widths, areas, min_AR,
                                       max_AR, mib_clusters, hyperparams)

        overlap = total_overlap(centers, heights, widths)

        if overlap > prev_overlap - epsilon:
            break
                
    if overlap > prev_overlap:
        set_centers(netlist, old_centers, original_idx)
        set_heights_widths(netlist, old_h, old_w, original_idx)
    else:
        set_centers(netlist, centers, original_idx)
        set_heights_widths(netlist, heights, widths, original_idx)

    # update epsilon
    hyperparams['epsilon'] = max(hyperparams.get('epsilon', 1e-5) * \
                                 hyperparams.get('epsilon_decay', 1),
                                 hyperparams.get('min_epsilon', 1e-5))
