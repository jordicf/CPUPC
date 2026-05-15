import numpy as np
from numpy.typing import NDArray
from numpy import float64, int32

def _new_centers_of_mass(centers: NDArray, heights: NDArray,
                         widths: NDArray, fixed: set[int] = set[int](), 
                         tick_threshold: float = 0.001) -> NDArray:
    """
    Calculates the center of mass of all rectangles. 
    
    Regions not shared have density 1, 
    regions shared by k rectangles have density 1/k.

    Params:
        centers, heights, widths: arrays of rectangle definitions
        fixed: set of indices to skip
        tick_threshold: minimum spacing (as fraction of total range) 
                        between consecutive ticks in Hanan grid

    Returns:
        array[[float, float]]: (x,y) coordinates of the new centers of mass
    """

    N: int = len(centers)

    # center x and y values
    cx: NDArray[float64] = centers[:, 0]
    cy: NDArray[float64] = centers[:, 1]

    # rectangle edge coordinates
    x_start: NDArray[float64] = cx - widths / 2
    x_end:   NDArray[float64] = cx + widths / 2
    y_start: NDArray[float64] = cy - heights / 2
    y_end:   NDArray[float64] = cy + heights / 2

    # build ticks
    x_ticks: NDArray[float64] = np.unique(np.concatenate([x_start, x_end])) 
    y_ticks: NDArray[float64] = np.unique(np.concatenate([y_start, y_end]))

    # merge ticks that are too close (trade accuracy for speed)
    def merge_ticks(ticks: NDArray[float64], threshold: float) -> NDArray:
        if len(ticks) == 0:
            return ticks
        total_range = ticks[-1] - ticks[0]
        if total_range == 0:
            return ticks
        min_gap = threshold * total_range

        new_ticks = []

        deleted = 0

        i = 0
        while i < len(ticks):
            j = i + 1
            while j < len(ticks) and ticks[j] - ticks[i] < min_gap:
                j += 1

            deleted += (j - i  - 1)

            new_ticks.append(np.mean(ticks[i : j]))
            i = j

        return np.array(new_ticks)

    x_ticks = merge_ticks(x_ticks, tick_threshold)
    y_ticks = merge_ticks(y_ticks, tick_threshold)

    nx, ny = len(x_ticks), len(y_ticks)
    hanan_grid: NDArray = np.zeros((nx-1, ny-1), dtype=int)

    # map rectangle coordinates to their closest tick
    def closest_index(ticks: NDArray, value: float) -> int:
        idx: int = int(np.searchsorted(ticks, value))
        if idx == 0:
            return 0
        if idx == len(ticks):
            return len(ticks) - 1
        return idx if abs(ticks[idx] - value) < abs(ticks[idx-1] - value) else idx-1

    i_start: NDArray[int32] = np.array([closest_index(x_ticks, xs) for xs in x_start])
    i_end:   NDArray[int32] = np.array([closest_index(x_ticks, xe) for xe in x_end])
    j_start: NDArray[int32] = np.array([closest_index(y_ticks, ys) for ys in y_start])
    j_end :  NDArray[int32] = np.array([closest_index(y_ticks, ye) for ye in y_end])

    # fill hanan grid
    for s, e, sj, ej in zip(i_start, i_end, j_start, j_end):
        if e > s and ej > sj:  # ensure non-empty slices
            hanan_grid[s:e, sj:ej] += 1

    # calculate cell areas
    dx: NDArray = np.diff(x_ticks)
    dy: NDArray = np.diff(y_ticks)
    area_grid: NDArray = np.outer(dx, dy)

    # cell centers
    x_centers: NDArray[float64] = (x_ticks[:-1] + x_ticks[1:]) / 2
    y_centers: NDArray[float64] = (y_ticks[:-1] + y_ticks[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')

    # density and mass grids
    density_grid: NDArray = np.zeros_like(hanan_grid, dtype=float)
    mask: NDArray = hanan_grid > 0
    density_grid[mask] = 1.0 / hanan_grid[mask]
    mass_grid: NDArray = density_grid * area_grid

    # prefix sums
    prefix_mass: NDArray = mass_grid.cumsum(axis=0).cumsum(axis=1)
    prefix_mass_x: NDArray = (mass_grid * xx).cumsum(axis=0).cumsum(axis=1)
    prefix_mass_y: NDArray = (mass_grid * yy).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(cum_grid: NDArray, i0: int, i1: int, j0: int, j1: int) -> float:
        res: float = cum_grid[i1-1, j1-1]
        if i0 > 0: res -= cum_grid[i0-1, j1-1]
        if j0 > 0: res -= cum_grid[i1-1, j0-1]
        if i0 > 0 and j0 > 0: res += cum_grid[i0-1, j0-1]
        return res

    # compute centers of mass
    cm_list: NDArray = centers.copy()

    for r in range(N):
        if r in fixed:
            continue
        i0, i1 = i_start[r], i_end[r]
        j0, j1 = j_start[r], j_end[r]

        if i1 <= i0 or j1 <= j0:
            # degenerate case (e.g x_start[r] and x_end[r] got mapped to the same tick)
            cm_list[r] = centers[r]  # fall back to previous center
            continue

        m:  float = rect_sum(prefix_mass,   i0, i1, j0, j1)
        mx: float = rect_sum(prefix_mass_x, i0, i1, j0, j1)
        my: float = rect_sum(prefix_mass_y, i0, i1, j0, j1)

        assert m > 0

        cm_list[r] = (mx/m, my/m)

    return cm_list

def repel_rectangles(
        centers: NDArray, heights: NDArray[float64], 
        widths: NDArray[float64], H: float, W: float, 
        fixed: set[int] = set[int](), hyperparams: dict = {}, 
    ) -> NDArray:
    
    hyperparams = hyperparams.get('repel_rectangles', {})
    
    epsilon: float = hyperparams.get('epsilon', 1e-2)
    threshold: NDArray = np.column_stack((widths, heights)) * epsilon
    
    resolution: float = hyperparams.get('resolution', 1e-3) # minimum relative distance between hanan grid ticks
    
    while True:
        prev_centers = centers

        centers = _new_centers_of_mass(centers, heights, widths, fixed, tick_threshold=resolution)
        
        # legalise
        centers[:,0] = np.maximum(widths/2, np.minimum(W - widths/2, centers[:,0]))
        centers[:,1] = np.maximum(heights/2, np.minimum(H - heights/2, centers[:,1]))

        if np.all(np.abs(prev_centers - centers) <= threshold):
            break

    # update epsilon
    hyperparams['epsilon'] = max(hyperparams.get('epsilon', 1e-2) * \
                                 hyperparams.get('epsilon_decay', 1),
                                 hyperparams.get('min_epsilon', 1e-4))
    
    # update resolution
    hyperparams['resolution'] = resolution * hyperparams.get('resolution_decay', 1)

    return centers