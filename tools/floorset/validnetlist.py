# (c) Jordi Cortadella 2026
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

from shapely import unary_union
import torch
from itertools import combinations
from shapely.geometry import Polygon
from .fsconst import FsConstraint
from .utils import (
    check_boundary_const,
    check_clust_const,
    check_fixed_const,
    check_preplaced_const,
    check_mib_const,
)


def check_constraints(
    data_file: str,
    label_file: str,
    ref_file: str,
    check_area: bool,
    check_overlap: bool,
    check_mib: bool,
    check_adj_cluster: bool,
    check_boundary: bool,
    check_hard: bool,
    check_fixed: bool,
    aspect_ratio: float = 0.0,
) -> list[str]:
    """Checks the constraints of a floorplan solution against a reference solution.
    Returns a list of strings with the encountered errors."""

    modules, solution, reference = read_floorset_files(data_file, label_file, ref_file)

    errors: list[str] = []

    if check_area:
        errors.extend(check_area_requirements(modules, solution))

    if check_overlap:
        errors.extend(check_overlaps(modules, solution))

    if check_hard:
        indices = torch.Tensor(
            [
                i
                for i in range(len(modules))
                if modules[i][FsConstraint.HARD] == 1 and modules[i][FsConstraint.FIXED] == 0
            ]
        ).long()
        errors.extend(check_fixed_const(indices, solution, reference))

    if check_fixed:
        indices = torch.Tensor(
            [i for i in range(len(modules)) if modules[i][FsConstraint.FIXED] == 1]
        ).long()
        errors.extend(check_preplaced_const(indices, solution, reference))

    if check_mib:
        mibs = torch.Tensor([modules[i][FsConstraint.MIB].item() for i in range(len(modules))]).long()
        errors.extend(check_mib_const(mibs, solution))

    if check_adj_cluster:
        clusters = torch.Tensor(
            [modules[i][FsConstraint.ADJ_CLUSTER].item() for i in range(len(modules))]
        ).long()
        errors.extend(check_clust_const(clusters, solution))

    if check_boundary:
        errors.extend(check_boundary_constraints(modules, solution, reference))
    
    if aspect_ratio > 0:
        errors.extend(check_aspect_ratio_constraints(modules, solution, aspect_ratio))
    
    return errors


def read_floorset_files(
    data_file: str, label_file: str, ref_file: str
) -> tuple[torch.Tensor, list[Polygon], list[Polygon]]:
    """Reads a floorplan in the floorset format (data and label files) and
    a reference floorplan (label file) and returns the netlist,
    the solution polygons and the reference polygons."""

    n = torch.load(data_file)
    assert len(n) == 1
    nmodules = len(n[0][0])

    solution = get_polygons(label_file)
    reference = get_polygons(ref_file)
    assert len(solution) == nmodules, "Unexpected number of modules in label file"
    assert len(reference) == nmodules, "Unexpected number of modules in reference file"
    return n[0][0], solution, reference


def get_polygons(label_file: str) -> list[Polygon]:
    """Reads the label file and returns a list of polygons for each module."""
    rects = torch.load(label_file)
    assert len(rects) == 1, "Unexpected format of label file"
    assert len(rects[0][0]) == 8, "Unexpected format of label file"
    return [Polygon([(float(x), float(y)) for x, y in r]) for r in rects[0][1]]


def check_area_requirements(
    modules: torch.Tensor, solution: list[Polygon]
) -> list[str]:
    """Checks the area constraint of a floorplan solution against a reference solution.
    Returns a list of strings with the encountered errors."""
    errors: list[str] = []
    for i in range(len(modules)):
        area = modules[i][0].item()
        sol_area = solution[i].area
        if sol_area - area < -1e-3:
            errors.append(
                f"Module M${i}: area requirement not met (expected {area}, got {sol_area})"
            )
    return errors


def check_overlaps(
    modules: torch.Tensor, solution: list[Polygon], threshold: float = 1e-3
) -> list[str]:
    """Checks for overlaps between modules in a floorplan solution.
    Returns a list of strings with the encountered errors."""
    errors: list[str] = []
    n = len(modules)
    for i, j in combinations(range(n), 2):
        area = solution[i].intersection(solution[j]).area
        if area > threshold:
            errors.append(f"Modules M${i} and M${j} overlap")
    return errors


def check_boundary_constraints(
    modules: torch.Tensor, solution: list[Polygon], reference: list[Polygon]
) -> list[str]:
    """Checks for violations of the boundary constraint by comparing the 
    intersection area of polygons with the die boundary.

    Args:
        modules (torch.Tensor): Indices representing modules with boundary constraints.
        solution (list): Solutions list containing polygons of the predicted solution.
        reference (list): Solutions list containing polygons of the reference solution.

    Returns:
        list[str]: A list of error messages for violations found.
    """
    # Get the boundary of the die, which is the union of all reference polygons
    bbox = unary_union(reference)
    assert isinstance(bbox, Polygon), "Unexpected geometry type for die boundary"
    width = max(x for x, y in bbox.exterior.coords)
    height = max(y for x, y in bbox.exterior.coords)
    boundary = torch.Tensor([modules[i][5].item() for i in range(len(modules))]).long()
    return check_boundary_const(boundary, solution, width, height)


def check_aspect_ratio_constraints(
    modules, solution, aspect_ratio: float
) -> list[str]:
    """
    Check for violations of the aspect ratio constraint by evaluating the aspect ratio of each polygon.

    Args:
        modules (torch.Tensor): Tensor containing module information.
        solution (list): Solutions list containing polygons.
        aspect_ratio (float): The maximum allowed aspect ratio.

    Returns:
        list[str]: A list of error messages for violations found.
    """
    errors: list[str] = []
    for i in range(len(modules)):
        if modules[i][FsConstraint.HARD] == 1 or modules[i][FsConstraint.FIXED] == 1:
            continue  # Skip hard and fixed modules, as they may not have a defined aspect ratio
        minx, miny, maxx, maxy = solution[i].bounds
        width = maxx - minx
        height = maxy - miny
        if height > 0 and width > 0:
            ar = max(width / height, height / width)
            if ar > aspect_ratio:
                errors.append(
                    f"Module M${i}: aspect ratio constraint violated (aspect ratio {ar:.2f} exceeds bound {aspect_ratio})"
                )
    return errors
