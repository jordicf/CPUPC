# (c) Jordi Cortadella 2026
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

import torch
from typing import Any, Optional
from .fsconst import FsNetlist, FsConstraint
from .names import FsNames, module_name, pin_name, mib_name, cluster_name
from cpupc.utils.utils import Python_object
from cpupc.utils.keywords import KW
from cpupc.netlist.module import Boundary
from cpupc.geometry.fpolygon import RPoint, vertices2polygon


def read_floorset_netlist(
    data_file: str, label_file: str, die_pins: bool, names: Optional[Python_object]
) -> tuple[Python_object, float, float]:
    """
    Reads a netlist in the floorset format
    :param netlist_file: input netlist file
    :param floorplan_file: floorplan file
    :param die_pins: computes the die boundary including the pins
    :return: the netlist and the width and height of the die
    """

    n = torch.load(data_file)
    assert len(n) == 1
    modules = n[0][FsNetlist.MODULES]
    module2module = n[0][FsNetlist.MODULE2MODULE]
    pin2module = n[0][FsNetlist.PIN2MODULE]
    pin_pos = n[0][FsNetlist.PIN_POS]
    nmodules = len(modules)
    npins = len(pin_pos)

    # Check that names are consistent with the number of modules and pins
    if names is not None:
        assert isinstance(names, dict), "Names file does not contain a dictionary"
        assert FsNames.MODULES in names, "Names file does not contain modules"
        assert FsNames.PINS in names, "Names file does not contain pins"
        assert FsNames.MIB in names, "Names file does not contain MIBs"
        assert FsNames.CLUSTER in names, "Names file does not contain adjacent clusters"
        assert (
            len(names[FsNames.MODULES]) == nmodules
        ), "Inconsistent number of modules in names file"
        assert (
            len(names[FsNames.PINS]) == npins
        ), "Inconsistent number of pins in names file"

    # Read the modules
    dict_modules: dict[str, dict] = dict()
    for i in range(nmodules):
        mod_name = module_name(i, names)
        hard = False
        info: dict[str, Any] = dict()

        if modules[i][FsConstraint.HARD] == 1:
            info[str(KW.HARD)] = True
            hard = True
        if modules[i][FsConstraint.FIXED] == 1:
            info[str(KW.FIXED)] = True
            hard = True

        if not hard:
            info[str(KW.AREA)] = float(modules[i][FsConstraint.AREA])
        if modules[i][FsConstraint.MIB] != 0:
            idx = int(modules[i][FsConstraint.MIB])
            info[str(KW.MIB)] = mib_name(idx, names)
        if modules[i][FsConstraint.ADJ_CLUSTER] != 0:
            idx = int(modules[i][FsConstraint.ADJ_CLUSTER])
            info[str(KW.ADJ_CLUSTER)] = cluster_name(idx, names)
        if modules[i][FsConstraint.BOUNDARY] != 0:
            constraint = Boundary.from_code(int(modules[i][FsConstraint.BOUNDARY]))
            assert constraint is not None
            info[str(KW.BOUNDARY)] = constraint
        dict_modules[mod_name] = info

    # Read the pins and derive the boundaries of the die
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for p in range(npins):
        pin_name_str = pin_name(p, names)
        info_pin: dict[str, Any] = dict()
        info_pin[str(KW.IO_PIN)] = True
        x, y = float(pin_pos[p][0]), float(pin_pos[p][1])
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
        info_pin[str(KW.RECTANGLES)] = [x, y, 0, 0]
        info_pin[str(KW.FIXED)] = True
        dict_modules[pin_name_str] = info_pin

    # Read the nets module - module
    list_nets: list[list] = []
    for src, dst, w in module2module:
        list_nets.append([module_name(int(src), names), module_name(int(dst), names), float(w)])

    # Read the nets pin - module
    for src, dst, w in pin2module:
        list_nets.append([pin_name(int(src), names), module_name(int(dst), names), float(w)])

    # Now we go to the label file to read rectangles of modules
    rects = torch.load(label_file)
    metrics = rects[0][0]
    rectangles = rects[0][1]
    assert len(metrics) == 8, "Unexpected format in label file"
    assert len(rectangles) == nmodules, "Unexpected format in label file"

    # Forget the pins in die boundary calculation
    if not die_pins:
        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")

    # Extract rectangles for each module and calculate the strop
    for i in range(nmodules):
        mod_name = module_name(i, names)
        vertices: set[RPoint] = {(float(x), float(y)) for x, y in rectangles[i]}
        xmin = min(xmin, min(v[0] for v in vertices))
        xmax = max(xmax, max(v[0] for v in vertices))
        ymin = min(ymin, min(v[1] for v in vertices))
        ymax = max(ymax, max(v[1] for v in vertices))
        polygon = vertices2polygon(vertices)
        strop = polygon.calculate_best_strop()
        assert strop is not None, f"Cannot calculate strop for module {mod_name}"
        list_rectangles: list[list[float]] = []
        for rect in strop.all_rectangles():
            center = rect.center
            width = rect.width
            height = rect.height
            list_rectangles.append([center[0], center[1], width, height])
        dict_modules[mod_name][str(KW.RECTANGLES)] = list_rectangles

    return (
        {
            str(KW.MODULES): dict_modules,
            str(KW.NETS): list_nets,
        },
        xmax - xmin,
        ymax - ymin,
    )
