# (c) Jordi Cortadella 2026
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

import torch
from typing import Any
from cpupc.utils.utils import Python_object
from cpupc.utils.keywords import KW
from cpupc.netlist.module import Boundary
from cpupc.geometry.fpolygon import RPoint, vertices2polygon


def read_floorset_netlist(
    data_file: str, label_file: str, verbose: bool = False
) -> Python_object:
    """
    Reads a netlist in the floorset format
    :param netlist_file: input netlist file
    :param floorplan_file: floorplan file
    :param verbose: verbose output
    :return: the netlist
    """

    n = torch.load(data_file)
    assert len(n) == 1
    modules = n[0][0]
    module2module = n[0][1]
    pin2module = n[0][2]
    pin_pos = n[0][3]
    nmodules = len(modules)
    npins = len(pin_pos)

    # Read the modules
    dict_modules: dict[str, dict] = dict()
    for i in range(nmodules):
        mod_name = f"M_{i}"
        hard = False
        info: dict[str, Any] = dict()

        if modules[i][1] == 1:
            info[KW.HARD] = True
            hard = True
        if modules[i][2] == 1:
            info[KW.FIXED] = True
            hard = True

        if not hard:
            info[KW.AREA] = float(modules[i][0])
        if modules[i][3] != 0:
            info[KW.MIB] = f"MIB_{int(modules[i][3])}"
        if modules[i][4] != 0:
            info[KW.ADJ_CLUSTER] = f"CL_{int(modules[i][4])}"
        if modules[i][5] != 0:
            constraint = Boundary.from_code(int(modules[i][5]))
            assert constraint is not None
            info[KW.BOUNDARY] = constraint
        dict_modules[mod_name] = info

    # Read the pins
    for p in range(npins):
        pin_name = f"P_{p}"
        info_pin: dict[str, Any] = dict()
        info_pin[KW.IO_PIN] = True
        x, y = float(pin_pos[p][0]), float(pin_pos[p][1])
        info_pin[KW.RECTANGLES] = [x, y, 0, 0]
        info_pin[KW.FIXED] = True
        dict_modules[pin_name] = info_pin

    # Read the nets module - module
    list_nets: list[list] = []
    for src, dst, w in module2module:
        list_nets.append([f"M_{int(src)}", f"M_{int(dst)}", float(w)])

    # Read the nets pin - module
    for src, dst, w in pin2module:
        list_nets.append([f"P_{int(src)}", f"M_{int(dst)}", float(w)])

    # Now we go to the label file to read rectangles of modules
    rects = torch.load(label_file)
    metrics = rects[0][0]
    rectangles = rects[0][1]
    assert len(metrics) == 8, "Unexpected format in label file"
    assert len(rectangles) == nmodules, "Unexpected format in label file"

    # Extract rectangles for each module and calculate the strop
    for i in range(nmodules):
        mod_name = f"M_{i}"
        vertices: set[RPoint] = {(float(x),float(y)) for x,y in rectangles[i]}
        polygon = vertices2polygon(vertices)
        strop = polygon.calculate_best_strop()
        assert strop is not None, f"Cannot calculate strop for module {mod_name}"
        list_rectangles: list[list[float]] = []
        for rect in strop.all_rectangles():
            center = rect.center
            width = rect.width
            height = rect.height
            list_rectangles.append([center[0], center[1], width, height])
        dict_modules[mod_name][KW.RECTANGLES] = list_rectangles

    return {
        KW.MODULES: dict_modules,
        KW.NETS: list_nets,
    }
