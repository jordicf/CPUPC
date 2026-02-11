# (c) Jordi Cortadella 2026
# For the CPUPC Project.
# Licensed under the MIT License
# (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

import torch
from .fsconst import FsConstraint
from cpupc.netlist.module import Boundary
from cpupc.netlist.netlist import Netlist
from cpupc.geometry.fpolygon import Vertices


def write_netlist(
    net: Netlist,
) -> tuple[list[torch.Tensor], tuple[torch.Tensor, list[torch.Tensor]]]:
    """
    Writes a netlist in the floorset format
    :param net: input netlist
    :return: data and label tensors
    """

    # Dictionaries to names into indices
    mod2idx = _name2idx([m.name for m in net.modules if not m.is_iopin])
    pin2idx = _name2idx([m.name for m in net.modules if m.is_iopin])
    mib: dict[str, int] = dict()
    adjs: dict[str, int] = dict()

    # Calculate module centers
    net.calculate_centers_from_rectangles()
    assert all(m.center is not None for m in net.modules), "Some module has no center"

    for m in net.modules:
        if m.cluster is not None and m.cluster not in adjs:
            adjs[m.cluster] = len(adjs) + 1
        if m.mib is not None and m.mib not in mib:
            mib[m.mib] = len(mib) + 1

    num_blocks = len(mod2idx)
    num_pins = len(pin2idx)

    t_module = torch.zeros(num_blocks, 6)
    for m in net.modules:
        if m.is_iopin:
            continue
        i = mod2idx[m.name]
        t_module[i][FsConstraint.AREA] = m.area()
        t_module[i][FsConstraint.HARD] = 1.0 if m.is_hard and not m.is_fixed else 0.0
        t_module[i][FsConstraint.FIXED] = 1.0 if m.is_fixed else 0.0
        t_module[i][FsConstraint.MIB] = float(mib[m.mib]) if m.mib is not None else 0.0
        t_module[i][FsConstraint.ADJ_CLUSTER] = (
            float(adjs[m.cluster]) if m.cluster is not None else 0.0
        )
        t_module[i][FsConstraint.BOUNDARY] = (
            float(Boundary.code(m.boundary)) if m.boundary is not None else 0.0
        )

    num_hard = sum(
        1.0 if t_module[i][j] != 0 else 0
        for j in range(1, 6)
        for i in range(len(mod2idx))
    )

    t_pin = torch.zeros((num_pins, 2))
    for m in net.modules:
        if not m.is_iopin:
            continue
        i = pin2idx[m.name]
        assert m.center is not None, "Pin without position"
        t_pin[i][0] = m.center.x
        t_pin[i][1] = m.center.y

    t_b2b = torch.tensor([])
    t_p2b = torch.tensor([])
    hpwl_b2b, hpwl_p2b = 0.0, 0.0
    num_b2b, num_p2b = 0, 0
    for edge in net.edges:
        assert len(edge.modules) == 2
        src = edge.modules[0]
        dst = edge.modules[1]
        weight = edge.weight
        assert dst.name in mod2idx
        dst_idx = mod2idx[dst.name]
        if src.is_iopin:
            src_idx = pin2idx[src.name]
            t_p2b = torch.cat(
                (t_p2b, torch.tensor([[src_idx, dst_idx, weight]])), dim=0
            )
            hpwl_p2b += edge.hpwl
            num_p2b += 1
        else:
            src_idx = mod2idx[src.name]
            t_b2b = torch.cat(
                (t_b2b, torch.tensor([[src_idx, dst_idx, weight]])), dim=0
            )
            hpwl_b2b += edge.hpwl
            num_b2b += 1

    data: list[torch.Tensor] = [t_module, t_b2b, t_p2b, t_pin]

    # Polygons of the solution
    sol: list[torch.Tensor] = [torch.empty(0, 2)] * num_blocks
    all_vertices: Vertices = []
    for m in net.modules:
        if m.is_iopin:
            continue
        i = mod2idx[m.name]
        assert m.num_rectangles > 0, f"Module {m.name} has no rectangles"
        vertices = m.polygon().vertices()
        all_vertices.extend(vertices)
        vertices.append(vertices[0])  # close the polygon
        sol[i] = torch.tensor(vertices, dtype=torch.float32)

    xmin = min(v[0] for v in all_vertices)
    ymin = min(v[1] for v in all_vertices)
    xmax = max(v[0] for v in all_vertices)
    ymax = max(v[1] for v in all_vertices)

    area = (xmax - xmin) * (ymax - ymin)
    metrics = torch.tensor(
        [
            area,
            float(num_pins),
            float(num_b2b + num_p2b),
            float(num_b2b),
            float(num_p2b),
            float(num_hard),
            hpwl_b2b,
            hpwl_p2b,
        ]
    )

    label: tuple[torch.Tensor, list[torch.Tensor]] = (metrics, sol)

    return data, label


def _name2idx(names: list[str]) -> dict[str, int]:
    """Helper function to convert a list of names into a dictionary of name to index.
    The names are sorted by their numerical suffix (if it exists) and then by their name.
    """
    name_idx = [(name, _str_suffix(name)) for name in names]
    name_idx.sort(key=lambda x: (x[1], x[0]))
    return {name: i for i, (name, _) in enumerate(name_idx)}


def _str_suffix(s: str) -> int:
    """Helper function to extract the numerical suffix of a string.
    In case there is no numerical suffix, -1 is returned.
    For example, for "M_10" it returns 10, for "P_5" it returns 5, and for "M_fixed" it returns -1.
    """
    temp_idx = s.rfind(next(filter(lambda x: not x.isdigit(), s[::-1])))
    suffix = s[temp_idx + 1 :]
    return int(suffix) if suffix.isdigit() else -1
