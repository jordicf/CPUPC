# (c) Ylham Imam, 2026 — refactor
# For the CPUPC Project.
# Licensed under the MIT License (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

from __future__ import annotations

import sys
import math
import argparse
import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import casadi as ca

from cpupc.die.die import Die
from cpupc.netlist.netlist import Netlist
from cpupc.geometry.geometry import Rectangle

# Import matplotlib for visualization (optional)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# -------------------------
# Types
# -------------------------
BoxType = tuple[float, float, float, float]
InputModule = tuple[BoxType, list[BoxType], list[BoxType], list[BoxType], list[BoxType]]
OptionalList = dict[int, float]
OptionalMatrix = dict[int, OptionalList]
HyperEdge = tuple[float, list[int]]
HyperGraph = list[tuple[float, list[int], list[str]]]


def parse_options(
    prog: Optional[str] = None, args: Optional[list[str]] = None
) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="A tool for module legalization (CasADi)",
        usage="%(prog)s [options]",
    )
    parser.add_argument("netlist", type=str, help="Input netlist (.yaml)")
    parser.add_argument("die", type=str, help="Input die (.yaml)")
    parser.add_argument(
        "--min_aspect_ratio",
        type=float,
        default=1.0,
        help="Min aspect ratio (w/h lower bound)",
    )
    parser.add_argument("--max_ratio", type=float, default=3.0, help="Max aspect ratio")
    parser.add_argument("--num_iter", type=int, default=15, help="Number of iterations")
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="No-overlap distance radius multiplier",
    )
    parser.add_argument(
        "--wl_mult", type=float, default=1.0, help="HPWL weight multiplier"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--outfile", type=str, default=None, help="Output YAML")
    parser.add_argument("--palette_seed", type=int, default=None)
    parser.add_argument(
        "--tau_initial",
        type=float,
        default=None,
        help="Initial tau for soft constraints",
    )
    parser.add_argument(
        "--tau_decay", type=float, default=0.3, help="Tau decay factor per step"
    )
    parser.add_argument("--otol_initial", type=float, default=1e-1)
    parser.add_argument("--otol_final", type=float, default=1e-4)
    parser.add_argument("--rtol_initial", type=float, default=1e-1)
    parser.add_argument("--rtol_final", type=float, default=1e-4)
    parser.add_argument("--tol_decay", type=float, default=0.5)
    parser.add_argument(
        "--plot", action="store_true", help="Enable visualization of each iteration"
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="plots",
        help="Directory for saving plots (default: plots)",
    )
    return vars(parser.parse_args(args))


def netlist_to_utils(netlist: Netlist):
    ml: list[InputModule] = []
    al: list[float] = []
    xl: OptionalMatrix = {}
    yl: OptionalMatrix = {}
    wl: OptionalMatrix = {}
    hl: OptionalMatrix = {}
    mod_map: dict[str, int] = {}
    og_names: list[str] = []
    terminal_map: dict[str, tuple[float, float]] = {}
    min_ar_list: list[float] = []
    max_ar_list: list[float] = []

    # Terminals (io_pin): treat coordinates as constants
    # Following legalizer.py: terminals are is_iopin, not just is_fixed
    for module in netlist.modules:
        if module.is_iopin:
            if hasattr(module, "center") and module.center:
                if hasattr(module.center, "x") and hasattr(module.center, "y"):
                    terminal_map[module.name] = (
                        float(module.center.x),
                        float(module.center.y),
                    )
                else:
                    terminal_map[module.name] = (
                        float(module.center[0]),
                        float(module.center[1]),
                    )
            else:
                rect = module.rectangles[0]
                terminal_map[module.name] = (float(rect.center.x), float(rect.center.y))
            continue

        # Normal modules (including fixed and hard)
        mod_map[module.name] = len(ml)
        og_names.append(module.name)
        if module.aspect_ratio is not None:
            min_ar_list.append(float(module.aspect_ratio.min_wh))
            max_ar_list.append(float(module.aspect_ratio.max_wh))
        else:
            min_ar_list.append(float("nan"))
            max_ar_list.append(float("nan"))
        b: InputModule = ((0, 0, 0, 0), [], [], [], [])
        trunk_defined = False
        for rect in module.rectangles:
            r = (rect.center.x, rect.center.y, rect.shape.w, rect.shape.h)
            if rect.location == Rectangle.StropLocation.TRUNK:
                b = (r, b[1], b[2], b[3], b[4])
                trunk_defined = True
            elif rect.location == Rectangle.StropLocation.NORTH:
                b[1].append(r)
            elif rect.location == Rectangle.StropLocation.SOUTH:
                b[2].append(r)
            elif rect.location == Rectangle.StropLocation.EAST:
                b[3].append(r)
            elif rect.location == Rectangle.StropLocation.WEST:
                b[4].append(r)
            elif not trunk_defined:
                b = (r, b[1], b[2], b[3], b[4])
                trunk_defined = True
            else:
                b[1].append(r)
        # Following legalizer.py lines 1582-1602
        if module.is_hard:
            xl[len(ml)] = {}
            yl[len(ml)] = {}
            wl[len(ml)] = {}
            hl[len(ml)] = {}
        if module.is_fixed:
            # For fixed modules: store position for trunk
            if len(ml) not in xl:
                xl[len(ml)] = {}
                yl[len(ml)] = {}
            xl[len(ml)][0] = b[0][0]
            yl[len(ml)][0] = b[0][1]
        if module.is_hard:
            wl[len(ml)][0] = b[0][2]
            hl[len(ml)][0] = b[0][3]
            i = 1
            for q in range(1, 5):
                bq = b[q]
                if isinstance(bq, list):
                    for x, y, w, h in bq:
                        xl[len(ml)][i] = x
                        yl[len(ml)][i] = y
                        wl[len(ml)][i] = w
                        hl[len(ml)][i] = h
                        i += 1
        ml.append(b)
        al.append(module.area())

    hyper: HyperGraph = []
    for edge in netlist.edges:
        modules = []
        terminals = []
        weight = edge.weight
        for e_mod in edge.modules:
            if e_mod.name in mod_map:
                modules.append(mod_map[e_mod.name])
            elif e_mod.name in terminal_map:
                terminals.append(e_mod.name)
        if modules:
            hyper.append((weight, modules, terminals))

    return ml, al, xl, yl, wl, hl, hyper, og_names, terminal_map, min_ar_list, max_ar_list


def compute_options(options):
    die = Die(options["die"])
    die_width: float = die.width
    die_height: float = die.height
    netlist = Netlist(options["netlist"])
    ml, al, xl, yl, wl, hl, hyper, og_names, terminal_map, min_ar_list, max_ar_list = netlist_to_utils(netlist)
    return (
        ml,
        al,
        xl,
        yl,
        wl,
        hl,
        die_width,
        die_height,
        hyper,
        min_ar_list,
        max_ar_list,
        options["min_aspect_ratio"],
        options["max_ratio"],
        og_names,
        terminal_map,
    )


@dataclass
class ModuleVars:
    """
    Stores variables for all rectangles in a module (trunk + branches)
    Similar to ModelModule in legalizer.py
    """

    x: list[
        ca.SX
    ]  # x-coordinates of all rectangles [trunk, north_branches, south_branches, east_branches, west_branches]
    y: list[ca.SX]  # y-coordinates
    w: list[ca.SX]  # widths
    h: list[ca.SX]  # heights
    N: list[int]  # indices of North branches
    S: list[int]  # indices of South branches
    E: list[int]  # indices of East branches
    W: list[int]  # indices of West branches
    c: int  # total number of rectangles
    area_expr: ca.SX  # sum of all rectangle areas
    x_sum: ca.SX  # for center of mass calculation
    y_sum: ca.SX  # for center of mass calculation


class CasadiLegalizer:
    def __init__(
        self,
        ml: list[InputModule],
        al: list[float],
        xl: OptionalMatrix,
        yl: OptionalMatrix,
        wl: OptionalMatrix,
        hl: OptionalMatrix,
        die_width: float,
        die_height: float,
        hyper: HyperGraph,
        min_ar_list: list[float],
        max_ar_list: list[float],
        min_aspect_ratio: float,
        max_ratio: float,
        og_names: list[str],
        wl_mult: float,
        tau_initial: Optional[float],
        tau_decay: float,
        otol_initial: float,
        otol_final: float,
        rtol_initial: float,
        rtol_final: float,
        tol_decay: float,
        terminal_map: dict[str, tuple[float, float]],
        verbose: bool = False,
    ) -> None:
        self.ml = ml
        self.al = al
        self.xl = xl
        self.yl = yl
        self.wl = wl
        self.hl = hl
        self.dw = die_width
        self.dh = die_height
        self.hyper = hyper
        self.min_ar_list = min_ar_list
        self.max_ar_list = max_ar_list
        self.min_aspect_ratio = min_aspect_ratio
        self.max_ratio = max_ratio
        self.og_names = og_names
        self.wl_mult = wl_mult
        if tau_initial is not None:
            self.tau_initial = tau_initial
        else:
            # Default tau_initial: average module width * average module height
            # (computed from trunk rectangles in input ml).
            if len(ml) > 0:
                avg_w = sum(float(m[0][2]) for m in ml) / float(len(ml))
                avg_h = sum(float(m[0][3]) for m in ml) / float(len(ml))
                self.tau_initial = max(1e-9, avg_w * avg_h)
            else:
                self.tau_initial = 1.0
        self.tau_decay = tau_decay
        self.otol_initial = otol_initial
        self.otol_final = otol_final
        self.rtol_initial = rtol_initial
        self.rtol_final = rtol_final
        self.tol_decay = tol_decay
        self.terminal_map = terminal_map  # terminals as constants
        self._solver_verbose = verbose

        # Define symbolic variables once at initialization
        self.vars: list[ca.SX] = []
        self.lbx: list[float] = []
        self.ubx: list[float] = []
        self.x0: list[float] = []  # will be updated as warm-start
        self.modules: list[ModuleVars] = []
        self.module_has_variables: list[bool] = []
        # Per module / rect variable index layout for robust unpacking.
        # Each entry: {"x": idx|None, "y": idx|None, "w": idx|None, "h": idx|None}
        self.rect_var_layouts: list[list[dict[str, Optional[int]]]] = []
        self.module_kinds: list[str] = []  # "fixed" | "hard" | "soft"

        # Initialize module expressions / bounds for ALL rectangles
        # fixed: constants only; hard: x,y vars + constant w,h; soft: x,y,w,h vars.
        for idx, trunk_data in enumerate(self.ml):
            (x0, y0, w0, h0), Nb, Sb, Eb, Wb = trunk_data
            has_fixed_position = (
                idx in self.xl and 0 in self.xl[idx] and idx in self.yl and 0 in self.yl[idx]
            )
            has_fixed_dimensions = (
                idx in self.wl and 0 in self.wl[idx] and idx in self.hl and 0 in self.hl[idx]
            )
            if has_fixed_position:
                module_kind = "fixed"
            elif has_fixed_dimensions:
                module_kind = "hard"
            else:
                module_kind = "soft"
            self.module_kinds.append(module_kind)

            # Trunk (rectangle 0)
            rect_layouts: list[dict[str, Optional[int]]] = []
            trunk_x, trunk_y, trunk_w, trunk_h, trunk_layout = self._define_rect_by_kind(
                module_kind, idx, 0, (x0, y0, w0, h0)
            )
            rect_layouts.append(trunk_layout)
            x_list = [trunk_x]
            y_list = [trunk_y]
            w_list = [trunk_w]
            h_list = [trunk_h]

            N_indices = []
            S_indices = []
            E_indices = []
            W_indices = []

            rect_count = 1  # Start at 1 (trunk is 0)

            # Add North branches
            for bx, by, bw, bh in Nb:
                bx_var, by_var, bw_var, bh_var, bl = self._define_rect_by_kind(
                    module_kind, idx, rect_count, (bx, by, bw, bh)
                )
                x_list.append(bx_var)
                y_list.append(by_var)
                w_list.append(bw_var)
                h_list.append(bh_var)
                rect_layouts.append(bl)
                N_indices.append(rect_count)
                rect_count += 1

            # Add South branches
            for bx, by, bw, bh in Sb:
                bx_var, by_var, bw_var, bh_var, bl = self._define_rect_by_kind(
                    module_kind, idx, rect_count, (bx, by, bw, bh)
                )
                x_list.append(bx_var)
                y_list.append(by_var)
                w_list.append(bw_var)
                h_list.append(bh_var)
                rect_layouts.append(bl)
                S_indices.append(rect_count)
                rect_count += 1

            # Add East branches
            for bx, by, bw, bh in Eb:
                bx_var, by_var, bw_var, bh_var, bl = self._define_rect_by_kind(
                    module_kind, idx, rect_count, (bx, by, bw, bh)
                )
                x_list.append(bx_var)
                y_list.append(by_var)
                w_list.append(bw_var)
                h_list.append(bh_var)
                rect_layouts.append(bl)
                E_indices.append(rect_count)
                rect_count += 1

            # Add West branches
            for bx, by, bw, bh in Wb:
                bx_var, by_var, bw_var, bh_var, bl = self._define_rect_by_kind(
                    module_kind, idx, rect_count, (bx, by, bw, bh)
                )
                x_list.append(bx_var)
                y_list.append(by_var)
                w_list.append(bw_var)
                h_list.append(bh_var)
                rect_layouts.append(bl)
                W_indices.append(rect_count)
                rect_count += 1

            # Calculate total area and center of mass
            area_expr = ca.SX(0)
            x_sum = ca.SX(0)
            y_sum = ca.SX(0)
            for i in range(rect_count):
                rect_area = w_list[i] * h_list[i]
                area_expr = area_expr + rect_area
                x_sum = x_sum + x_list[i] * rect_area
                y_sum = y_sum + y_list[i] * rect_area

            self.modules.append(
                ModuleVars(
                    x=x_list,
                    y=y_list,
                    w=w_list,
                    h=h_list,
                    N=N_indices,
                    S=S_indices,
                    E=E_indices,
                    W=W_indices,
                    c=rect_count,
                    area_expr=area_expr,
                    x_sum=x_sum,
                    y_sum=y_sum,
                )
            )
            self.module_has_variables.append(module_kind != "fixed")
            self.rect_var_layouts.append(rect_layouts)

        self.has_decision_vars = len(self.vars) > 0
        self.x_sym = ca.vertcat(*self.vars) if self.vars else ca.SX.zeros(0, 1)

        # 优化四：只保留 tau 这一个参数，其他的全通过 lbx/ubx / lbg 传进去
        self.tau_param = ca.SX.sym("tau_param", 1)
        self.params = self.tau_param

        # ==============================================================
        # 优化一：在 __init__ 中一次性构建所有约束和目标函数 (Static NLP)
        # ==============================================================
        g: list[ca.SX] = []
        lbg: list[float] = []
        ubg: list[float] = []

        # 1. Bounds constraints (die) - only for modules with decision vars
        for mod_idx, m in enumerate(self.modules):
            if not self.module_has_variables[mod_idx]:
                continue
            for i in range(m.c):
                g.append(m.x[i] - 0.5 * m.w[i])
                lbg.append(0.0)
                ubg.append(float("inf"))
                g.append(m.y[i] - 0.5 * m.h[i])
                lbg.append(0.0)
                ubg.append(float("inf"))
                g.append(m.x[i] + 0.5 * m.w[i] - self.dw)
                lbg.append(float("-inf"))
                ubg.append(0.0)
                g.append(m.y[i] + 0.5 * m.h[i] - self.dh)
                lbg.append(float("-inf"))
                ubg.append(0.0)

        # 2. Minimal area requirements (soft modules only)
        for i, m in enumerate(self.modules):
            if self.al[i] > 1e-9 and self.module_kinds[i] == "soft":
                g.append(m.area_expr)
                lbg.append(self.al[i])
                ubg.append(float("inf"))

        # 3. Aspect ratio constraint (linear in w,h)
        for i, m in enumerate(self.modules):
            ar_min_i = self.min_aspect_ratio
            ar_max_i = self.max_ratio
            if i < len(self.min_ar_list) and i < len(self.max_ar_list):
                mn = self.min_ar_list[i]
                mx = self.max_ar_list[i]
                if not (math.isnan(mn) or math.isnan(mx)):
                    ar_min_i = float(mn)
                    ar_max_i = float(mx)
            ar_min_i = max(ar_min_i, 1e-8)
            ar_max_i = max(ar_max_i, ar_min_i)
            if self.module_kinds[i] != "soft":
                continue
            for rect_idx in range(m.c):
                g.append(m.h[rect_idx] - ar_max_i * m.w[rect_idx])
                lbg.append(float("-inf"))
                ubg.append(0.0)
                g.append(ar_min_i * m.w[rect_idx] - m.h[rect_idx])
                lbg.append(float("-inf"))
                ubg.append(0.0)

        # 3.5. Attachment constraints: branches must attach to trunk
        for mod_idx, m in enumerate(self.modules):
            if not self.module_has_variables[mod_idx]:
                continue
            trunk_x, trunk_y, trunk_w, trunk_h = m.x[0], m.y[0], m.w[0], m.h[0]

            for rect_idx in m.N:
                bx, by, bw, bh = (
                    m.x[rect_idx],
                    m.y[rect_idx],
                    m.w[rect_idx],
                    m.h[rect_idx],
                )
                g.append(by - (trunk_y + 0.5 * trunk_h + 0.5 * bh))
                lbg.append(0.0)
                ubg.append(0.0)
                g.append(bx - (trunk_x - 0.5 * trunk_w + 0.5 * bw))
                lbg.append(0.0)
                ubg.append(float("inf"))
                g.append((trunk_x + 0.5 * trunk_w - 0.5 * bw) - bx)
                lbg.append(0.0)
                ubg.append(float("inf"))

            for rect_idx in m.S:
                bx, by, bw, bh = (
                    m.x[rect_idx],
                    m.y[rect_idx],
                    m.w[rect_idx],
                    m.h[rect_idx],
                )
                g.append(by - (trunk_y - 0.5 * trunk_h - 0.5 * bh))
                lbg.append(0.0)
                ubg.append(0.0)
                g.append(bx - (trunk_x - 0.5 * trunk_w + 0.5 * bw))
                lbg.append(0.0)
                ubg.append(float("inf"))
                g.append((trunk_x + 0.5 * trunk_w - 0.5 * bw) - bx)
                lbg.append(0.0)
                ubg.append(float("inf"))

            for rect_idx in m.E:
                bx, by, bw, bh = (
                    m.x[rect_idx],
                    m.y[rect_idx],
                    m.w[rect_idx],
                    m.h[rect_idx],
                )
                g.append(bx - (trunk_x + 0.5 * trunk_w + 0.5 * bw))
                lbg.append(0.0)
                ubg.append(0.0)
                g.append(by - (trunk_y - 0.5 * trunk_h + 0.5 * bh))
                lbg.append(0.0)
                ubg.append(float("inf"))
                g.append((trunk_y + 0.5 * trunk_h - 0.5 * bh) - by)
                lbg.append(0.0)
                ubg.append(float("inf"))

            for rect_idx in m.W:
                bx, by, bw, bh = (
                    m.x[rect_idx],
                    m.y[rect_idx],
                    m.w[rect_idx],
                    m.h[rect_idx],
                )
                g.append(bx - (trunk_x - 0.5 * trunk_w - 0.5 * bw))
                lbg.append(0.0)
                ubg.append(0.0)
                g.append(by - (trunk_y - 0.5 * trunk_h + 0.5 * bh))
                lbg.append(0.0)
                ubg.append(float("inf"))
                g.append((trunk_y + 0.5 * trunk_h - 0.5 * bh) - by)
                lbg.append(0.0)
                ubg.append(float("inf"))

        # 3.6. Intra-module non-overlap
        for mod_idx, m in enumerate(self.modules):
            if not self.module_has_variables[mod_idx]:
                continue
            if len(m.N) > 1:
                for ii in range(len(m.N) - 1):
                    idx1, idx2 = m.N[ii], m.N[ii + 1]
                    g.append(m.x[idx1] + 0.5 * m.w[idx1] - m.x[idx2] + 0.5 * m.w[idx2])
                    lbg.append(float("-inf"))
                    ubg.append(0.0)

            if len(m.S) > 1:
                for ii in range(len(m.S) - 1):
                    idx1, idx2 = m.S[ii], m.S[ii + 1]
                    g.append(m.x[idx1] + 0.5 * m.w[idx1] - m.x[idx2] + 0.5 * m.w[idx2])
                    lbg.append(float("-inf"))
                    ubg.append(0.0)

            if len(m.E) > 1:
                for ii in range(len(m.E) - 1):
                    idx1, idx2 = m.E[ii], m.E[ii + 1]
                    g.append(m.y[idx1] + 0.5 * m.h[idx1] - m.y[idx2] + 0.5 * m.h[idx2])
                    lbg.append(float("-inf"))
                    ubg.append(0.0)

            if len(m.W) > 1:
                for ii in range(len(m.W) - 1):
                    idx1, idx2 = m.W[ii], m.W[ii + 1]
                    g.append(m.y[idx1] + 0.5 * m.h[idx1] - m.y[idx2] + 0.5 * m.h[idx2])
                    lbg.append(float("-inf"))
                    ubg.append(0.0)

        # 4. Inter-module non-overlap: 静态添加所有模块对
        def smax(a: ca.SX, b: ca.SX, tau: ca.SX) -> ca.SX:
            return 0.5 * (a + b + ca.sqrt((a - b) * (a - b) + 4 * tau * tau))

        self.overlap_g_indices: dict[int, list[int]] = {}
        pair_idx_counter = 0
        inter_module_constraints_added = 0
        fixed_fixed_pairs_skipped = 0

        for i in range(len(self.modules)):
            for j in range(i + 1, len(self.modules)):
                # Both modules are fixed constants: no need to add overlap constraints.
                # Keep pair index mapping with an empty list to preserve active-mask indexing.
                if (not self.module_has_variables[i]) and (not self.module_has_variables[j]):
                    self.overlap_g_indices[pair_idx_counter] = []
                    pair_idx_counter += 1
                    fixed_fixed_pairs_skipped += 1
                    continue

                mi, mj = self.modules[i], self.modules[j]
                rect_g_indices: list[int] = []
                for rect_i in range(mi.c):
                    for rect_j in range(mj.c):
                        xi, yi, wi, hi = (
                            mi.x[rect_i],
                            mi.y[rect_i],
                            mi.w[rect_i],
                            mi.h[rect_i],
                        )
                        xj, yj, wj, hj = (
                            mj.x[rect_j],
                            mj.y[rect_j],
                            mj.w[rect_j],
                            mj.h[rect_j],
                        )
                        t1 = (xi - xj) * (xi - xj) - 0.25 * (wi + wj) * (wi + wj)
                        t2 = (yi - yj) * (yi - yj) - 0.25 * (hi + hj) * (hi + hj)
                        g.append(smax(t1, t2, self.tau_param))
                        rect_g_indices.append(len(g) - 1)
                        lbg.append(0.0)
                        ubg.append(float("inf"))
                        inter_module_constraints_added += 1
                self.overlap_g_indices[pair_idx_counter] = rect_g_indices
                pair_idx_counter += 1

        if self._solver_verbose:
            print("[Section 4] Inter-module non-overlap (static, all pairs):")
            print(f"  - Total module pairs: {pair_idx_counter}")
            print(f"  - Fixed-fixed pairs skipped: {fixed_fixed_pairs_skipped}")
            print(f"  - Total rectangle-pair constraints: {inter_module_constraints_added}")

        # 0. Module type summary (constraints removed for constants by construction)
        if self._solver_verbose:
            n_fixed = sum(1 for k in self.module_kinds if k == "fixed")
            n_hard = sum(1 for k in self.module_kinds if k == "hard")
            n_soft = sum(1 for k in self.module_kinds if k == "soft")
            print(f"[Section 0] Module kinds: fixed={n_fixed}, hard={n_hard}, soft={n_soft}")

        # Objective: LSE(HPWL) with terminals as constants
        def stable_logsumexp(vals: list[ca.SX], a: float) -> ca.SX:
            ax = [a * v for v in vals]
            mloc = ax[0]
            for v in ax[1:]:
                mloc = ca.fmax(mloc, v)
            s = 0
            for v in ax:
                s = s + ca.exp(v - mloc)
            return mloc + ca.log(s)

        alpha = 1.0
        obj_terms: list[ca.SX] = []

        def module_center(m: ModuleVars) -> tuple[ca.SX, ca.SX]:
            area_safe = m.area_expr + 1e-10
            cx = m.x_sum / area_safe
            cy = m.y_sum / area_safe
            return cx, cy

        for weight, vertices, terminals in self.hyper:
            pts_x: list[ca.SX] = []
            pts_y: list[ca.SX] = []
            for vi in vertices:
                cx, cy = module_center(self.modules[vi])
                pts_x.append(cx)
                pts_y.append(cy)
            for tname in terminals:
                if tname in self.terminal_map:
                    tx, ty = self.terminal_map[tname]
                    pts_x.append(ca.DM(tx))
                    pts_y.append(ca.DM(ty))
            if len(pts_x) < 2:
                continue
            lse_x = (
                stable_logsumexp(pts_x, alpha)
                + stable_logsumexp([-v for v in pts_x], alpha)
            ) / alpha
            lse_y = (
                stable_logsumexp(pts_y, alpha)
                + stable_logsumexp([-v for v in pts_y], alpha)
            ) / alpha
            obj_terms.append((lse_x + lse_y) * (weight * self.wl_mult))

        if obj_terms:
            f = ca.sum1(ca.vertcat(*obj_terms))
        else:
            f = ca.SX(0)

        self._lbg_base = lbg
        self._ubg_base = ubg
        self.g_sym = ca.vertcat(*g) if g else ca.SX.zeros(0, 1)
        self.f_sym = f
        self.nlp = {"x": self.x_sym, "p": self.params, "f": self.f_sym, "g": self.g_sym}

        if self.has_decision_vars:
            self.solver = ca.nlpsol(
                "solver",
                "ipopt",
                self.nlp,
                {
                    "ipopt.print_level": 0 if not self._solver_verbose else 5,
                    "print_time": 0 if not self._solver_verbose else 1,
                    "ipopt.tol": otol_initial,
                    "ipopt.acceptable_tol": otol_initial * 10,
                    "ipopt.constr_viol_tol": rtol_initial,
                    "ipopt.acceptable_constr_viol_tol": rtol_initial * 10,
                    "ipopt.warm_start_init_point": "yes",
                    "ipopt.mu_strategy": "adaptive",
                    "ipopt.linear_solver": "mumps",
                    #"ipopt.mu_oracle": "quality-function",
                    "ipopt.max_iter": 20,
                    "ipopt.hessian_approximation": "limited-memory",
                    "expand": False,
                },
            )
            self.const_eval = None
        else:
            self.solver = None
            self.const_eval = ca.Function("const_eval", [self.params], [self.f_sym, self.g_sym])
        self.nlp_initialized = True

    def _new_scalar_var(
        self, name: str, lb: float, ub: float, init: float
    ) -> tuple[ca.SX, int]:
        """Create one scalar decision variable with bounds/init and return (sym, index)."""
        idx = len(self.vars)
        v = ca.SX.sym(name, 1)
        self.vars.append(v)
        self.lbx.append(float(lb))
        self.ubx.append(float(ub))
        self.x0.append(float(init))
        return v, idx

    def _define_rect_consts(
        self, rect: BoxType
    ) -> tuple[ca.SX, ca.SX, ca.SX, ca.SX, dict[str, Optional[int]]]:
        cx, cy, w, h = rect
        return (
            ca.SX(float(cx)),
            ca.SX(float(cy)),
            ca.SX(float(max(w, 0.1))),
            ca.SX(float(max(h, 0.1))),
            {"x": None, "y": None, "w": None, "h": None},
        )

    def _hard_rect_dims(self, module_idx: int, rect_idx: int, rect: BoxType) -> tuple[float, float]:
        """Return fixed dimensions for hard module rect, falling back to geometry if absent."""
        _, _, rw, rh = rect
        wv = self.wl.get(module_idx, {}).get(rect_idx, rw)
        hv = self.hl.get(module_idx, {}).get(rect_idx, rh)
        return float(max(wv, 0.1)), float(max(hv, 0.1))

    def _define_rect_by_kind(
        self, kind: str, module_idx: int, rect_idx: int, rect: BoxType
    ) -> tuple[ca.SX, ca.SX, ca.SX, ca.SX, dict[str, Optional[int]]]:
        """
        Build one rectangle according to module kind:
        - fixed: constants x,y,w,h
        - hard : vars x,y; constants w,h
        - soft : vars x,y,w,h
        """
        cx, cy, rw, rh = rect
        if kind == "fixed":
            return self._define_rect_consts(rect)

        x, x_idx = self._new_scalar_var(
            f"x_m{module_idx}_r{rect_idx}", 0.0, self.dw, float(cx)
        )
        y, y_idx = self._new_scalar_var(
            f"y_m{module_idx}_r{rect_idx}", 0.0, self.dh, float(cy)
        )

        if kind == "hard":
            wv, hv = self._hard_rect_dims(module_idx, rect_idx, rect)
            w = ca.SX(wv)
            h = ca.SX(hv)
            return x, y, w, h, {"x": x_idx, "y": y_idx, "w": None, "h": None}

        # soft
        w, w_idx = self._new_scalar_var(
            f"w_m{module_idx}_r{rect_idx}", 0.1, self.dw, float(max(rw, 0.1))
        )
        h, h_idx = self._new_scalar_var(
            f"h_m{module_idx}_r{rect_idx}", 0.1, self.dh, float(max(rh, 0.1))
        )
        return x, y, w, h, {"x": x_idx, "y": y_idx, "w": w_idx, "h": h_idx}

    def solve(
        self,
        x0: list[float],
        current_tau_val: float,
        active_mask_vals: list[float],
        prev_vals: Optional[list[float]],
        step_cap_val: Optional[float],
        otol: float,
        rtol: float,
        verbose: bool = False,
    ) -> dict[str, Any]:
        del verbose  # IPOPT verbosity fixed at construction; kept for call-site compatibility

        if not self.has_decision_vars:
            f_val, g_val = self.const_eval(float(current_tau_val))
            return {"x": ca.DM.zeros(0, 1), "f": f_val, "g": g_val}

        dynamic_lbx = list(self.lbx)
        dynamic_ubx = list(self.ubx)
        if prev_vals is not None and step_cap_val is not None and step_cap_val > 0:
            for k in range(len(self.vars)):
                pv = float(prev_vals[k])
                dynamic_lbx[k] = max(float(dynamic_lbx[k]), pv - step_cap_val)
                dynamic_ubx[k] = min(float(dynamic_ubx[k]), pv + step_cap_val)

        dynamic_lbg = list(self._lbg_base)
        for pair_idx, is_active in enumerate(active_mask_vals):
            if is_active <= 0.5:
                for idx in self.overlap_g_indices[pair_idx]:
                    dynamic_lbg[idx] = float("-inf")

        sol = self.solver(
            x0=x0,
            lbx=dynamic_lbx,
            ubx=dynamic_ubx,
            lbg=dynamic_lbg,
            ubg=self._ubg_base,
            p=current_tau_val,
        )
        return {"x": sol["x"], "f": sol["f"], "g": sol.get("g", None)}

    @property
    def _lbg(self) -> list[float]:
        return self._lbg_base

    @property
    def _ubg(self) -> list[float]:
        return self._ubg_base

    # Dummy helper (kept for interface completeness)
    def _numeric(self, *_, **__) -> list[float]:
        return []


def visualize_iteration(
    ml: list[InputModule],
    og_names: list[str],
    terminal_map: dict[str, tuple[float, float]],
    die_width: float,
    die_height: float,
    iteration: int,
    hpwl: float,
    overlap: float,
    output_dir: str,
    tau: float,
) -> None:
    """Visualize current layout with HPWL and overlap information"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  matplotlib not available, skipping visualization")
        return

    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw die boundary
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            die_width,
            die_height,
            fill=False,
            edgecolor="darkblue",
            linewidth=3,
            linestyle="--",
        )
    )

    # Color palette for modules
    colors = plt.cm.tab20(range(20))
    color_idx = 0

    # Draw all modules (including rectilinear)
    for mod_idx, (trunk, Nb, Sb, Eb, Wb) in enumerate(ml):
        module_name = og_names[mod_idx] if mod_idx < len(og_names) else f"M{mod_idx}"
        color = colors[color_idx % len(colors)]
        color_idx += 1

        # Draw all rectangles of this module
        all_rects = [trunk] + Nb + Sb + Eb + Wb
        for rect_idx, (cx, cy, w, h) in enumerate(all_rects):
            x_left = cx - w / 2
            y_bottom = cy - h / 2

            rect = patches.Rectangle(
                (x_left, y_bottom),
                w,
                h,
                facecolor=color,
                edgecolor="black",
                linewidth=1.5 if rect_idx == 0 else 1.0,  # Thicker border for trunk
                alpha=0.7,
            )
            ax.add_patch(rect)

            # Add label only on trunk
            if rect_idx == 0 and w > 0.3 and h > 0.3:
                ax.text(
                    cx,
                    cy,
                    module_name,
                    ha="center",
                    va="center",
                    fontsize=8,
                    weight="bold",
                    color="white" if sum(color[:3]) < 1.5 else "black",
                )

    # Draw terminals
    for term_name, (tx, ty) in terminal_map.items():
        ax.plot(
            tx, ty, "ro", markersize=10, markeredgecolor="darkred", markeredgewidth=2
        )
        ax.text(
            tx,
            ty + 0.5,
            term_name,
            ha="center",
            va="bottom",
            fontsize=7,
            color="red",
            weight="bold",
        )

    # Set limits and aspect
    ax.set_xlim(-0.5, die_width + 0.5)
    ax.set_ylim(-0.5, die_height + 0.5)
    ax.set_aspect("equal")

    # Grid
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel("X coordinate", fontsize=10)
    ax.set_ylabel("Y coordinate", fontsize=10)

    # Title with statistics
    ax.set_title(
        # f"Iteration {iteration}\n"
        f"HPWL: {hpwl:.2f}, Overlap: {overlap:.4f}, τ: {tau:.2e}\n"
        f"Die: {die_width:.1f} × {die_height:.1f}, Modules: {len(ml)}, Terminals: {len(terminal_map)}",
        fontsize=12,
        pad=20,
    )

    # Save figure
    output_file = os.path.join(output_dir, f"iter_{iteration:03d}.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Plot saved: {output_file}")


def main(prog: Optional[str] = None, args: Optional[list[str]] = None) -> int:
    options = parse_options(prog, args)

    # Check if plotting is enabled
    plot_enabled = options.get("plot", False)
    if plot_enabled and not MATPLOTLIB_AVAILABLE:
        print("⚠️  Warning: --plot enabled but matplotlib not available")
        print("    Install matplotlib: pip install matplotlib")
        plot_enabled = False

    # Create plot directory if needed
    if plot_enabled:
        plot_dir = options.get("plot_dir", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        print(f"📁 Plot directory: {plot_dir}")

    (
        ml,
        al,
        xl,
        yl,
        wl,
        hl,
        die_width,
        die_height,
        hyper,
        min_ar_list,
        max_ar_list,
        min_aspect_ratio,
        max_ratio,
        og_names,
        terminal_map,
    ) = compute_options(options)

    # Count module types
    n_fixed = 0
    n_hard = 0
    n_soft = 0
    for i in range(len(ml)):
        has_fixed_pos = i in xl and 0 in xl[i] and i in yl and 0 in yl[i]
        has_fixed_dim = i in wl and 0 in wl[i] and i in hl and 0 in hl[i]
        if has_fixed_pos:
            n_fixed += 1
        elif has_fixed_dim:
            n_hard += 1
        else:
            n_soft += 1
    
    # Count nets with custom weights
    n_weighted = sum(1 for (w, _, _) in hyper if abs(w - 1.0) > 1e-9)
    
    print(f"Module types: {n_soft} soft, {n_hard} hard, {n_fixed} fixed")
    print(f"Terminals: {len(terminal_map)}")
    print(f"Nets: {len(hyper)} ({n_weighted} with custom weights)")

    last_x = None
    die_area = die_width * die_height

    # Calculate total module area for overlap threshold
    total_module_area = sum(al)  # al contains all module areas
    overlap_threshold = 1e-5 # 1% of total module area
    #overlap_threshold = 1e-5

    def compute_total_overlap(ml: list[InputModule]) -> float:
        """Compute total overlap area between all module pairs"""
        total_overlap = 0.0
        for mi in range(len(ml)):
            for mj in range(mi + 1, len(ml)):
                bi = ml[mi][0]  # trunk box: (cx, cy, w, h)
                bj = ml[mj][0]

                cx1, cy1, w1, h1 = bi
                cx2, cy2, w2, h2 = bj
                left1 = cx1 - 0.5 * w1
                right1 = cx1 + 0.5 * w1
                bottom1 = cy1 - 0.5 * h1
                top1 = cy1 + 0.5 * h1
                left2 = cx2 - 0.5 * w2
                right2 = cx2 + 0.5 * w2
                bottom2 = cy2 - 0.5 * h2
                top2 = cy2 + 0.5 * h2

                overlap_left = max(left1, left2)
                overlap_right = min(right1, right2)
                overlap_bottom = max(bottom1, bottom2)
                overlap_top = min(top1, top2)
                if overlap_left < overlap_right and overlap_bottom < overlap_top:
                    overlap_area = (overlap_right - overlap_left) * (
                        overlap_top - overlap_bottom
                    )
                    total_overlap += overlap_area
        return total_overlap

    init_tau = options.get("tau_initial")
    if init_tau is None:
        init_tau = 1e-3

    model = CasadiLegalizer(
        ml,
        al,
        xl,
        yl,
        wl,
        hl,
        die_width,
        die_height,
        hyper,
        min_ar_list,
        max_ar_list,
        min_aspect_ratio,
        max_ratio,
        og_names,
        wl_mult=options["wl_mult"],
        tau_initial=float(init_tau),
        tau_decay=1.0,
        otol_initial=options.get("otol_initial", 1e-1),
        otol_final=options.get("otol_final", 1e-4),
        rtol_initial=options.get("rtol_initial", 1e-1),
        rtol_final=options.get("rtol_final", 1e-4),
        tol_decay=options.get("tol_decay", 0.5),
        terminal_map=terminal_map,
        verbose=options["verbose"],
    )

    for i in range(options["num_iter"]):

        base_tau = options.get("tau_initial", None)
        if base_tau is None:
            base_tau = 1e-3
        tau_decay = options.get("tau_decay", 0.7)

        current_tau = float(base_tau * (tau_decay ** float(i)))

        if options["verbose"]:
            print(f"Iteration {i+1}/{options['num_iter']}: tau = {current_tau:.6e}")

        max_dim = max(die_width, die_height)
        radius_val = float(options.get("radius", 1.0))
        radius_val = max(0.0, min(1.0, radius_val))
        dist_threshold = radius_val * max_dim

        def rect_from_box(b: BoxType) -> tuple[float, float, float, float]:
            cx, cy, w, h = b
            return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)

        def l1_gap(b1: BoxType, b2: BoxType) -> float:
            x1a, y1a, x2a, y2a = rect_from_box(b1)
            x1b, y1b, x2b, y2b = rect_from_box(b2)
            dx = max(0.0, max(x1a, x1b) - min(x2a, x2b))
            dy = max(0.0, max(y1a, y1b) - min(y2a, y2b))

            if dx == 0.0 and dy == 0.0:
                return 0.0
            return dx + dy

        active_mask_vals: list[float] = []
        pair_idx_counter = 0
        for mi in range(len(ml)):
            for mj in range(mi + 1, len(ml)):
                bi = ml[mi][0]  # trunk box
                bj = ml[mj][0]
                if l1_gap(bi, bj) <= dist_threshold:
                    active_mask_vals.append(1.0)  # active
                else:
                    active_mask_vals.append(0.0)  # inactive
                pair_idx_counter += 1

        # Compute trust-region step size: proportional to die dimension, gradually decreasing
        #step_cap = 1.0* max_dim * (1.0 - 0.6 * (i / max(1, options["num_iter"] - 1)))
        step_cap = 0.333*radius_val* max_dim 

        current_otol = max(
            options.get("otol_initial", 1e-1) * (options.get("tol_decay", 0.5) ** i),
            options.get("otol_final", 1e-4),
        )
        current_rtol = max(
            options.get("rtol_initial", 1e-1) * (options.get("tol_decay", 0.5) ** i),
            options.get("rtol_final", 1e-4),
        )
        if options["verbose"]:
            print(f"    OTOL={current_otol:.6e}, RTOL={current_rtol:.6e}")

        if last_x is not None:
            model.x0 = list(map(float, last_x))

        # Solve optimization problem
        sol = model.solve(
            x0=model.x0,  # Use updated x0 (warm start if available)
            current_tau_val=current_tau,
            active_mask_vals=active_mask_vals,
            prev_vals=(list(map(float, last_x)) if last_x is not None else None),
            step_cap_val=step_cap if last_x is not None else None,
            otol=current_otol,
            rtol=current_rtol,
            verbose=options["verbose"],
        )
        x = sol["x"].full().flatten()
        fval = (
            float(sol["f"])
            if hasattr(sol["f"], "__float__")
            else float(sol["f"].full().item())
        )

        prev_x = last_x
        last_x = x

        # Extract solution and update ml for next iteration
        # Robust mapping via recorded per-rectangle variable layout.
        new_ml: list[InputModule] = []
        for mod_idx, (trunk, Nb, Sb, Eb, Wb) in enumerate(ml):
            m = model.modules[mod_idx]
            layouts = model.rect_var_layouts[mod_idx]
            old_rects: list[BoxType] = [trunk] + list(Nb) + list(Sb) + list(Eb) + list(Wb)
            new_rects: list[BoxType] = []
            for rect_idx in range(m.c):
                old_rx, old_ry, old_rw, old_rh = old_rects[rect_idx]
                lay = layouts[rect_idx]
                rx = float(x[lay["x"]]) if lay["x"] is not None else float(old_rx)
                ry = float(x[lay["y"]]) if lay["y"] is not None else float(old_ry)
                rw = float(x[lay["w"]]) if lay["w"] is not None else float(old_rw)
                rh = float(x[lay["h"]]) if lay["h"] is not None else float(old_rh)
                new_rects.append((rx, ry, rw, rh))

            # Reconstruct (trunk, Nb, Sb, Eb, Wb) structure
            new_trunk = new_rects[0]
            new_Nb = [new_rects[idx] for idx in m.N]
            new_Sb = [new_rects[idx] for idx in m.S]
            new_Eb = [new_rects[idx] for idx in m.E]
            new_Wb = [new_rects[idx] for idx in m.W]

            new_ml.append((new_trunk, new_Nb, new_Sb, new_Eb, new_Wb))

        ml = new_ml

        current_overlap = compute_total_overlap(ml)

        print(
            f"Iteration {i+1}/{options['num_iter']}: tau = {current_tau:.6e}, objective = {fval:.6f}, overlap = {current_overlap:.6f}"
        )

        if options["verbose"]:
            print(f"    Overlap threshold = {overlap_threshold:.6f}")

        # Visualization if enabled
        if plot_enabled:
            visualize_iteration(
                ml=ml,
                og_names=og_names,
                terminal_map=terminal_map,
                die_width=die_width,
                die_height=die_height,
                iteration=i + 1,
                hpwl=fval,
                overlap=current_overlap,
                output_dir=options.get("plot_dir", "plots"),
                tau=current_tau,
            )

        # Iteration stop criterion: stop when overlap area < total_module_area * 0.01
        if current_overlap < overlap_threshold:
            print(
                f"Early termination: overlap area {current_overlap:.6f} < threshold {overlap_threshold:.6f} (total_module_area * 0.01)"
            )
            break

    # Write output YAML with updated centers/sizes of ALL rectangles
    if options["outfile"] is not None:
        net = Netlist(options["netlist"])
        ml_iter = 0
        for m in net.modules:
            # Skip terminals (is_iopin), not just fixed
            if m.is_iopin:
                continue

            # Get corresponding structure from ml
            trunk, Nb, Sb, Eb, Wb = ml[ml_iter]
            mod_idx = ml_iter
            ml_iter += 1

            # Update all rectangles for this module from final ml geometry.
            if m.rectangles:
                # Update trunk (rectangles[0])
                tx, ty, tw, th = trunk
                m.rectangles[0].center.x = tx
                m.rectangles[0].center.y = ty
                m.rectangles[0].shape.w = tw
                m.rectangles[0].shape.h = th

                # Update branches in order (Nb + Sb + Eb + Wb)
                k = 1  # Next rectangle index
                for group in (Nb, Sb, Eb, Wb):
                    for (gx, gy, gw, gh) in group:
                        if k < len(m.rectangles):
                            m.rectangles[k].center.x = gx
                            m.rectangles[k].center.y = gy
                            m.rectangles[k].shape.w = gw
                            m.rectangles[k].shape.h = gh
                            k += 1

        net.write_yaml(options["outfile"])

    return 0


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    sys.exit(main())
