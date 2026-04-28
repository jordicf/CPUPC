#!/usr/bin/env python3
r"""
Mixed-Space Convex Floorplan Optimisation  (CasADi / IPOPT)
============================================================

Variable encoding — **hybrid linear + log space**:

  Linear space :  x_i, y_i          (module center coordinates)
  Log    space :  W_i = ln(w_i),  H_i = ln(h_i)    (log of dimensions)
"""

import yaml
import numpy as np
from scipy.spatial import Delaunay
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import sys
import networkx as nx

sys.setrecursionlimit(10000)


def load_netlist(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_die_info(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)



def _parse_module_ar_bounds(
    info: dict,
    module_name: str,
    default_min_ar: float | None,
    default_max_ar: float | None,
) -> tuple[float, float]:
    """
    Parse per-module aspect_ratio from netlist info (soft modules only).
    Fallback to CLI defaults when absent or malformed.
    Fixed and hard modules do not use aspect_ratio in this flow; callers skip this.
    """
    ar = info.get("aspect_ratio", None)
    if ar is None:
        if default_min_ar is None or default_max_ar is None:
            raise ValueError(
                f"Module {module_name} has no aspect_ratio in netlist, and CLI "
                "fallback (--min-aspect-ratio/--max-ratio) is not fully specified."
            )
        return float(default_min_ar), float(default_max_ar)

    if isinstance(ar, (int, float)):
        v = float(ar)
        if v <= 0:
            if default_min_ar is None or default_max_ar is None:
                raise ValueError(
                    f"Module {module_name} has invalid scalar aspect_ratio={ar}, and CLI "
                    "fallback (--min-aspect-ratio/--max-ratio) is not fully specified."
                )
            return float(default_min_ar), float(default_max_ar)
        inv = 1.0 / v
        return float(min(v, inv)), float(max(v, inv))

    if isinstance(ar, (list, tuple)) and len(ar) == 2:
        a0, a1 = ar[0], ar[1]
        if isinstance(a0, (int, float)) and isinstance(a1, (int, float)):
            ar_min = float(a0)
            ar_max = float(a1)
            if ar_min > 0 and ar_max > 0 and ar_min <= ar_max:
                return ar_min, ar_max

    if default_min_ar is None or default_max_ar is None:
        raise ValueError(
            f"Module {module_name} has invalid aspect_ratio={ar}, and CLI "
            "fallback (--min-aspect-ratio/--max-ratio) is not fully specified."
        )
    return float(default_min_ar), float(default_max_ar)


def extract_modules_and_terminals(
    netlist_data,
    default_min_ar: float | None = None,
    default_max_ar: float | None = None,
):
    """
    Returns
    -------
    movable_modules : list[str]  – modules with fixed=False (includes hard)
    hard_modules : list[str]     – modules with hard=True and fixed=False
    fixed_modules : list[str]    – modules with fixed=True
    module_data : dict  – {name: {area, x, y, w, h, hard, fixed, ar_min?, ar_max?}};
        ar_min/ar_max only for soft (non-fixed, non-hard) modules.
    terminal_coords : dict  – {name: (x, y)}
    """
    movable_modules = []
    hard_modules = []
    fixed_modules = []
    module_data = {}
    terminal_coords = {}

    modules_dict = netlist_data.get('Modules', netlist_data)

    for name, info in modules_dict.items():
        if not isinstance(info, dict):
            continue

        is_terminal = info.get('terminal', False) or info.get('io_pin', False)
        if is_terminal:
            rects = info.get('rectangles', [[0, 0, 0, 0]])
            if rects and len(rects[0]) >= 2:
                terminal_coords[name] = (float(rects[0][0]), float(rects[0][1]))
            elif 'center' in info:
                c = info['center']
                terminal_coords[name] = (float(c[0]), float(c[1]))
            continue

        area = info.get('area', 0)
        rects = info.get('rectangles', [[0, 0, 1, 1]])
        if rects and len(rects[0]) == 4:
            x, y, w, h = [float(v) for v in rects[0]]
        else:
            x, y, w, h = 0.0, 0.0, 1.0, 1.0

        if area <= 0 and w > 0 and h > 0:
            area = w * h

        is_fixed = info.get('fixed', False)
        is_hard = info.get('hard', False) and not is_fixed
        entry = {
            'area': area, 'x': x, 'y': y, 'w': w, 'h': h,
            'hard': is_hard,
            'fixed': is_fixed,
        }
        # Soft only: aspect ratio from netlist / CLI. Fixed/hard: no ar fields.
        if not is_fixed and not is_hard:
            ar_min, ar_max = _parse_module_ar_bounds(
                info, name, default_min_ar, default_max_ar
            )
            entry['ar_min'] = ar_min
            entry['ar_max'] = ar_max
        module_data[name] = entry
        
        if is_fixed:
            fixed_modules.append(name)
        else:
            movable_modules.append(name)
            if is_hard:
                hard_modules.append(name)

    return movable_modules, hard_modules, fixed_modules, module_data, terminal_coords


def extract_nets(netlist_data, movable_modules, fixed_modules, module_data, terminal_coords):
    mod_idx = {name: i for i, name in enumerate(movable_modules)}
    fixed_coords = {
        name: (float(module_data[name]['x']), float(module_data[name]['y']))
        for name in fixed_modules
        if name in module_data
    }
    nets = []

    for net in netlist_data.get('Nets', []):
        if not isinstance(net, (list, tuple)):
            continue

        weight = 1.0
        pins = list(net)
        if (len(pins) >= 2
                and isinstance(pins[-1], (int, float))
                and not isinstance(pins[-1], str)):
            weight = float(pins.pop())

        mod_ids = []
        term_pos = []
        for p in pins:
            p = str(p)
            if p in mod_idx:
                mod_ids.append(mod_idx[p])
            elif p in terminal_coords:
                term_pos.append(terminal_coords[p])
            elif p in fixed_coords:
                term_pos.append(fixed_coords[p])

        if len(mod_ids) + len(term_pos) >= 2:
            nets.append((weight, mod_ids, term_pos))

    return nets




def break_cycles(edges, n, module_data, modules, axis='h', verbose=False):
    G = nx.DiGraph(edges)
    total_removed = 0
    iteration = 0

    while True:
        try:
            cycle_edges = nx.find_cycle(G, orientation='original')
            iteration += 1

            best_slack = -float('inf')
            edge_to_remove = None

            for edge in cycle_edges:
                u, v = edge[:2]
                di = module_data[modules[u]]
                dj = module_data[modules[v]]

                wi = di.get('w', di.get('width', 0.0))
                hi = di.get('h', di.get('height', 0.0))
                wj = dj.get('w', dj.get('width', 0.0))
                hj = dj.get('h', dj.get('height', 0.0))

                if axis == 'h':
                    slack = (dj['x'] - di['x']) - 0.5 * (wi + wj)
                else:
                    slack = (dj['y'] - di['y']) - 0.5 * (hi + hj)

                if slack > best_slack:
                    best_slack = slack
                    edge_to_remove = (u, v)

            if edge_to_remove:
                G.remove_edge(*edge_to_remove)
                total_removed += 1
            else:
                fallback_edge = cycle_edges[-1][:2]
                G.remove_edge(*fallback_edge)
                total_removed += 1

        except nx.NetworkXNoCycle:
            if verbose:
                print(f"  [break_cycles] Cleaned {total_removed} edges in {iteration} iterations. Now a DAG.")
            break

    return list(G.edges()), total_removed


def determine_relation_type(di, dj, die_width=None, die_height=None):
    """
    Classify the constraint between two modules as horizontal ('h') or vertical ('v').

    dx = |xi - xj| - 0.5*(wi + wj)  : signed gap in X  (negative → x-overlap)
    dy = |yi - yj| - 0.5*(hi + hj)  : signed gap in Y  (negative → y-overlap)

    Rules:
      dx > dy  →  horizontal constraint  (larger x-separation, or smaller x-overlap)
      dx < dy  →  vertical   constraint
      dx == dy →  tie-break: wider die → 'h', taller die → 'v'

    When both dx and dy are negative (modules overlap in both dimensions), the same
    comparison still holds: the less-negative value corresponds to the axis that needs
    the smaller shift to resolve the overlap, so we assign that axis's constraint.
    """
    xi, yi = di['x'], di['y']
    xj, yj = dj['x'], dj['y']

    wi = di.get('w', di.get('width', 0))
    hi = di.get('h', di.get('height', 0))
    wj = dj.get('w', dj.get('width', 0))
    hj = dj.get('h', dj.get('height', 0))

    dx = abs(xi - xj) - 0.5 * (wi + wj)
    dy = abs(yi - yj) - 0.5 * (hi + hj)

    if dx > dy:
        return 'h'
    elif dx < dy:
        return 'v'
    else:
        # Exact tie: fall back to die aspect ratio
        if die_width is not None and die_height is not None:
            return 'h' if die_width >= die_height else 'v'
        return 'h'  # default if die dimensions unknown


def build_constraint_graphs(module_data, movable_modules, fixed_modules,
                            die_width=None, die_height=None):
    """Build H/V constraint graphs in unified index space."""

    all_modules = movable_modules + fixed_modules
    total = len(all_modules)

    init_tag = 'full'
    h_edges = []
    v_edges = []
    stats = {'h': 0, 'v': 0, 'pairs': 0}

    for i in range(total):
        for j in range(i + 1, total):
            di = module_data[all_modules[i]]
            dj = module_data[all_modules[j]]
            if di.get('fixed', False) and dj.get('fixed', False):
                continue
            stats['pairs'] += 1

            he = (i, j) if di['x'] <= dj['x'] else (j, i)
            ve = (i, j) if di['y'] <= dj['y'] else (j, i)

            rel_type = determine_relation_type(di, dj, die_width, die_height)

            if rel_type == 'h':
                h_edges.append(he)
                stats['h'] += 1
            else:
                v_edges.append(ve)
                stats['v'] += 1

    ih0, iv0 = len(h_edges), len(v_edges)

    # ── Break cycles ───────────────────────────────────────────────────────
    h_edges, hr = break_cycles(h_edges, total, module_data, all_modules, axis='h')
    v_edges, vr = break_cycles(v_edges, total, module_data, all_modules, axis='v')

    # ── Transitive reduction ───────────────────────────────────────────────
    G_h_final = nx.transitive_reduction(nx.DiGraph(h_edges))
    G_v_final = nx.transitive_reduction(nx.DiGraph(v_edges))
    h_is_dag = nx.is_directed_acyclic_graph(G_h_final)
    v_is_dag = nx.is_directed_acyclic_graph(G_v_final)

    h_edges_reduced = list(G_h_final.edges())
    v_edges_reduced = list(G_v_final.edges())

    print(
        f"  graphs  n_mod={total} init={init_tag} | "
        f"H0={ih0} V0={iv0} break1=-{hr + vr} -> H={len(h_edges)} V={len(v_edges)} | "
        f"reduce -> H={len(h_edges_reduced)} V={len(v_edges_reduced)} | "
        f"DAG(H)={h_is_dag} DAG(V)={v_is_dag}"
    )

    return h_edges_reduced, v_edges_reduced, all_modules

def _compact_path_str(path, all_modules, head=2, tail=2):
    """Short string for long critical paths: a -> b … (N) … y -> z."""
    names = [str(all_modules[n]) for n in path if isinstance(n, int)]
    n = len(names)
    if n == 0:
        return '—'
    if n <= head + tail:
        return ' -> '.join(names)
    return (
        f"{' -> '.join(names[:head])} … ({n} nodes) … "
        f"{' -> '.join(names[-tail:])}"
    )


def _dim_fn_for_critical_path(direction: str):
    if direction == 'h':
        return lambda m: float(m.get('w', m.get('width', 0.0)))
    return lambda m: float(m.get('h', m.get('height', 0.0)))


def critical_path_metrics(
    G_edges,
    module_data,
    all_modules,
    direction='h',
    verbose: bool = False,
):
    """
    same as check_critical_path: weighted DAG longest path + dummy sink; does not modify input edge list.

    :returns: (path_without_sink, max_length) no edges: ([], 0.0), cycles: ([], inf)
    """
    get_dim = _dim_fn_for_critical_path(direction)
    if not G_edges:
        if verbose:
            tag = 'H' if direction == 'h' else 'V'
            print(f"  critical [{tag}] (no edges)")
        return [], 0.0

    WG = nx.DiGraph()
    WG.add_edges_from(G_edges)

    for u, v in WG.edges():
        u_name = all_modules[u]
        WG[u][v]['weight'] = max(get_dim(module_data[u_name]), 0.0)

    DUMMY_SINK = '__SINK__'
    WG.add_node(DUMMY_SINK)
    terminal_nodes = [n for n in WG.nodes() if WG.out_degree(n) == 0 and n != DUMMY_SINK]

    for node in terminal_nodes:
        node_name = all_modules[node]
        WG.add_edge(node, DUMMY_SINK, weight=max(get_dim(module_data[node_name]), 0.0))

    try:
        path = nx.dag_longest_path(WG, weight='weight')
        max_length = float(nx.dag_longest_path_length(WG, weight='weight'))
    except nx.NetworkXUnfeasible:
        if verbose:
            tag = 'H' if direction == 'h' else 'V'
            print(f"  critical [{tag}] (graph has cycles — cannot compute path)")
        return [], float('inf')

    if path and path[-1] == DUMMY_SINK:
        path = path[:-1]

    return path, max_length


def calculate_longest_path_length(G_edges, module_data, all_modules, direction='h'):
    """return critical path heuristic length (same as critical_path_metrics)"""
    _, L = critical_path_metrics(G_edges, module_data, all_modules, direction=direction, verbose=False)
    return float(L)


def check_critical_path(G_edges, module_data, all_modules, direction='h'):
    """
    check weighted longest path on constraint DAG (critical path heuristic)

    edge weights are predecessor module dimensions in separation direction (w for horizontal, h for vertical),
    followed by dummy sink to account for trailing module's own size.
    if graph has cycles, cannot compute.

    :param G_edges: constraint edge list, e.g. [(0, 1), (1, 2)] (unified index all_modules)
    :param module_data: module dictionary, with coordinates and w/h (or width/height)
    :param all_modules: module name list with unified index
    :param direction: 'h' horizontal graph, 'v' vertical graph
    :returns: (path_indices, max_length) no edges: ([], 0), cycles: ([], inf)
    """
    path, max_length = critical_path_metrics(
        G_edges, module_data, all_modules, direction=direction, verbose=True
    )
    if max_length == float('inf'):
        return [], float('inf')

    if not G_edges:
        return path, max_length

    tag = 'H' if direction == 'h' else 'V'
    cpath = _compact_path_str(path, all_modules)
    print(
        f"  critical [{tag}] nodes={len(path)} sum={max_length:.4g} | {cpath}"
    )

    return path, max_length


def setup_model(movable_modules, hard_modules, fixed_modules, module_data, terminal_coords, nets,
                die_width, die_height, h_edges_reduced, v_edges_reduced, all_modules,
                alpha):
    r"""
    Build the convex NLP in mixed linear / log space.

    Variables (4n, where n = len(soft)+2n for hard modules, terminals and fixed modules are constants):
        x[i]  — module center x  (linear)
        y[i]  — module center y  (linear)
        W[i]  — ln(w_i)          (log of width)
        H[i]  — ln(h_i)          (log of height)
    
    """
    n = len(movable_modules)
    hard_set = set(hard_modules)
    W_F = float(die_width)
    H_F = float(die_height)
    # ---------- symbolic variables ----------
    x = ca.MX.sym('x', n)
    y = ca.MX.sym('y', n)
    W = ca.MX.sym('W', n)      # ln(w_i)
    H = ca.MX.sym('H', n)      # ln(h_i)
    opt_vars = ca.vertcat(x, y, W, H)

    # ---------- bounds & initial guess ----------
    lbx, ubx, x0 = [], [], []
    eps_pos = 0.5               # minimum centre if none give

    for i in range(n):
        d = module_data[movable_modules[i]]
        lbx.append(eps_pos);  ubx.append(W_F)
        x0.append(max(d['x'], eps_pos))

    for i in range(n):
        d = module_data[movable_modules[i]]
        lbx.append(eps_pos);  ubx.append(H_F)
        x0.append(max(d['y'], eps_pos))

    for i in range(n):
        d = module_data[movable_modules[i]]
        area = d['area']
        if movable_modules[i] in hard_set:
            w_fix = max(float(d['w']), 0.01)
            lbx.append(np.log(w_fix));  ubx.append(np.log(w_fix))
            x0.append(np.log(w_fix))
            continue
        ar_min = max(float(d['ar_min']), 1e-8)
        ar_max = max(float(d['ar_max']), ar_min)
        if area > 0:
            # w/h in [r_min, r_max] and w*h = area  =>  w in [sqrt(A*r_min), sqrt(A*r_max)]
            w_min = max(np.sqrt(area * ar_min), 0.01)
            w_max = min(np.sqrt(area * ar_max), W_F)
        else:
            w_min, w_max = 0.01, W_F
        lbx.append(np.log(w_min));  ubx.append(np.log(w_max))
        x0.append(np.log(max(d['w'], 0.01)))

    for i in range(n):
        d = module_data[movable_modules[i]]
        area = d['area']
        if movable_modules[i] in hard_set:
            h_fix = max(float(d['h']), 0.01)
            lbx.append(np.log(h_fix));  ubx.append(np.log(h_fix))
            x0.append(np.log(h_fix))
            continue
        ar_min = max(float(d['ar_min']), 1e-8)
        ar_max = max(float(d['ar_max']), ar_min)
        if area > 0:
            # h = area / w  => h in [sqrt(A/r_max), sqrt(A/r_min)]
            h_min = max(np.sqrt(area / ar_max), 0.01)
            h_max = min(np.sqrt(area / ar_min), H_F)
        else:
            h_min, h_max = 0.01, H_F
        lbx.append(np.log(h_min));  ubx.append(np.log(h_max))
        x0.append(np.log(max(d['h'], 0.01)))

    # ---------- constraints ----------
    g, lbg, ubg = [], [], []

    # 1. Area:  W_i + H_i  ≥  ln(A_i)     (linear, convex)
    n_area = 0
    for i in range(n):
        if movable_modules[i] in hard_set:
            continue
        area = module_data[movable_modules[i]]['area']
        if area > 0:
            g.append(W[i] + H[i])
            lbg.append(np.log(area));  ubg.append(ca.inf)
            n_area += 1

    # 2. Aspect ratio  (linear, convex)
    #    ln(r_min) <= W_i - H_i <= ln(r_max)
    #    equivalent to r_min <= w_i/h_i <= r_max
    n_ar = 0
    for i in range(n):
        if movable_modules[i] in hard_set:
            continue
        d = module_data[movable_modules[i]]
        ar_min = max(float(d['ar_min']), 1e-8)
        ar_max = max(float(d['ar_max']), ar_min)
        log_ar_min = np.log(ar_min)
        log_ar_max = np.log(ar_max)
        g.append(W[i] - H[i])
        lbg.append(log_ar_min);  ubg.append(log_ar_max)
        n_ar += 1

    def half_w_expr(idx):
        if idx < n:
            name = movable_modules[idx]
            if name in hard_set:
                return 0.5 * float(module_data[name]['w'])
            return 0.5 * ca.exp(W[idx])
        return 0.5 * module_data[all_modules[idx]]['w']

    def half_h_expr(idx):
        if idx < n:
            name = movable_modules[idx]
            if name in hard_set:
                return 0.5 * float(module_data[name]['h'])
            return 0.5 * ca.exp(H[idx])
        return 0.5 * module_data[all_modules[idx]]['h']

    def cx_expr(idx):
        if idx < n:
            return x[idx]
        return module_data[all_modules[idx]]['x']

    def cy_expr(idx):
        if idx < n:
            return y[idx]
        return module_data[all_modules[idx]]['y']

    # 3. Boundary
    #    Left   :  half_w(i) − x_i  ≤  0
    #    Bottom :  half_h(i) − y_i  ≤  0
    #    Right  :  x_i + half_w(i)  ≤  W_F
    #    Top    :  y_i + half_h(i)  ≤  H_F
    for i in range(n):
        half_w = half_w_expr(i)
        half_h = half_h_expr(i)

        g.append(half_w - x[i])
        lbg.append(-ca.inf);  ubg.append(0.0)

        g.append(half_h - y[i])
        lbg.append(-ca.inf);  ubg.append(0.0)

        g.append(x[i] + half_w - W_F)
        lbg.append(-ca.inf);  ubg.append(0.0)

        g.append(y[i] + half_h - H_F)
        lbg.append(-ca.inf);  ubg.append(0.0)

    # 4. Non-overlap HCG over unified graph
    n_h = 0
    for (a, b) in h_edges_reduced:
        if a >= n and b >= n:
            continue
        g.append(cx_expr(a) + half_w_expr(a) + half_w_expr(b) - cx_expr(b))
        lbg.append(-ca.inf);  ubg.append(0.0)
        n_h += 1

    # 5. Non-overlap VCG over unified graph
    n_v = 0
    for (a, b) in v_edges_reduced:
        if a >= n and b >= n:
            continue
        g.append(cy_expr(a) + half_h_expr(a) + half_h_expr(b) - cy_expr(b))
        lbg.append(-ca.inf);  ubg.append(0.0)
        n_v += 1

    # ---------- objective: numerically-stable LSE-HPWL (max trick) ----------
    # log(Σ exp(z_i)) = m + log(Σ exp(z_i − m)),  m = max(z_i)
    # HPWL ≈ α · [ lse(x_i/α) + lse(−x_i/α) + lse(y_i/α) + lse(−y_i/α) ]

    def _stable_lse(vals):
        """log-sum-exp with max-trick.  *vals* may mix ca.SX/MX and float."""
        v = ca.vertcat(*vals)
        m = v[0]
        for k in range(1, v.shape[0]):
            m = ca.fmax(m, v[k])
        return m + ca.log(ca.sum1(ca.exp(v - m)))

    obj_parts = []

    for (wgt, mod_ids, term_pos) in nets:
        # ---- x-direction ----
        zp, zn = [], []
        for idx in mod_ids:
            zp.append(x[idx] / alpha)
            zn.append(-x[idx] / alpha)
        for (px, py) in term_pos:
            zp.append(px / alpha)
            zn.append(-px / alpha)

        if len(zp) >= 2:
            hpwl_x = alpha * (_stable_lse(zp) + _stable_lse(zn))
        else:
            hpwl_x = 0.0

        # ---- y-direction ----
        zp, zn = [], []
        for idx in mod_ids:
            zp.append(y[idx] / alpha)
            zn.append(-y[idx] / alpha)
        for (px, py) in term_pos:
            zp.append(py / alpha)
            zn.append(-py / alpha)

        if len(zp) >= 2:
            hpwl_y = alpha * (_stable_lse(zp) + _stable_lse(zn))
        else:
            hpwl_y = 0.0

        obj_parts.append(wgt * (hpwl_x + hpwl_y))

    if obj_parts:
        objective = ca.sum1(ca.vertcat(*obj_parts))
    else:
        objective = ca.sum1(x) + ca.sum1(y)

    g_vec = ca.vertcat(*g) if g else ca.MX()
    nlp = {'x': opt_vars, 'f': objective, 'g': g_vec}


    print(
        f"  NLP  n={n} vars={opt_vars.size1()} g={len(g)} "
        f"(area {n_area}, ar {n_ar}, bd {4 * n}, H {n_h}, V {n_v}) "
        f"nets={len(nets)} α={alpha:.4g}"
    )

    return nlp, {
        'x0': x0, 'lbx': lbx, 'ubx': ubx,
        'lbg': lbg, 'ubg': ubg,
        'n': n,
    }



def solve_casadi(nlp, problem_data, max_iters=500, verbose=True):
    opts = {}
    opts['ipopt.max_iter'] = max_iters
    opts['ipopt.print_level'] = 5 if verbose else 0
    opts['print_time'] = verbose

    opts['ipopt.tol'] = 1e-4
    opts['ipopt.acceptable_tol'] = 1e-2
    opts['ipopt.acceptable_iter'] = 15
    opts['ipopt.acceptable_constr_viol_tol'] = 1e-3
    opts['ipopt.constr_viol_tol'] = 1e-4

    opts['ipopt.linear_solver'] = 'mumps'
    opts['ipopt.hessian_approximation'] = 'limited-memory'
    opts['ipopt.mu_strategy'] = 'adaptive'
    opts['ipopt.warm_start_init_point'] = 'yes'
    opts['ipopt.bound_relax_factor'] = 1e-8
    opts['ipopt.honor_original_bounds'] = 'yes'

    try:
        S = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = S(x0=problem_data['x0'],
                lbx=problem_data['lbx'], ubx=problem_data['ubx'],
                lbg=problem_data['lbg'], ubg=problem_data['ubg'])

        stats = S.stats()
        status = stats.get('return_status', 'Unknown')
        ok = stats.get('success', False)

        usable = ['Solve_Succeeded', 'Solved_To_Acceptable_Level',
                   'Maximum_Iterations_Exceeded',
                   'Search_Direction_Becomes_Too_Small',
                   'Feasible_Point_Found']
        if ok:
            print(f"  ✓ Solver succeeded  ({status})")
            return True, sol
        elif any(s in status for s in usable):
            print(f"  Solver: {status} — accepting as best feasible")
            return True, sol
        else:
            print(f"  ✗ Solver failed: {status}")
            return False, sol

    except Exception as e:
        import traceback
        print(f"  Solver error: {e}")
        traceback.print_exc()
        return False, None



def extract_solution(sol, movable_modules, module_data, problem_data):
    """Convert optimised variables back to physical coordinates."""
    n = problem_data['n']
    v = sol['x'].full().flatten()

    x_vals = v[0*n : 1*n]
    y_vals = v[1*n : 2*n]
    W_vals = v[2*n : 3*n]
    H_vals = v[3*n : 4*n]

    for i, name in enumerate(movable_modules):
        module_data[name].update({
            'x': float(x_vals[i]),
            'y': float(y_vals[i]),
            'w': float(np.exp(W_vals[i])),
            'h': float(np.exp(H_vals[i])),
        })



# I/O & visualisation

def save_results(netlist_data, movable_modules, module_data, output_file):
    updated = netlist_data.copy()
    modules_dict = updated.get('Modules', updated)
    if 'Modules' not in updated:
        updated = {'Modules': modules_dict}

    # Only update movable modules' rectangles
    for name in movable_modules:
        if name in modules_dict:
            d = module_data[name]
            modules_dict[name]['rectangles'] = [
                [d['x'], d['y'], d['w'], d['h']]]
            # Keep output compatible with Module._check_consistency():
            # hard modules cannot define area/length/center/aspect_ratio.
            is_hard = bool(d.get('hard', False)) or bool(modules_dict[name].get('hard', False))
            if is_hard:
                for k in ('area', 'length', 'center', 'aspect_ratio'):
                    modules_dict[name].pop(k, None)

    with open(output_file, 'w') as f:
        yaml.dump(updated, f, default_flow_style=False, sort_keys=False)


def calculate_overlap(movable_modules, fixed_modules, module_data):
    total = 0.0
    n = len(movable_modules)
    m = len(fixed_modules)
    
    # Overlap between movable modules
    for i in range(n):
        di = module_data[movable_modules[i]]
        for j in range(i + 1, n):
            dj = module_data[movable_modules[j]]
            ox = max(0, min(di['x'] + di['w']/2, dj['x'] + dj['w']/2)
                        - max(di['x'] - di['w']/2, dj['x'] - dj['w']/2))
            oy = max(0, min(di['y'] + di['h']/2, dj['y'] + dj['h']/2)
                        - max(di['y'] - di['h']/2, dj['y'] - dj['h']/2))
            total += ox * oy
    
    # Overlap between movable and fixed modules
    for i in range(n):
        di = module_data[movable_modules[i]]
        for f_name in fixed_modules:
            df = module_data[f_name]
            ox = max(0, min(di['x'] + di['w']/2, df['x'] + df['w']/2)
                        - max(di['x'] - di['w']/2, df['x'] - df['w']/2))
            oy = max(0, min(di['y'] + di['h']/2, df['y'] + df['h']/2)
                        - max(di['y'] - di['h']/2, df['y'] - df['h']/2))
            total += ox * oy
    
    return total


def report_overlap_pairs_with_graph_status(
    movable_modules,
    fixed_modules,
    module_data,
    h_edges,
    v_edges,
    all_modules,
    eps=1e-9,
):
    """
    Print only overlapping module pairs that have NO edge in both H/V graphs.
    """
    idx = {name: i for i, name in enumerate(all_modules)}
    h_set = set(h_edges)
    v_set = set(v_edges)

    def _edge_flags(a_name, b_name):
        if a_name not in idx or b_name not in idx:
            return False, False
        ia, ib = idx[a_name], idx[b_name]
        in_h = (ia, ib) in h_set or (ib, ia) in h_set
        in_v = (ia, ib) in v_set or (ib, ia) in v_set
        return in_h, in_v

    pairs = []
    n = len(movable_modules)

    # movable-movable overlaps
    for i in range(n):
        a = movable_modules[i]
        da = module_data[a]
        for j in range(i + 1, n):
            b = movable_modules[j]
            db = module_data[b]
            ox = max(0.0, min(da['x'] + da['w'] / 2, db['x'] + db['w'] / 2)
                        - max(da['x'] - da['w'] / 2, db['x'] - db['w'] / 2))
            oy = max(0.0, min(da['y'] + da['h'] / 2, db['y'] + db['h'] / 2)
                        - max(da['y'] - da['h'] / 2, db['y'] - db['h'] / 2))
            ov = ox * oy
            if ov > eps:
                in_h, in_v = _edge_flags(a, b)
                pairs.append((ov, ox, oy, a, b, in_h, in_v))

    # movable-fixed overlaps
    for a in movable_modules:
        da = module_data[a]
        for b in fixed_modules:
            db = module_data[b]
            ox = max(0.0, min(da['x'] + da['w'] / 2, db['x'] + db['w'] / 2)
                        - max(da['x'] - da['w'] / 2, db['x'] - db['w'] / 2))
            oy = max(0.0, min(da['y'] + da['h'] / 2, db['y'] + db['h'] / 2)
                        - max(da['y'] - da['h'] / 2, db['y'] - db['h'] / 2))
            ov = ox * oy
            if ov > eps:
                in_h, in_v = _edge_flags(a, b)
                pairs.append((ov, ox, oy, a, b, in_h, in_v))

    missing_pairs = [t for t in pairs if (not t[5] and not t[6])]
    missing_pairs.sort(key=lambda t: -t[0])

    if not missing_pairs:
        print("  Overlap pairs missing graph edges: none")
        return []

    print(f"  Overlap pairs missing graph edges: {len(missing_pairs)}")
    for rank, (ov, ox, oy, a, b, in_h, in_v) in enumerate(missing_pairs, 1):
        edge_state = f"H-edge={'Y' if in_h else 'N'}, V-edge={'Y' if in_v else 'N'}"
        print(f"    #{rank:02d}  {a} <-> {b}  ov={ov:.6g} (ox={ox:.6g}, oy={oy:.6g})  {edge_state}")
    return missing_pairs



def calculate_true_hpwl_from_netlist_data(netlist_data) -> float:
    """
    Compute true HPWL from output netlist YAML data.
    Includes movable modules, fixed modules, and terminals.
    """
    modules_dict = netlist_data.get("Modules", netlist_data)
    coords = {}

    for name, info in modules_dict.items():
        if not isinstance(info, dict):
            continue

        rects = info.get("rectangles", [])
        if rects and isinstance(rects[0], (list, tuple)) and len(rects[0]) >= 2:
            coords[name] = (float(rects[0][0]), float(rects[0][1]))
            continue

        center = info.get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            coords[name] = (float(center[0]), float(center[1]))

    total = 0.0
    for net in netlist_data.get("Nets", []):
        if not isinstance(net, (list, tuple)):
            continue

        pins = list(net)
        weight = 1.0
        if pins and isinstance(pins[-1], (int, float)) and not isinstance(pins[-1], bool):
            weight = float(pins.pop())

        xs, ys = [], []
        for pin in pins:
            pin_name = str(pin)
            if pin_name in coords:
                x, y = coords[pin_name]
                xs.append(x)
                ys.append(y)

        if len(xs) >= 2:
            total += weight * ((max(xs) - min(xs)) + (max(ys) - min(ys)))

    return total


def visualize(movable_modules, hard_modules, fixed_modules, terminal_coords,
              module_data, die_w, die_h, output_image, title_extra=''):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-die_w * 0.05, die_w * 1.1)
    ax.set_ylim(-die_h * 0.05, die_h * 1.1)
    ax.set_aspect('equal')

    ax.add_patch(patches.Rectangle(
        (0, 0), die_w, die_h, fill=False, edgecolor='red', linewidth=2))

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(movable_modules), 1)))
    hard_set = set(hard_modules)

    # Draw movable modules:
    #   - soft: filled color
    #   - hard: filled color + thicker edge + hatch
    for i, name in enumerate(movable_modules):
        d = module_data[name]
        x0 = d['x'] - d['w'] / 2
        y0 = d['y'] - d['h'] / 2
        is_hard = name in hard_set
        ax.add_patch(patches.Rectangle(
            (x0, y0), d['w'], d['h'],
            facecolor=colors[i % 20],
            edgecolor='black',
            linewidth=1.6 if is_hard else 1.0,
            hatch='xx' if is_hard else None,
            alpha=0.72 if is_hard else 0.65))
        ax.text(d['x'], d['y'], name,
                ha='center', va='center', fontsize=5)
    
    # Draw fixed modules in gray (hatched)
    for name in fixed_modules:
        d = module_data[name]
        x0 = d['x'] - d['w'] / 2
        y0 = d['y'] - d['h'] / 2
        ax.add_patch(patches.Rectangle(
            (x0, y0), d['w'], d['h'],
            facecolor='gray', edgecolor='black', alpha=0.5, hatch='///'))
        ax.text(d['x'], d['y'], name,
                ha='center', va='center', fontsize=5, weight='bold')

    # Draw terminals as red stars with labels
    for tname, (tx, ty) in terminal_coords.items():
        ax.plot(tx, ty, marker='*', markersize=7, color='red', zorder=5)
        ax.text(tx, ty, tname, color='red', fontsize=5, ha='left', va='bottom')

    # Legend for categories
    soft_legend = patches.Patch(facecolor='lightblue', edgecolor='black', alpha=0.65, label='soft')
    hard_legend = patches.Patch(facecolor='lightblue', edgecolor='black', hatch='xx', linewidth=1.6, alpha=0.72, label='hard')
    fixed_legend = patches.Patch(facecolor='gray', edgecolor='black', hatch='///', alpha=0.5, label='fixed')
    terminal_legend = plt.Line2D([0], [0], marker='*', color='red', linestyle='None', markersize=8, label='terminal')
    ax.legend(handles=[soft_legend, hard_legend, fixed_legend, terminal_legend], loc='upper right', fontsize=8)

    ax.set_title(f'Die {die_w:.1f}×{die_h:.1f}  {title_extra}')
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    plt.close()



def optimize_floorplan(netlist_file, die_file, output_file, output_image,
                       max_iter=500, min_aspect_ratio=None, max_ratio=None, alpha=None):

    print('=' * 70)
    print('Mixed-Space Convex Floorplan Optimisation')
    print('  Variables: x,y (linear) + W,H (log of dimensions)')
    print('=' * 70)

    netlist_data = load_netlist(netlist_file)
    die_info = load_die_info(die_file)
    die_w = die_info.get('width', 800.0)
    die_h = die_info.get('height', 800.0)

    movable_modules, hard_modules, fixed_modules, module_data, terminal_coords = extract_modules_and_terminals(
        netlist_data,
        default_min_ar=min_aspect_ratio,
        default_max_ar=max_ratio,
    )
    n = len(movable_modules)
    nh = len(hard_modules)
    m = len(fixed_modules)
    print(f"  Movable modules: {n}")
    print(f"  Hard modules:    {nh}")
    print(f"  Fixed modules:   {m}")
    print(f"  Terminals:       {len(terminal_coords)}")
    print(f"  Die:             {die_w:.2f} × {die_h:.2f}")

    nets = extract_nets(netlist_data, movable_modules, fixed_modules, module_data, terminal_coords)
    print(f"  Nets:            {len(nets)}")

    if alpha is None:
        alpha = max(die_w, die_h) / 20.0
    print(f"  α (LSE smoothing): {alpha:.4f}")
    if min_aspect_ratio is not None or max_ratio is not None:
        if min_aspect_ratio is None or max_ratio is None:
            raise ValueError("Provide both --min-aspect-ratio and --max-ratio, or neither.")
        if min_aspect_ratio <= 0:
            raise ValueError("min_aspect_ratio must be > 0")
        if max_ratio <= 0:
            raise ValueError("max_ratio must be > 0")
        if min_aspect_ratio > max_ratio:
            raise ValueError("min_aspect_ratio must be <= max_ratio")
        print(f"  CLI fallback aspect ratio range (w/h): [{min_aspect_ratio:.4f}, {max_ratio:.4f}]")
    else:
        print("  CLI fallback aspect ratio range: not set (must come from netlist)")

    # Clamp movable modules' initial positions within bounds
    for name in movable_modules:
        d = module_data[name]
        d['w'] = max(d['w'], 0.01)
        d['h'] = max(d['h'], 0.01)
        mx = 0.5 * d['w'] + 0.01
        my = 0.5 * d['h'] + 0.01
        d['x'] = max(mx, min(d['x'], die_w - mx))
        d['y'] = max(my, min(d['y'], die_h - my))
    print()

    print('Building constraint graphs ...')
    h_edges, v_edges, all_modules = build_constraint_graphs(module_data, movable_modules, fixed_modules, die_w, die_h)
    print()

    h_path, min_chip_width = check_critical_path(
        h_edges, module_data, all_modules, direction='h')
    v_path, min_chip_height = check_critical_path(
        v_edges, module_data, all_modules, direction='v')

    if min_chip_width > float(die_w) + 1e-6:
        print(
            f"  !! H critical {min_chip_width:.4g} > die_w {die_w} | "
            f"{_compact_path_str(h_path, all_modules)}"
        )
    if min_chip_height > float(die_h) + 1e-6:
        print(
            f"  !! V critical {min_chip_height:.4g} > die_h {die_h} | "
            f"{_compact_path_str(v_path, all_modules)}"
        )


    print('Setting up mixed-space model ...')
    nlp, pdata = setup_model(
        movable_modules, hard_modules, fixed_modules, module_data, terminal_coords, nets,
        die_w, die_h, h_edges, v_edges, all_modules,
        alpha=alpha)
    print()

    print('Solving ...')
    ok, sol = solve_casadi(nlp, pdata, max_iters=max_iter, verbose=True)

    if not ok or sol is None:
        print('\n✗ Optimisation failed — no usable solution.')
        return False

    extract_solution(sol, movable_modules, module_data, pdata)

    overlap = calculate_overlap(movable_modules, fixed_modules, module_data)
    report_overlap_pairs_with_graph_status(
        movable_modules, fixed_modules, module_data,
        h_edges, v_edges, all_modules
    )

    save_results(netlist_data, movable_modules, module_data, output_file)
    output_netlist_data = load_netlist(output_file)
    hpwl_true = calculate_true_hpwl_from_netlist_data(output_netlist_data)
    visualize(movable_modules, hard_modules, fixed_modules, terminal_coords,
              module_data, die_w, die_h, output_image,
              title_extra=f'HPWL={hpwl_true:.1f}')

    print(f'\n{"=" * 70}')
    print(f'Optimisation completed!')
    print(f'  HPWL (true, output netlist): {hpwl_true:.2f}')
    #print(f'  Overlap: {overlap:.2f}')
    print(f'  Output:  {output_file}, {output_image}')
    print(f'{"=" * 70}\n')
    return True



# CLI

def main():
    p = argparse.ArgumentParser(
        description='Mixed-Space Convex Floorplan Optimisation')
    p.add_argument('--netlist', required=True, help='Netlist YAML file')
    p.add_argument('--die', required=True, help='Die info YAML file')
    p.add_argument('--output', default='output_cvx_legalizer.yaml')
    p.add_argument('--output-image', default='output_cvx_legalizer.png')
    p.add_argument('--max-iter', type=int, default=500)
    p.add_argument('--max-ratio', type=float, default=None,
                   help='Fallback max aspect ratio ρ (used only when module omits aspect_ratio)')
    p.add_argument('--min-aspect-ratio', type=float, default=None,
                   help='Fallback min aspect ratio for w/h (used only when module omits aspect_ratio)')
    p.add_argument('--alpha', type=float, default=1.0,
                   help='LSE smoothing α ')
    args = p.parse_args()

    optimize_floorplan(
        args.netlist, args.die, args.output, args.output_image,
        max_iter=args.max_iter,
        min_aspect_ratio=args.min_aspect_ratio,
        max_ratio=args.max_ratio,
        alpha=args.alpha)


if __name__ == '__main__':
    main()