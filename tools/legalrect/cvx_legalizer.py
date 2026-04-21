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
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import sys

sys.setrecursionlimit(10000)


def load_netlist(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_die_info(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)



def extract_modules_and_terminals(netlist_data):
    """
    Returns
    -------
    modules : list[str]
    module_data : dict  – {name: {area, x, y, w, h}}
    terminal_coords : dict  – {name: (x, y)}
    """
    modules = []
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

        if info.get('fixed', False):
            continue

        area = info.get('area', 0)
        rects = info.get('rectangles', [[0, 0, 1, 1]])
        if rects and len(rects[0]) == 4:
            x, y, w, h = [float(v) for v in rects[0]]
        else:
            x, y, w, h = 0.0, 0.0, 1.0, 1.0

        if area <= 0 and w > 0 and h > 0:
            area = w * h

        module_data[name] = {'area': area, 'x': x, 'y': y, 'w': w, 'h': h}
        modules.append(name)

    return modules, module_data, terminal_coords


def extract_nets(netlist_data, modules, module_data, terminal_coords):
    mod_idx = {name: i for i, name in enumerate(modules)}
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

        if len(mod_ids) + len(term_pos) >= 2:
            nets.append((weight, mod_ids, term_pos))

    return nets



def detect_cycles(edges, n):
    adj = [[] for _ in range(n)]
    for (i, j) in edges:
        adj[i].append(j)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    cycles = []

    def dfs(node, path):
        if color[node] == GRAY:
            cycle_start = path.index(node)
            cycles.append(path[cycle_start:] + [node])
            return
        if color[node] == BLACK:
            return
        color[node] = GRAY
        path.append(node)
        for nb in adj[node]:
            dfs(nb, path)
        path.pop()
        color[node] = BLACK

    for i in range(n):
        if color[i] == WHITE:
            dfs(i, [])
    return cycles


def topological_sort(edges, n):
    adj = [[] for _ in range(n)]
    in_degree = [0] * n
    for (i, j) in edges:
        adj[i].append(j)
        in_degree[j] += 1

    queue = [i for i in range(n) if in_degree[i] == 0]
    ordering = []
    while queue:
        node = queue.pop(0)
        ordering.append(node)
        for nb in adj[node]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)
    return len(ordering) == n, ordering


def break_cycles(edges, n, module_data, modules):
    total_removed = 0
    iteration = 0
    max_iterations = 10

    while iteration < max_iterations:
        cycles = detect_cycles(edges, n)
        if not cycles:
            break
        iteration += 1
        print(f"  Iteration {iteration}: Found {len(cycles)} cycles")

        edge_slacks = {}
        for (i, j) in edges:
            di = module_data[modules[i]]
            dj = module_data[modules[j]]
            if i < j:
                slack = dj['x'] - di['x']
            else:
                slack = dj['y'] - di['y']
            edge_slacks[(i, j)] = slack

        edges_to_remove = set()
        for cycle in cycles:
            best_slack = -float('inf')
            best_edge = None
            for k in range(len(cycle) - 1):
                e = (cycle[k], cycle[k + 1])
                if e in edge_slacks and edge_slacks[e] > best_slack:
                    best_slack = edge_slacks[e]
                    best_edge = e
            if best_edge:
                edges_to_remove.add(best_edge)

        edges = [e for e in edges if e not in edges_to_remove]
        total_removed += len(edges_to_remove)

    if iteration >= max_iterations:
        print(f"    Warning: max iterations ({max_iterations}) reached")
        rem = detect_cycles(edges, n)
        if rem:
            print(f"    {len(rem)} cycles remain")

    print(f"  Total removed {total_removed} edges to break cycles")
    return edges, total_removed


def build_constraint_graphs(module_data, modules):
    n = len(modules)
    h_edges = []
    v_edges = []
    stats = {'h': 0, 'v': 0, 'both': 0, 'pairs': 0}

    for i in range(n):
        for j in range(i + 1, n):
            di = module_data[modules[i]]
            dj = module_data[modules[j]]
            dx = abs(dj['x'] - di['x'])
            dy = abs(dj['y'] - di['y'])
            stats['pairs'] += 1

            he = (i, j) if di['x'] < dj['x'] else (j, i)
            ve = (i, j) if di['y'] < dj['y'] else (j, i)

            if abs(dx - dy) < 1e-6:
                h_edges.append(he)
                v_edges.append(ve)
                stats['both'] += 1
            elif dx > dy:
                h_edges.append(he)
                stats['h'] += 1
            else:
                v_edges.append(ve)
                stats['v'] += 1

    print(f"  Constraint selection based on center distances:")
    print(f"    Total pairs: {stats['pairs']}")
    print(f"    Horizontal: {stats['h']}, Vertical: {stats['v']}, "
          f"Both: {stats['both']}")
    print(f"    H-edges: {len(h_edges)}, V-edges: {len(v_edges)}")

    print(f"  Checking cycles in horizontal constraint graph...")
    h_edges, hr = break_cycles(h_edges, n, module_data, modules)
    print(f"  Checking cycles in vertical constraint graph...")
    v_edges, vr = break_cycles(v_edges, n, module_data, modules)
    print(f"  Total edges removed: {hr + vr}")
    print(f"  Final: H={len(h_edges)}, V={len(v_edges)}")

    return h_edges, v_edges



def setup_model(modules, module_data, terminal_coords, nets,
                die_width, die_height, h_edges, v_edges,
                alpha, min_aspect_ratio=1.0, max_ratio=3.0):
    r"""
    Build the convex NLP in mixed linear / log space.

    Variables (4n total):
        x[i]  — module center x  (linear)
        y[i]  — module center y  (linear)
        W[i]  — ln(w_i)          (log of width)
        H[i]  — ln(h_i)          (log of height)
    """
    n = len(modules)
    W_F = float(die_width)
    H_F = float(die_height)
    log_ar_min = np.log(min_aspect_ratio)
    log_ar_max = np.log(max_ratio)

    # ---------- symbolic variables ----------
    x = ca.MX.sym('x', n)
    y = ca.MX.sym('y', n)
    W = ca.MX.sym('W', n)      # ln(w_i)
    H = ca.MX.sym('H', n)      # ln(h_i)
    opt_vars = ca.vertcat(x, y, W, H)

    # ---------- bounds & initial guess ----------
    lbx, ubx, x0 = [], [], []
    eps_pos = 0.5               # minimum centre offset from origin

    for i in range(n):
        d = module_data[modules[i]]
        lbx.append(eps_pos);  ubx.append(W_F)
        x0.append(max(d['x'], eps_pos))

    for i in range(n):
        d = module_data[modules[i]]
        lbx.append(eps_pos);  ubx.append(H_F)
        x0.append(max(d['y'], eps_pos))

    for i in range(n):
        d = module_data[modules[i]]
        area = d['area']
        if area > 0:
            # w/h in [r_min, r_max] and w*h = area  =>  w in [sqrt(A*r_min), sqrt(A*r_max)]
            w_min = max(np.sqrt(area * min_aspect_ratio), 0.01)
            w_max = min(np.sqrt(area * max_ratio), W_F)
        else:
            w_min, w_max = 0.01, W_F
        lbx.append(np.log(w_min));  ubx.append(np.log(w_max))
        x0.append(np.log(max(d['w'], 0.01)))

    for i in range(n):
        d = module_data[modules[i]]
        area = d['area']
        if area > 0:
            # h = area / w  => h in [sqrt(A/r_max), sqrt(A/r_min)]
            h_min = max(np.sqrt(area / max_ratio), 0.01)
            h_max = min(np.sqrt(area / min_aspect_ratio), H_F)
        else:
            h_min, h_max = 0.01, H_F
        lbx.append(np.log(h_min));  ubx.append(np.log(h_max))
        x0.append(np.log(max(d['h'], 0.01)))

    # ---------- constraints ----------
    g, lbg, ubg = [], [], []

    # 1. Area:  W_i + H_i  ≥  ln(A_i)     (linear, convex)
    n_area = 0
    for i in range(n):
        area = module_data[modules[i]]['area']
        if area > 0:
            g.append(W[i] + H[i])
            lbg.append(np.log(area));  ubg.append(np.log(area))
            n_area += 1

    # 2. Aspect ratio  (linear, convex)
    #    ln(r_min) <= W_i - H_i <= ln(r_max)
    #    equivalent to r_min <= w_i/h_i <= r_max
    for i in range(n):
        g.append(W[i] - H[i])
        lbg.append(log_ar_min);  ubg.append(log_ar_max)

    # 3. Boundary  (convex: exp(·) is convex)
    #    Left   :  ½ exp(W_i) − x_i  ≤  0
    #    Bottom :  ½ exp(H_i) − y_i  ≤  0
    #    Right  :  x_i + ½ exp(W_i)  ≤  W_F
    #    Top    :  y_i + ½ exp(H_i)  ≤  H_F
    for i in range(n):
        half_w = 0.5 * ca.exp(W[i])
        half_h = 0.5 * ca.exp(H[i])

        g.append(half_w - x[i])
        lbg.append(-ca.inf);  ubg.append(0.0)

        g.append(half_h - y[i])
        lbg.append(-ca.inf);  ubg.append(0.0)

        g.append(x[i] + half_w - W_F)
        lbg.append(-ca.inf);  ubg.append(0.0)

        g.append(y[i] + half_h - H_F)
        lbg.append(-ca.inf);  ubg.append(0.0)

    # 4. Non-overlap HCG:  x_a + ½ exp(W_a) + ½ exp(W_b) − x_b  ≤  0
    for (a, b) in h_edges:
        g.append(x[a] + 0.5 * ca.exp(W[a]) + 0.5 * ca.exp(W[b]) - x[b])
        lbg.append(-ca.inf);  ubg.append(0.0)

    # 5. Non-overlap VCG:  y_a + ½ exp(H_a) + ½ exp(H_b) − y_b  ≤  0
    for (a, b) in v_edges:
        g.append(y[a] + 0.5 * ca.exp(H[a]) + 0.5 * ca.exp(H[b]) - y[b])
        lbg.append(-ca.inf);  ubg.append(0.0)

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


    print(f"  Mixed-space convex model:")
    print(f"    Variables : {opt_vars.size1()}  (n={n}, 4n={4*n})")
    print(f"    Constraints : {len(g)}  "
          f"(area={n_area}, ratio={2*n}, boundary={4*n}, "
          f"H-sep={len(h_edges)}, V-sep={len(v_edges)})")
    print(f"    Nets in objective : {len(nets)}")
    print(f"    α = {alpha:.4f}")

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
            print(f"  ⚠ Solver: {status} — accepting as best feasible")
            return True, sol
        else:
            print(f"  ✗ Solver failed: {status}")
            return False, sol

    except Exception as e:
        import traceback
        print(f"  Solver error: {e}")
        traceback.print_exc()
        return False, None



def extract_solution(sol, modules, module_data, problem_data):
    """Convert optimised variables back to physical coordinates."""
    n = problem_data['n']
    v = sol['x'].full().flatten()

    x_vals = v[0*n : 1*n]
    y_vals = v[1*n : 2*n]
    W_vals = v[2*n : 3*n]
    H_vals = v[3*n : 4*n]

    for i, name in enumerate(modules):
        module_data[name].update({
            'x': float(x_vals[i]),
            'y': float(y_vals[i]),
            'w': float(np.exp(W_vals[i])),
            'h': float(np.exp(H_vals[i])),
        })


# =====================================================================
# I/O & visualisation
# =====================================================================

def save_results(netlist_data, modules, module_data, output_file):
    updated = netlist_data.copy()
    modules_dict = updated.get('Modules', updated)
    if 'Modules' not in updated:
        updated = {'Modules': modules_dict}

    for name in modules:
        if name in modules_dict:
            d = module_data[name]
            modules_dict[name]['rectangles'] = [
                [d['x'], d['y'], d['w'], d['h']]]

    with open(output_file, 'w') as f:
        yaml.dump(updated, f, default_flow_style=False, sort_keys=False)


def calculate_overlap(modules, module_data):
    total = 0.0
    n = len(modules)
    for i in range(n):
        di = module_data[modules[i]]
        for j in range(i + 1, n):
            dj = module_data[modules[j]]
            ox = max(0, min(di['x'] + di['w']/2, dj['x'] + dj['w']/2)
                        - max(di['x'] - di['w']/2, dj['x'] - dj['w']/2))
            oy = max(0, min(di['y'] + di['h']/2, dj['y'] + dj['h']/2)
                        - max(di['y'] - di['h']/2, dj['y'] - dj['h']/2))
            total += ox * oy
    return total


def calculate_hpwl(modules, module_data, terminal_coords, nets):
    total = 0.0
    for (wgt, mod_ids, term_pos) in nets:
        xs, ys = [], []
        for idx in mod_ids:
            d = module_data[modules[idx]]
            xs.append(d['x']);  ys.append(d['y'])
        for (tx, ty) in term_pos:
            xs.append(tx);  ys.append(ty)
        if len(xs) >= 2:
            total += wgt * ((max(xs) - min(xs)) + (max(ys) - min(ys)))
    return total


def visualize(modules, module_data, die_w, die_h,
              output_image, title_extra=''):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-die_w * 0.05, die_w * 1.1)
    ax.set_ylim(-die_h * 0.05, die_h * 1.1)
    ax.set_aspect('equal')

    ax.add_patch(patches.Rectangle(
        (0, 0), die_w, die_h, fill=False, edgecolor='red', linewidth=2))

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(modules), 1)))
    for i, name in enumerate(modules):
        d = module_data[name]
        x0 = d['x'] - d['w'] / 2
        y0 = d['y'] - d['h'] / 2
        ax.add_patch(patches.Rectangle(
            (x0, y0), d['w'], d['h'],
            facecolor=colors[i % 20], edgecolor='black', alpha=0.7))
        ax.text(d['x'], d['y'], name,
                ha='center', va='center', fontsize=5)

    ax.set_title(f'Die {die_w:.1f}×{die_h:.1f}  {title_extra}')
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    plt.close()



def optimize_floorplan(netlist_file, die_file, output_file, output_image,
                       max_iter=500, min_aspect_ratio=1.0, max_ratio=3.0, alpha=None):

    print('=' * 70)
    print('Mixed-Space Convex Floorplan Optimisation')
    print('  Variables: x,y (linear) + W,H (log of dimensions)')
    print('=' * 70)

    netlist_data = load_netlist(netlist_file)
    die_info = load_die_info(die_file)
    die_w = die_info.get('width', 800.0)
    die_h = die_info.get('height', 800.0)

    modules, module_data, terminal_coords = \
        extract_modules_and_terminals(netlist_data)
    n = len(modules)
    print(f"  Modules:   {n}")
    print(f"  Terminals: {len(terminal_coords)}")
    print(f"  Die:       {die_w:.2f} × {die_h:.2f}")

    nets = extract_nets(netlist_data, modules, module_data, terminal_coords)
    print(f"  Nets:      {len(nets)}")

    if alpha is None:
        alpha = max(die_w, die_h) / 20.0
    print(f"  α (LSE smoothing): {alpha:.4f}")
    print(f"  aspect ratio range (w/h): [{min_aspect_ratio:.4f}, {max_ratio:.4f}]")

    if min_aspect_ratio <= 0:
        raise ValueError("min_aspect_ratio must be > 0")
    if max_ratio <= 0:
        raise ValueError("max_ratio must be > 0")
    if min_aspect_ratio > max_ratio:
        raise ValueError("min_aspect_ratio must be <= max_ratio")

    for name in modules:
        d = module_data[name]
        d['w'] = max(d['w'], 0.01)
        d['h'] = max(d['h'], 0.01)
        mx = 0.5 * d['w'] + 0.01
        my = 0.5 * d['h'] + 0.01
        d['x'] = max(mx, min(d['x'], die_w - mx))
        d['y'] = max(my, min(d['y'], die_h - my))
    print()

    print('Building constraint graphs ...')
    h_edges, v_edges = build_constraint_graphs(module_data, modules)
    print()

    print('Setting up mixed-space model ...')
    nlp, pdata = setup_model(
        modules, module_data, terminal_coords, nets,
        die_w, die_h, h_edges, v_edges,
        alpha=alpha, min_aspect_ratio=min_aspect_ratio, max_ratio=max_ratio)
    print()

    print('Solving ...')
    ok, sol = solve_casadi(nlp, pdata, max_iters=max_iter, verbose=True)

    if not ok or sol is None:
        print('\n✗ Optimisation failed — no usable solution.')
        return False

    extract_solution(sol, modules, module_data, pdata)

    hpwl = calculate_hpwl(modules, module_data, terminal_coords, nets)
    overlap = calculate_overlap(modules, module_data)

    save_results(netlist_data, modules, module_data, output_file)
    visualize(modules, module_data, die_w, die_h, output_image,
              title_extra=f'HPWL={hpwl:.1f}  Overlap={overlap:.1f}')

    print(f'\n{"=" * 70}')
    print(f'Optimisation completed!')
    print(f'  HPWL:    {hpwl:.2f}')
    print(f'  Overlap: {overlap:.2f}')
    print(f'  Output:  {output_file}, {output_image}')
    print(f'{"=" * 70}\n')
    return True


# =====================================================================
# CLI
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description='Mixed-Space Convex Floorplan Optimisation')
    p.add_argument('--netlist', required=True, help='Netlist YAML file')
    p.add_argument('--die', required=True, help='Die info YAML file')
    p.add_argument('--output', default='output_cvx_legalizer.yaml')
    p.add_argument('--output-image', default='output_cvx_legalizer.png')
    p.add_argument('--max-iter', type=int, default=500)
    p.add_argument('--max-ratio', type=float, default=3.0,
                   help='Max aspect ratio ρ')
    p.add_argument('--min-aspect-ratio', type=float, default=0.3,
                   help='Min aspect ratio for w/h (default 0.3)')
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