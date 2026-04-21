# (c) Yilhamujiang Yimamu 2026
# For the CPUPC Project.
# Licensed under the MIT License (see https://github.com/jordicf/CPUPC/blob/master/LICENSE.txt).

"""Unified CPUPC flow runner.

Usage:
    cpupc run --flow FLOW [global-options(die,netlist,output)] [flow-options] 

Flows:
    qp              Quadratic placement (initial floorplanning)
    inirects        Initial rectangles (contract & expand)
    cvx_legalizer   Convex legalizer
    fine_tune       Non-convex legalizer
"""

import argparse
import os
from time import time
from typing import Any
from tools.inirects import inirects
from tools.legalrect.cvx_legalizer import optimize_floorplan
from tools.legalrect import glb_legalizer
from tools.inipoints.qp import (PlacementDB, QuadraticPlacer,
                           visualize_placement, verify_fixed_modules)

FLOW_CHOICES = ["qp", "inirects", "cvx_legalizer", "fine_tune"]


def _add_qp_opts(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("qp flow options")
    g.add_argument("--max-iters", type=int, default=200,
                   help="Max QP iterations (default: 200)")
    g.add_argument("--tolerance", type=float, default=1e-6,
                   help="Convergence tolerance (default: 1e-6)")
    g.add_argument("--verify", action="store_true",
                   help="Verify fixed modules were not moved")


def _add_inirects_opts(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("inirects flow options")
    g.add_argument("--overlap_tolerance", type=float, default=1e-4,
                   help="Expansion phase overlap tolerance")
    g.add_argument("--patience", type=int, default=5,
                   help="Iterations w/o HPWL improvement before stopping")
    g.add_argument("--seed", type=int, default=5,
                   help="RNG seed for simulated annealing")
    g.add_argument("--swaps", type=int, default=100,
                   help="Swaps per SA iteration")
    g.add_argument("--split_threshold", type=float, default=0.5,
                   help="Area threshold for module splitting (0-1)")
    g.add_argument("--star", type=int, default=1,
                   help="Use star model for split nets (1=yes, 0=no)")


def _add_cvx_opts(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("cvx_legalizer flow options")
    g.add_argument("--max-iter", type=int, default=500,
                   help="Max IPOPT iterations (default: 500)")
    g.add_argument("--min-aspect-ratio", type=float, default=None,
                   help="Min aspect ratio fallback for w/h (default: None)")
    g.add_argument("--max-ratio", type=float, default=None,
                   help="Max aspect ratio fallback (default: None)")
    g.add_argument("--alpha", type=float, default=1.0,
                   help="LSE smoothing parameter (default: 1.0)")


def _add_fine_tune_opts(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("fine_tune flow options")
    g.add_argument("--max-ratio", type=float, default=3.0,
                   help="Max aspect ratio")
    g.add_argument("--num-iter", type=int, default=15,
                   help="Legalizer iterations")
    g.add_argument("--radius", type=float, default=1.0,
                   help="No-overlap distance multiplier")
    g.add_argument("--wl-mult", type=float, default=1.0,
                   help="HPWL weight multiplier")
    g.add_argument("--tau-initial", type=float, default=None,
                   help="Initial tau for soft constraints")
    g.add_argument("--tau-decay", type=float, default=0.3,
                   help="Tau decay factor")
    g.add_argument("--otol-initial", type=float, default=1e-1)
    g.add_argument("--otol-final", type=float, default=1e-4)
    g.add_argument("--rtol-initial", type=float, default=1e-1)
    g.add_argument("--rtol-final", type=float, default=1e-4)
    g.add_argument("--tol-decay", type=float, default=0.5)
    g.add_argument("--plot", action="store_true",
                   help="Visualise each iteration")
    g.add_argument("--plot-dir", type=str, default="plots")
    g.add_argument("--palette-seed", type=int, default=None)


_FLOW_OPT_FNS: dict[str, Any] = {
    "qp": _add_qp_opts,
    "inirects": _add_inirects_opts,
    "cvx_legalizer": _add_cvx_opts,
    "fine_tune": _add_fine_tune_opts,
}


def _run_qp(opts: dict[str, Any]) -> None:

    db = PlacementDB()
    db.load_yaml(opts["netlist"], opts["die"])

    qp = QuadraticPlacer(db)
    qp_time = qp.solve(max_iters=opts["max_iters"],
                        tolerance=opts["tolerance"])

    db.save_yaml(opts["output"])

    img = opts.get("output_image") or (
        os.path.splitext(opts["output"])[0] + ".png")
    visualize_placement(db, img, title="QP Placement Result")

    if opts.get("verify"):
        all_ok, _ = verify_fixed_modules(db, opts["netlist"])
        if not all_ok:
            print("Warning: some fixed modules were moved!")

    hpwl = db.calc_hpwl()
    print(f"HPWL = {hpwl:.2f}  |  QP solve: {qp_time:.3f}s")


def _run_inirects(opts: dict[str, Any]) -> None:
    

    tool_args = [
        "--netlist", opts["netlist"],
        "-d", opts["die"],
        "--output", opts["output"],
        "--overlap_tolerance", str(opts["overlap_tolerance"]),
        "--patience", str(opts["patience"]),
        "--swaps", str(opts["swaps"]),
        "--split_threshold", str(opts["split_threshold"]),
        "--star", str(opts["star"]),
    ]
    if opts.get("seed") is not None:
        tool_args += ["--seed", str(opts["seed"])]
    
    inirects.main(prog="cpupc run --flow inirects", args=tool_args)


def _run_cvx_legalizer(opts: dict[str, Any]) -> None:

    img = opts.get("output_image") or (
        os.path.splitext(opts["output"])[0] + ".png")
    optimize_floorplan(
        opts["netlist"], opts["die"], opts["output"], img,
        max_iter=opts["max_iter"],
        min_aspect_ratio=opts.get("min_aspect_ratio"),
        max_ratio=opts["max_ratio"],
        alpha=opts.get("alpha"),
    )


def _run_fine_tune(opts: dict[str, Any]) -> None:

    tool_args = [
        opts["netlist"],
        opts["die"],
        "--max_ratio", str(opts["max_ratio"]),
        "--num_iter", str(opts["num_iter"]),
        "--radius", str(opts["radius"]),
        "--wl_mult", str(opts["wl_mult"]),
        "--tau_decay", str(opts["tau_decay"]),
        "--otol_initial", str(opts["otol_initial"]),
        "--otol_final", str(opts["otol_final"]),
        "--rtol_initial", str(opts["rtol_initial"]),
        "--rtol_final", str(opts["rtol_final"]),
        "--tol_decay", str(opts["tol_decay"]),
        "--outfile", opts["output"],
    ]
    if opts.get("tau_initial") is not None:
        tool_args += ["--tau_initial", str(opts["tau_initial"])]
    if opts.get("palette_seed") is not None:
        tool_args += ["--palette_seed", str(opts["palette_seed"])]
    if opts.get("verbose"):
        tool_args.append("--verbose")
    if opts.get("plot"):
        tool_args.append("--plot")
        tool_args += ["--plot_dir", opts.get("plot_dir", "plots")]
    glb_legalizer.main(prog="cpupc run --flow fine_tune", args=tool_args)


_FLOW_RUNNERS: dict[str, Any] = {
    "qp": _run_qp,
    "inirects": _run_inirects,
    "cvx_legalizer": _run_cvx_legalizer,
    "fine_tune": _run_fine_tune,
}


# =====================================================================
# CLI parser
# =====================================================================

def _build_base_parser(
    prog: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Unified CPUPC flow runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""flows:
  qp              Quadratic placement (initial floorplanning)
  inirects        Initial rectangles (contract & expand)
  cvx_legalizer   Mixed-space convex legalizer
  fine_tune       CasADi-based fine-tuning legalizer

examples:
  cpupc run --flow qp --die die.yaml --out result.yaml netlist.yaml
  cpupc run --flow cvx_legalizer --max-iter 800 --die die.yaml --out out.yaml netlist.yaml
  cpupc run --flow fine_tune --num-iter 20 --die die.yaml netlist.yaml
""",
    )
    parser.add_argument("--flow", required=True, choices=FLOW_CHOICES,
                        help="Flow to execute")
    parser.add_argument("--die", required=True,
                        help="Die file (.yaml) or WIDTHxHEIGHT")
    parser.add_argument("--out", "--output", dest="output", default=None,
                        help="Output file (default: <flow>_<netlist>.yaml)")
    parser.add_argument("--output-image", default=None,
                        help="Output visualisation image")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print timings and extra info")
    parser.add_argument("netlist", help="Input netlist file (.yaml / .json)")
    return parser


# =====================================================================
# Entry point
# =====================================================================

def main(prog: str | None = None, args: list[str] | None = None) -> None:
    """Main function."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--flow", choices=FLOW_CHOICES)
    known, _ = pre.parse_known_args(args)

    parser = _build_base_parser(prog)

    if known.flow is not None:
        _FLOW_OPT_FNS[known.flow](parser)

    opts = vars(parser.parse_args(args))
    flow: str = opts["flow"]

    if opts["output"] is None:
        base = os.path.splitext(os.path.basename(opts["netlist"]))[0]
        opts["output"] = f"{flow}_{base}.yaml"

    if opts["verbose"]:
        print(f"Flow: {flow}")
        print(f"  netlist : {opts['netlist']}")
        print(f"  die     : {opts['die']}")
        print(f"  output  : {opts['output']}")
        start = time()

    _FLOW_RUNNERS[flow](opts)

    if opts["verbose"]:
        print(f"Total elapsed time: {time() - start:.3f}s")


if __name__ == "__main__":
    main()
