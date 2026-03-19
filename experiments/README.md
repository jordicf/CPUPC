RESULTS — CPUPC Tools (GSRC & MCNC)
=================================

Overview
--------
These visual results were generated with the `tools` suite in this repository using the GSRC and MCNC benchmark sets. They illustrate how different whitespace targets and die aspect ratios affect placement/layout.

Naming convention
-----------------
- Filenames or directories containing `0.85` indicate a 15% whitespace target (i.e., 85% target utilization).
- Filenames or directories containing `0.9` or `0.90` indicate a 10% whitespace target (i.e., 90% target utilization).
- A numeric suffix such as `_1`, `_2`, `_3`, `_4` denotes different die aspect-ratio variants. For example:
  - `ami49_0.9_die.yaml` and `ami49_0.9.yaml`: the die file and netlist for ami49 at 10% whitespace.
  - `ami49_0.85_2.yaml` / `ami49_die_2.yaml`: the `2` refers to the second die aspect-ratio variant (see folders like `1-1-0.85`, `2-1-0.85`, etc.).

Directory overview (examples)
-----------------------------
- `outputs/benchmarks/0.15_whitespace/` — benchmark netlists and die files for 15% whitespace (0.85).
- `outputs/benchmarks/0.1_whitespace/`  — benchmark netlists and die files for 10% whitespace (0.9).
- `outputs/test_results/` — final layout outputs for different datasets (e.g., `ami33_test`, `ami49_test`, `n100_test`). Final results typically end with `_final.yaml` and visualizations are saved in `Figs/` or sibling folders.

Understanding filenames
-----------------------
- Suffix `0.85` or `0.9` indicates whitespace target (15% and 10% respectively).
- Suffix `_1`, `_2`, `_3`, `_4` corresponds to different die aspect-ratio variants.
- Files named `*_final.yaml` contain the final netlist/layout produced by the tools.

Reproducing results (quick guide)
--------------------------------
You can re-run the tools locally using the command-line utilities in this repo. Two convenient options:

1) Run without installing CLI entry point (recommended for development):

```bash
# run the `run` tool via python -m (no PATH changes required)
python -m tools.cpupc_tools run --flow qp --die <DIE_FILE> --out <OUT_FILE> <NETLIST_FILE>

# example:
python -m tools.cpupc_tools run --flow qp --die outputs/benchmarks/0.15_whitespace/1-1-0.85/n100_die.yaml --out results_n100.yaml outputs/benchmarks/0.15_whitespace/n100.yaml
```

2) Install project in editable mode and use the `cpupc` command (optional):

```bash
pip install -e .
# then use the CLI if the script directory is on your PATH
cpupc run --flow qp --die <DIE_FILE> --out <OUT_FILE> <NETLIST_FILE>
```

Notes:
- Ensure required dependencies are installed (see `pyproject.toml` or `requirements.txt`).
- Keep argument order: pass optional flags (e.g., `--flow`, `--die`, `--out`) before the positional `netlist` argument.

Acknowledgements and citation
-----------------------------
These results were produced by algorithms implemented in this repository's `tools` modules and use GSRC and MCNC benchmark data. Please cite this project and the relevant benchmark sources when using these results in publications.

Contact & reproducibility
-------------------------
For exact hyperparameters, random seeds, or further reproduction details, consult the tool help or contact the repository maintainer:

```bash
python -m tools.cpupc_tools run --help
python -m tools.inirects.inirects --help
```