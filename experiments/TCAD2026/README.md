 TCAD2026 RESULTS — CPUPC Tools (GSRC & MCNC)
=================================

Overview
--------
These visual results were generated with the `tools` suite in this repository using the GSRC and MCNC benchmark sets. They illustrate how different whitespace targets and die aspect ratios affect placement/layout.

Naming convention
-----------------
- Directory names `0.15whitespace` indicate a 15% whitespace target; `0.1whitespace` indicate a 10% whitespace target.
- Benchmark files remain GSRC-style: `<benchmark>.blocks`, `<benchmark>.nets`, and `<benchmark>_initial.pl`.
- In 15% whitespace folders, suffixes `_1`, `_2`, `_3`, `_4` on `*_final.pl` denote different die aspect-ratio variants (e.g., `ami49_1_final.pl`, `ami49_2_final.pl`).
- In 10% whitespace folders, final output is stored as `<benchmark>_0.9_final.pl` (single run setting in current dataset).

Directory overview (examples)
-----------------------------
- `experiments/TCAD2026/<benchmark>_test/0.15whitespace/` — benchmark inputs (`.blocks`, `.nets`, `_initial.pl`) and final `.pl` outputs for 15% whitespace.
- `experiments/TCAD2026/<benchmark>_test/0.1whitespace/` — benchmark inputs (`.blocks`, `.nets`, `_initial.pl`) and final `.pl` outputs for 10% whitespace.
- Current dataset in this directory uses `.pl` placement files; there are no YAML netlist/result files in these folders.

Understanding filenames
-----------------------
- Folder name `0.15whitespace` or `0.1whitespace` indicates whitespace target (15% and 10% respectively).
- Suffix `_1`, `_2`, `_3`, `_4` on `*_final.pl` corresponds to different die aspect-ratio variants (used in current 15% whitespace sets).
- Files named `*_initial.pl` are initial placements; files named `*_final.pl` are final placements, both in GSRC `.pl` format.

Reproducing results (quick guide)
--------------------------------
You can re-run the tools locally using the command-line utilities in this repo. Two convenient options:

1) Run without installing CLI entry point (recommended for development):

```bash
# run the `run` tool via python -m (no PATH changes required)
python -m tools.cpupc_tools run --flow qp <NETLIST_FILE> --die <DIE_FILE> --out <OUT_FILE> 

# example:
python -m tools.cpupc_tools run --flow qp results_n100.yaml experiments/TCAD2026/benchmarks/0.15_whitespace/n100.yaml --die experiments/TCAD2026/benchmarks/0.15_whitespace/1-1-0.85/n100_die.yaml --out <OUT_FILE> 
```

2) Install project in editable mode and use the `cpupc` command (optional):

```bash
pip install -e .
# then use the CLI if the script directory is on your PATH
cpupc run --flow qp  <NETLIST_FILE> -die <DIE_FILE> --out <OUT_FILE> 
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