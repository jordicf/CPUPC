## Contraction-Expansion based floorplan

Relocates and reshapes non-fixed modules using the contraction-expansion algorithm. The contraction part of the algorithm is the fastpswap tool.

### Overview

The purpose of this tool is to generate initial layouts for other tools, to improve the quality of their results. The solutions that are found are not legal, but overlap is generally small.

It takes as input a die file, a netlist file, which must have initialized positions, and it produces a new netlist file, with positions and shapes changed. Note that this tool produces *non.deterministic* results, unless a seed is specified.

### Usage

#### Basic Command

```bash
cpupc inirects \
    --netlist <netlist.yaml> \
    --die <die.yaml> \
    --output <output.yaml>
```

#### Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--netlist` | str | *required* | Input netlist YAML file |
| `--die` | str | *required* | Die information YAML file |
| `--output` | str | *required* | Output netlist YAML file |
| `--patience` | int | 1 | Number of iterations without HPWL improvement to stop. |
| `--overlap_tolerance` | float | 1e-3 | Minimum relative change to stop the main expansion phase (the lower, the less overlap but more cpu time) |
| `--seed` | int | None | Seed for contraction phase|
| `--swaps` | int | 100 | Number of swaps to perform in the contraction phase (see fastpswap)|
| `--split_threshold` | float | 0.5 | Threshold for splitting rectangles in the contraction phase (see fastpswap)|
| `--star` | int | 1 | Use star model for split nets (1: True, 0: False; see fastpswap)|


### Dependencies
- **Python 3**
- **Numpy**
- **NetworkX**