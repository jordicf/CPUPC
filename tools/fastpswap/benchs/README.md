### Example 

To compare the cached version with the non-cached version, run from top level folder

```{bash}
time python3.11 -m tools.fastpswap.fastpswap tools/fastpswap/benchs/test200.yaml -o a.yaml --seed 42
```

Then change in `fastpswap.py` the line `from .anneal_cached import simulated_annealing` to `from .anneal import simulated_annealing`

Then run again

```{bash}
time python3.11 -m tools.fastpswap.fastpswap tools/fastpswap/test200.yaml -o b.yaml --seed 42
```

The second run should take much longer.

You can check the output of both runs is the exact same

```{bash}
diff a.yaml b.yaml
```

This should print nothing, meaning both files are identical.

Then run `rm a.yaml b.yaml` to clean the generated files


#### Results

| Version | Execution Time | Speedup (vs Plain) | Speedup (vs Non-Cached Numba)
| :--- | :--- | :--- | :--- |
| Cached Numba | 5.1s | 27x | 16x |
| Non-Cached Numba | 1m 22s | 1.7x | 1.0x |
| Pure Python | 2m 17s | 1.0x |  0.6x |