# SCUDEM XI 2026, Problem B: group dynamics in foraging

Team 26100. A continuous-time Markov chain model of two spider monkeys that can tip each other off about food.
The math (parameters, states, diagrams, equations) is in `writeup/model.pdf`.

## Files
- `ctmc.py`: the model. `PARAMS` at the top holds every chosen parameter with its units, and `deriv()` is the 14 equations.
  `run(params)` returns the steady-state intake of A and B.
- `analysis.py`: the sweeps and figures (distance, distance × accuracy, lying, punishment, patch size, bits, which factors matter).
  It prints its numbers and saves plots to `img/an_*.png`.
- `img/`: figures. `an_overview.png` has the main six on one page.
- `writeup/`: the model notes (LaTeX source and PDF).

## Run it
With [uv](https://docs.astral.sh/uv/):
```
uv run --no-project --with numpy --with matplotlib python ctmc.py
uv run --no-project --with numpy --with matplotlib python analysis.py
```
Or with plain pip: `pip install -r requirements.txt`, then `python ctmc.py` / `python analysis.py`. Each takes a few seconds.

## Try your own
```python
from ctmc import PARAMS, run
talk = run(PARAMS | {'d': 20, 'pA': 0.8, 'pB': 0.8})    # change any parameter
silent = run(PARAMS | {'d': 20, 'C': 0})                # C = 0 means no tips
print(talk, silent)
```
`'phi': 1e6` turns punishment off (feuds end instantly), which gives back the 9-state model.
