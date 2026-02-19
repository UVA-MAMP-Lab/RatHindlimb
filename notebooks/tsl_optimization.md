---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

```python
import opensim as osim
from osimpy import OsimGraph
from tsl_optimization import optimize_fiber_length, calc_tsl
from itertools import product
from pathlib import Path

project_dir = Path('../')
model_dir = project_dir / 'models' / 'osim'
data_dir = project_dir / 'data'
model_file = model_dir / 'rat_hindlimb_millard_y2j_knee_fixed_markers_params.osim'
graph = OsimGraph.from_file(str(model_file))
```

```python
import scipy.io as sio
import polars as pl
import numpy as np

control = sio.loadmat(str(data_dir / 'motion' / 'Control.mat'))
baseline_right_ik = control['Timepoints']['Baseline'][0,0]['Phases'][0,0]['RightStanceSwing'][0,0]['IK'][0,0]
avg_right_ik = baseline_right_ik['Average'][0,0]*np.pi/180
std_right_ik = baseline_right_ik['StdDev'][0,0]*np.pi/180

# Problem: Currently scipy io cannot access MATLAB string arrays. Might just have to hard code it
# ik_columns = control['Info'][0,0]['IKLabels'][0].tolist()
# print(ik_columns)
ik_columns = [
    'time', 'sacrum_pitch', 'sacrum_roll', 'sacrum_yaw', 'sacrum_x', 'sacrum_y', 'sacrum_z',
    'sacroiliac_r_flx', 'hip_r_flx', 'hip_r_add', 'hip_r_int', 'knee_r_flx', 'ankle_r_flx',
    'ankle_r_add', 'ankle_r_int', 'sacroiliac_l_flx', 'hip_l_flx', 'hip_l_add', 'hip_l_int',
    'knee_l_flx', 'ankle_l_flx', 'ankle_l_add', 'ankle_l_int'
]

avg_right_ik_df = pl.DataFrame(avg_right_ik, schema=ik_columns)
std_right_ik_df = pl.DataFrame(std_right_ik, schema=ik_columns)

coords = ["hip_r_flx", "hip_r_add", "hip_r_int", "knee_r_flx", "ankle_r_flx"]
n_coords = len(coords)
res = 1
n_std = 1
n_combos_per_step = res**n_coords

avg_right_coords = avg_right_ik_df[coords]
std_right_coords = std_right_ik_df[coords]
n_rows = avg_right_coords.shape[0]
n_combos = n_rows*n_combos_per_step

ub = avg_right_coords + n_std*std_right_coords
lb = avg_right_coords - n_std*std_right_coords
dist = np.linspace(lb, ub, res) # 1st dimension is the resolution, 2nd dimension is the timesteps, 3rd dimension is the coords
coord_combos = np.array([list(product(*dist[:, i, :].T)) for i in range(n_rows)]).reshape(n_combos, n_coords)
walk_df = pl.DataFrame(coord_combos, schema=coords)
```

```python
# Normalized fiber length range for optimization
lm_norm_range = (0.5, 1.5) 
lm_walk_range = (0.6, 1.2)

print("Getting full ROM lengths...")
results_full = graph.get_all_muscle_lengths_rom(min_points=100)
print("Got full ROM lengths")

print("Getting walk lengths...")
lengths_walk = graph.get_muscle_lengths_from_data(graph.get_muscle_names(), walk_df)
print("Got walk lengths")

tsl_results = {}
for muscle_name, lmt in results_full.items():
  muscle: osim.Muscle = graph.get_muscle(muscle_name) 
  lm_opt = float(muscle.get_optimal_fiber_length())
  lm_range = lm_opt * np.asarray(lm_norm_range)
  alpha_opt = float(muscle.get_pennation_angle_at_optimal())
  
  millard: osim.Millard2012EquilibriumMuscle = osim.Millard2012EquilibriumMuscle.safeDownCast(muscle)
  afl = millard.getActiveForceLengthCurve()
  pfl = millard.getFiberForceLengthCurve()
  tfl = millard.getTendonForceLengthCurve()

  lmt_full = np.clip(np.sort(np.unique(lmt.select(pl.col(muscle_name)))), 1.0e-6, None) # Ensure lmt is sorted, unique, and non-zero
  lm = optimize_fiber_length(lmt_full, lm_opt, alpha_opt, afl, pfl, tfl, lm_norm_range)
  tsl = calc_tsl(lmt_full, lm, lm_opt, alpha_opt, afl, pfl, tfl)

  lmt_walk = np.clip(np.sort(np.unique(lengths_walk.select(pl.col(muscle_name)))), 1.0e-6, None)
  lm_walk = optimize_fiber_length(lmt_walk, lm_opt, alpha_opt, afl, pfl, tfl, lm_walk_range)
  tsl_walk = calc_tsl(lmt_walk, lm_walk, lm_opt, alpha_opt, afl, pfl, tfl)
  

  tsl_results[muscle_name] = {
      'lmt_full': lmt_full,
      'lm_opt': lm_opt,
      'lm': lm,
      'tsl': tsl,
      'lmt_walk': lmt_walk,
      'lm_walk': lm_walk,
      'tsl_walk': tsl_walk
  }
```

```python
#| label: tbl-tsl-comparison
from IPython.display import Markdown as md
from tabulate import tabulate
import pandas as pd
import numpy as np

johnson_params = pd.read_csv(str(data_dir / "parameters" / "johnson_2011_parameters.csv")).set_index("Abbreviation")
johnson_tsl_mm = johnson_params["lts (mm)"]
# johnson_std = johnson_params[]

tsl_rows = []
for muscle_name, res in tsl_results.items():
    abbrev = muscle_name.split("_", 1)[1] if "_" in muscle_name else muscle_name
    tsl_rows.append(
        {
            "Abbreviation": abbrev,
            "Johnson TSL (mm)": johnson_tsl_mm.get(abbrev, np.nan),
            "Full ROM TSL (mm)": np.mean(res["tsl"]) * 1000,
            "Walk TSL (mm)": np.mean(res["tsl_walk"]) * 1000,
        }
    )

tsl_df = (
    pd.DataFrame(tsl_rows)
    .sort_values("Abbreviation")
    .set_index("Abbreviation")
)
tsl_df.to_csv(data_dir / 'parameters' /'tsl_comparison.csv')

md(tabulate(tsl_df, headers="keys", tablefmt="pipe", floatfmt=".2f"))
```
