# Vehicle-Dynamics-Guided Two-Stage PINN for Track Irregularity Identification

A unified implementation of a two-stage vehicle-dynamics-guided
physics-informed neural network (PINN) for track irregularity
identification through wheelset displacement reconstruction.

The framework reconstructs wheelset displacement through an intermediate
vehicle dynamic response:

\[ A_c `\rightarrow `{=tex}A_b `\rightarrow `{=tex}Z_w \]

where:

- (A_c): car-body acceleration response
- (A_b): bogie acceleration response
- (Z_w): wheelset displacement

The first stage estimates the intermediate bogie response, and the
second stage reconstructs wheelset displacement.

This repository provides a unified implementation of two training modes:

- **PINN-La**: intermediate bogie acceleration supervision enabled
- **PINN-Lb**: intermediate bogie acceleration supervision disabled

------------------------------------------------------------------------

## Repository Structure

    Vehicle-Dynamics-Guided-Two-Stage-PINN/
    ├── data/                            # Raw vehicle response data (data root)
    │   ├── Acc_all_4500m_Speed80.txt    # Car-body + bogie acceleration responses
    │   └── Dis_all_4500m_Speed80.txt    # Wheelset displacement (training target)
    ├── model/                           # Model source code
    │   └── PINN-La_Lb.py                # Training script (unified PINN-La / PINN-Lb)
    ├── requirements.txt                 # Python dependencies
    └── README.md

Results are written next to the script, i.e. under
`model/<result_folder_name>/PINN-La/` or `.../PINN-Lb/`, and are created
automatically on the first run.

### The `data/` directory

`data/` is the **data root** of the project. It holds the raw
vehicle-dynamics signals used for training, and it is the directory that
`CONFIG["data_dir"]` must point to:

```python
"data_dir": r".../Vehicle-Dynamics-Guided-Two-Stage-PINN/data"
```

The script builds the full input paths by joining `data_dir` with the
filenames below, so `data_dir` must be the folder that **directly**
contains the `.txt` files — pointing it at the repository root instead
raises `FileNotFoundError`.

File naming convention:

    Acc_all_4500m_Speed{speed}.txt
    Dis_all_4500m_Speed{speed}.txt

- `{speed}` is the running speed in km/h and is substituted
  automatically from `CONFIG["speed"]`.
- Every speed requires **one acceleration file and one displacement
  file** as a pair. The shipped example is `speed = 80` (80 km/h), which
  matches the default `"speed": 80` in `CONFIG`.
- To train on another speed, add the corresponding pair to `data/` and
  set `"speed"` accordingly.

File properties:

- Plain whitespace-separated ASCII tables, loaded with `np.loadtxt`.
- One row per sampling instant.
- The acceleration and displacement files must contain the **same number
  of rows** so that both channels can be aligned in time.
- For the shipped 80 km/h case: 4500 m of track sampled at
  `fs = 2000 Hz`, i.e. roughly 4×10^5 rows per file (~60 MB each).

Only the 80 km/h pair is bundled with this repository. Additional speeds
are not distributed and must be supplied by the user.

------------------------------------------------------------------------

## 1. Model Variants

The training mode is controlled by:

```python
"model_variant": "La"
```

or

```python
"model_variant": "Lb"
```

### PINN-La

PINN-La uses measured bogie acceleration as intermediate supervision.

Loss:

\[ L\_{La}=L\_{wheel}+`\lambda `{=tex}L\_{bogie}+`\mu `{=tex}L\_{reg} \]

Features:

- Bogie acceleration measurement is required.
- The predicted bogie response is supervised.
- The intermediate reconstruction error contributes to optimization.

------------------------------------------------------------------------

### PINN-Lb

PINN-Lb removes intermediate bogie supervision while retaining the same
two-stage architecture.

Loss:

\[ L\_{Lb}=L\_{wheel}+`\mu `{=tex}L\_{reg} \]

Features:

- Bogie supervision is disabled.
- The network still reconstructs the intermediate bogie response
  internally.
- If bogie acceleration measurements are available, they are only used
  for diagnostic evaluation.
- Without bogie measurements, PINN-Lb can still be trained by setting
  `"bogie_col": None`.

------------------------------------------------------------------------

## 2. Data Format

All input files are read from the `data/` directory described above.
Both files are whitespace-separated ASCII tables with one sample per row
and must have the same number of rows.

### Acceleration Data

Example:
    data/Acc_all_4500m_Speed80.txt

Columns:

  Column   Description                      Required

  -------- -------------------------------- --------------------------------------------

  0        Car-body vertical acceleration   Yes
  1        Car-body pitch acceleration      Yes
  2        Bogie acceleration               Required for PINN-La; optional for PINN-Lb

Column indices are configurable via `"zc_col"`, `"beta_col"` and
`"bogie_col"` in `CONFIG`.

For PINN-Lb without bogie measurements:

```python
"bogie_col": None
```

------------------------------------------------------------------------

### Wheelset Displacement Data

Example:
    data/Dis_all_4500m_Speed80.txt

The wheelset displacement is used as the training target.

Columns:

  Column   Description               Required

  -------- ------------------------- ----------

  6        Wheelset displacement     Yes

The column index is configurable via `"wheel_col"` in `CONFIG`
(default `6`). Only this column is read; any other columns in the file
are ignored.

------------------------------------------------------------------------

## 3. Data Processing Pipeline

The preprocessing procedure is:
    Raw data
        |
        v
    Train/Validation/Test split
        |
        v
    Low-pass filtering
        |
        v
    Down-sampling
        |
        v
    Z-score normalization
        |
        v
    Sliding-window segmentation
        |
        v
    PINN training

Normalization parameters are calculated only from the training set.

------------------------------------------------------------------------

## 4. Configuration

Main parameters are defined in:

```python
CONFIG
```

Example:

```python
CONFIG = {
    "model_variant": "La",
    "data_dir": r".../Vehicle-Dynamics-Guided-Two-Stage-PINN/data",
    "speed": 80,
    "highcut": 2,
    "epochs": 300,
}
```

Switch modes:

PINN-La:

```python
"model_variant": "La"
```

PINN-Lb:

```python
"model_variant": "Lb"
```

------------------------------------------------------------------------

## 5. Running

Run:

```bash
python model/PINN-La_Lb.py
```

The program performs:

1. Data loading
2. Signal preprocessing
3. Model training
4. Best model selection
5. Performance evaluation
6. Result saving

------------------------------------------------------------------------

## 6. Output Files

With `"result_base_dir": None`, results are written next to the script:

    model/Simulation_PINN_Results/
    └── PINN-La/                          # or PINN-Lb/, depending on model_variant
        └── 80km_h-2Hz/                   # {speed}km_h-{highcut}Hz
            ├── run1/                     # one folder per independent run
            │   ├── best_model.pth
            │   ├── train_results.mat
            │   └── test_results.mat
            ├── run2/
            ├── ...
            └── runs_summary/
                ├── parameters.txt        # the resolved CONFIG of this experiment
                └── run_summary.txt       # per-run metrics + mean/std over runs

`best_model.pth` contains:

- trained network parameters
- model configuration
- normalization statistics

The `.mat` files hold the de-normalized predictions and targets plus the
loss histories. When no bogie labels are available, the internal bogie
prediction is stored as `bogie_pred_normalized` (normalized scale), since
it cannot be converted back to physical units.

------------------------------------------------------------------------

## 7. Frequency-domain Integration

Frequency-domain integration is used to reconstruct velocity and
displacement states.

For non-zero frequency components:

\[ `\hat{v}`{=tex}\_k=`\frac{\hat{a}_k}{j\omega_k}`{=tex} \]

The zero-frequency component is separately treated and set to zero
before inverse Fourier transformation.

------------------------------------------------------------------------

## 8. Notes

PINN-La requires:
    Car-body acceleration
    +
    Bogie acceleration
    +
    Wheelset displacement

PINN-Lb requires:
    Car-body acceleration
    +
    Wheelset displacement

Bogie acceleration is optional in PINN-Lb and does not participate in
optimization.
