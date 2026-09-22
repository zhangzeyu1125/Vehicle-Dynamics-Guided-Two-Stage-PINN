"""
======================================================================
Two-Stage Physics-Informed Neural Network: Unified Training of
PINN-La / PINN-Lb
======================================================================

[Description]
This program selects whether to use intermediate bogie supervision via
`model_variant` in CONFIG:
1. Load vehicle response and wheelset displacement data;
2. Split the data into training / validation / test sets according to the
   specified ratios;
3. Apply low-pass filtering and downsampling to each signal, and normalize
   using training-set statistics only;
4. Stage 1: obtain a/v/d states from carbody vertical / pitch acceleration
   via FFT-based frequency-domain integration, and predict bogie acceleration
   through the custom Stage-1 cell; La applies supervision to it, while Lb
   only records the diagnostic error when labels are available;
5. Stage 2: fuse the carbody states with the bogie a/v/d states predicted in
   Stage 1 to identify the wheelset displacement (track irregularity)
   corresponding to the end of the sequence;
6. PINN-La: wheel MSE + decay weight x bogie MSE + L1 regularization;
   PINN-Lb: wheel MSE + L1 regularization, with bogie labels excluded from
   backpropagation;
7. La requires ground-truth bogie acceleration columns; Lb may omit them
   (they can still be retained for diagnostic purposes);
   Ground-truth bogie acceleration is never used as a network input feature
   in either mode;
8. Both modes select the best model by validation wheelset loss, with a fixed
   number of training epochs given by CONFIG["epochs"];
9. Each mode saves best_model.pth, MAT numerical results, and a run summary
   separately;

[Data Format]
- Acceleration file (default):
    Column 0: carbody vertical acceleration
    Column 1: carbody pitch acceleration
    Column 2: bogie acceleration (required for La; optional for Lb - when
              absent, the file may contain only columns 0 and 1)
- Displacement file (default):
    Column 6: wheelset displacement
All column indices above can be modified in CONFIG. For Lb, if the existing
bogie column should not be read, set bogie_col=None.

[Usage]
- All paths, experiment parameters, training hyperparameters, model
  hyperparameters, data splits, filtering parameters, and optimizer
  parameters that require manual adjustment are centralized in the CONFIG
  section below.
- When switching operating conditions, the following are usually modified
  first:
    model_variant ("La" or "Lb"), data_dir / speed / highcut
    The {speed} placeholder in data filenames changes automatically with
    `speed`.
- If result_base_dir = None, results are saved by default to
  "current script directory / result_folder_name";
  if a full directory is specified, that directory is used as the result root.
- In either case, results are written into separate PINN-La / PINN-Lb
  subdirectories to avoid overwriting.
- Current default settings: 80 km/h, 2 Hz, 5 independent runs;
  300 full training epochs per run.
======================================================================
"""

import copy
import os
import random
import time

import numpy as np
import scipy.io as sio
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.signal import butter, filtfilt, resample_poly
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset

# ============================================================
# 1. Configuration
# ============================================================
CONFIG = {
    # Model variant: La = bogie supervision; Lb = no bogie supervision
    "model_variant": "La",  # La: requires ground-truth bogie labels; Lb: labels are optional and used for diagnostics only

    # Paths and filenames
    'data_dir': r"C:\Users\Lily\Desktop\Vehicle-Dynamics-Guided-Two-Stage-PINN",
    'input_filename': "Acc_all_4500m_Speed{speed}.txt",
    'output_filename': "Dis_all_4500m_Speed{speed}.txt",

    # None: results are saved automatically to
    # "current script directory / result_folder_name"
    # Alternatively, specify a full result directory directly,
    # e.g. r"D:\\Results\\PINN"
    'result_base_dir': None,
    'result_folder_name': "Simulation_PINN_Results",

    # Data column indices
    'zc_col': 0,  # Carbody vertical acceleration column
    'beta_col': 1,  # Carbody pitch acceleration column
    'bogie_col': 2,  # Required for La; for Lb it can be set to None or is skipped automatically when the file has too few columns. Existing columns are used for diagnostics by default
    'wheel_col': 6,  # Wheelset displacement column

    # Randomness and run settings
    'seed': 42,
    'num_runs': 5,
    'device': 'cpu',  # Can be changed to 'cuda' if the environment supports it

    # Operating condition and data preprocessing
    'speed': 80,  # km/h
    'highcut': 2,  # Low-pass cutoff frequency, Hz
    'fs': 2000,  # Original sampling frequency, Hz
    'lowpass_order': 4,  # Butterworth low-pass filter order
    'downsample_nyquist_factor': 2.0,  # Downsampling target is approximately highcut * 2
    'seq_len': 20,

    # Data split
    'train_ratio': 0.7,
    'val_ratio': 0.1,
    'test_ratio': 0.2,

    # Model architecture
    'hidden_size': 16,
    'dropout_rate': 0.3,

    # Small constant to avoid division by zero at DC in the FFT
    # frequency-domain integration
    'fft_dc_epsilon': 1e-6,

    # Training and optimization
    'batch_size': 16,
    'epochs': 300,
    'lr': 0.001,
    'weight_decay': 1e-5,
    'l1_lambda': 0.0005,
    'grad_clip_max_norm': 1.0,

    # Learning rate scheduler (StepLR)
    'scheduler_step_size': 50,
    'scheduler_gamma': 0.5,

    # Improvement threshold used only for saving the best validation model;
    # it does not trigger early stopping
    'min_delta': 0.001,

    # Bogie intermediate supervision weights for PINN-La;
    # the following three are unused by PINN-Lb
    'bogie_weight_init': 1.0,
    'bogie_weight_min': 0.00,
    'bogie_decay_rate': 0.9,

    # DataLoader
    'train_drop_last': True,
    'val_drop_last': True,
    'test_drop_last': True,

}




def set_global_seed(seed):
    """Set the global random seed so that the whole set of repeated experiments is reproducible."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Initialize the global random seed
set_global_seed(CONFIG['seed'])


# ============================================================
# 2. Data Preprocessing and Utility Functions
# ============================================================
def butter_lowpass_filter(data, cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std, mean, std


def denormalize_data(normed, mean, std):
    return normed * std + mean


def calculate_metrics(true, pred):
    """Compute MAE, MSE, RMSE, Pearson R and the square of Pearson R."""
    mae = np.mean(np.abs(true - pred))
    mse = np.mean((true - pred) ** 2)
    rmse = np.sqrt(mse)

    if len(true) > 1:  # Ensure enough data points to compute the correlation
        correlation, _ = pearsonr(true, pred)
        r_squared = correlation ** 2  # Coefficient of determination
    else:
        correlation = 0.0
        r_squared = 0.0

    return mae, mse, rmse, correlation, r_squared

# ============================================================
# 3. Dataset
# ============================================================
class BogieDataset(Dataset):
    """Optional bogie labels: returns (x, bogie, wheel) with labels and (x, wheel) without labels."""

    def __init__(self, zc_ddot, beta_ddot, bogie_acc, zw_true, seq_len):
        self.zc_ddot = zc_ddot
        self.beta_ddot = beta_ddot
        self.bogie_acc = bogie_acc
        self.zw_true = zw_true
        self.seq_len = seq_len

    def __len__(self):
        # To reproduce the original simulation experiment, the original
        # sliding-window count definition is kept, so the number of samples
        # is unchanged.
        return len(self.zc_ddot) - self.seq_len

    def __getitem__(self, idx):
        end = idx + self.seq_len
        x_seq = np.stack([self.zc_ddot[idx:end], self.beta_ddot[idx:end]], axis=1)
        y_zw = self.zw_true[end - 1]
        x_tensor = torch.tensor(x_seq, dtype=torch.float32)
        wheel_tensor = torch.tensor(y_zw, dtype=torch.float32)
        if self.bogie_acc is None:
            return x_tensor, wheel_tensor
        bogie_tensor = torch.tensor(self.bogie_acc[idx:end], dtype=torch.float32)
        return x_tensor, bogie_tensor, wheel_tensor


def unpack_batch(batch):
    """Keep the original triplet order when labels exist; use None to mark missing labels and never fabricate labels."""
    if len(batch) == 3:
        x_seq, y_bogie_norm, y_zw_norm = batch
    elif len(batch) == 2:
        x_seq, y_zw_norm = batch
        y_bogie_norm = None
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")
    return x_seq, y_bogie_norm, y_zw_norm


def resolve_bogie_column(acc_data, params):
    """La must have labels; Lb auto-detects missing columns and supports explicit bogie_col=None."""
    variant = params['model_variant']
    col = params['bogie_col']
    if col is None:
        if variant == 'La':
            raise ValueError("PINN-La requires bogie acceleration labels: please set a valid bogie_col and provide that data column.")
        return None
    if not isinstance(col, (int, np.integer)) or col < 0:
        raise ValueError("bogie_col must be a non-negative integer or None.")
    if col >= acc_data.shape[1]:
        if variant == 'La':
            raise ValueError(
                f"PINN-La requires ground-truth bogie acceleration: bogie_col={col}, "
                f"but the acceleration file has only {acc_data.shape[1]} columns."
            )
        print(f"PINN-Lb: the acceleration file does not contain bogie_col={col}; bogie ground truth is not read and diagnostic metrics are skipped.")
        return None
    return col


# ============================================================
# 4. Two-Stage Network Architecture
# ============================================================
# Stage 1: bogie acceleration prediction
class BogieStageCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Two gates
        self.damping_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.stiffness_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # Output layer: predicts bogie acceleration
        self.out = nn.Linear(hidden_size, 1)

        # Learnable c_t parameters
        # forget_factor: controls how fast historical information is forgotten (range 0-1)
        # mix_factor: controls the mixing ratio between the hidden state and the cell state (range 0-1)
        self.forget_factor = nn.Parameter(torch.tensor(0.9))
        self.mix_factor = nn.Parameter(torch.tensor(0.5))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev, c_prev):
        # x: [B, input_size]
        combined = torch.cat([x, h_prev], dim=1)

        damping = self.sigmoid(self.damping_gate(combined))
        stiffness = self.tanh(self.stiffness_gate(combined))
        hidden_t = damping * stiffness

        # Use learnable parameters (sigmoid keeps them in the 0-1 range)
        alpha = torch.sigmoid(self.forget_factor)  # forget factor
        beta = torch.sigmoid(self.mix_factor)  # mixing factor
        # Cell state update: retain historical information + add new information
        c_t = alpha * c_prev + (1 - alpha) * hidden_t
        # Hidden state update: fuse the instantaneous state and long-term memory
        h_t = beta * hidden_t + (1 - beta) * c_t

        bogie_pred = self.out(h_t).squeeze(-1)
        return h_t, c_t, bogie_pred


# Stage 2: wheelset displacement prediction
class WheelStageCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size

        # Two gates
        self.damping_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.stiffness_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # Output layer: predicts wheelset displacement
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.out = nn.Linear(hidden_size, 1)

        # Learnable c_t parameters
        self.forget_factor = nn.Parameter(torch.tensor(0.9))
        self.mix_factor = nn.Parameter(torch.tensor(0.5))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev, c_prev):
        # x: [B, 7]
        combined = torch.cat([x, h_prev], dim=1)

        damping = self.sigmoid(self.damping_gate(combined))
        stiffness = self.tanh(self.stiffness_gate(combined))

        hidden_t = damping * stiffness

        # Version that keeps c
        # Use learnable parameters (sigmoid keeps them in the 0-1 range)
        alpha = torch.sigmoid(self.forget_factor)  # forget factor
        beta = torch.sigmoid(self.mix_factor)  # mixing factor
        # Cell state update
        c_t = alpha * c_prev + (1 - alpha) * hidden_t
        # Hidden state update
        h_t = beta * hidden_t + (1 - beta) * c_t

        y = self.dropout(h_t)
        wheel_pred = self.out(y).squeeze(-1)

        return h_t, c_t, wheel_pred


class TwoStagePhysLSTM(nn.Module):
    def __init__(self, car_feat_dim, hidden_size, dropout_rate, fft_dc_epsilon):
        super().__init__()
        self.hidden_size = hidden_size
        self.fft_dc_epsilon = fft_dc_epsilon

        # Stage 1 input: concatenated carbody a, v, d
        # car_feat_dim=2 -> input dim = 2 * 3 = 6
        self.cell_stage1 = BogieStageCell(
            input_size=car_feat_dim * 3,
            hidden_size=hidden_size
        )

        # Stage 2 input is fixed to 7 dims:
        # [zc_dot, zc, beta_dot, beta, a_b, v_b, d_b]
        self.cell_stage2 = WheelStageCell(
            input_size=7,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate
        )

    def freq_integration(self, sig, delta_t):
        """Perform FFT-based frequency-domain integration along the sequence
        time axis to obtain acceleration, velocity and displacement.

        Args:
            sig: Input signal of shape [B, L, C].
            delta_t: Time interval between adjacent samples, in seconds.

        Returns:
            (acc, vel, disp): Acceleration, velocity and displacement with the
            same shape as the input.

        Notes:
            The zero-frequency components of both velocity and displacement
            are set to zero.
        """
        B, L, C = sig.shape

        x_freq = fft.fft(sig, dim=1)
        omega = 2 * torch.pi * fft.fftfreq(L, d=delta_t).to(sig.device)
        omega[0] = self.fft_dc_epsilon
        omega = omega.view(1, L, 1).expand(B, L, C)

        vel = x_freq / (1j * omega)
        vel[:, 0, :] = 0
        vel = fft.ifft(vel, dim=1).real

        disp = fft.fft(vel, dim=1) / (1j * omega)
        disp[:, 0, :] = 0
        disp = fft.ifft(disp, dim=1).real

        return sig, vel, disp

    def forward(self, car_acc, bogie_true=None, delta_t=1.0):
        B, L, C = car_acc.shape

        # Frequency-domain integration of the carbody acceleration
        a, v, d = self.freq_integration(car_acc, delta_t)
        stage1_input = torch.cat([a, v, d], dim=2)

        # Stage 1: carbody -> bogie
        h1 = torch.zeros(B, self.hidden_size, device=car_acc.device)
        c1 = torch.zeros(B, self.hidden_size, device=car_acc.device)

        bogie_preds = []
        total_bogie_loss = torch.tensor(0., device=car_acc.device)

        for t in range(L):
            x1_t = stage1_input[:, t, :]
            h1, c1, bogie_pred_t = self.cell_stage1(x1_t, h1, c1)
            bogie_preds.append(bogie_pred_t.unsqueeze(1))

            if bogie_true is not None:
                total_bogie_loss += F.mse_loss(bogie_pred_t, bogie_true[:, t])  # supervised at every time step

        bogie_preds_seq = torch.cat(bogie_preds, dim=1)

        if bogie_true is not None:
            total_bogie_loss = total_bogie_loss / L
        else:
            total_bogie_loss = torch.tensor(0., device=car_acc.device)

        # Frequency-domain integration of the bogie acceleration generated in Stage 1
        bogie_acc = bogie_preds_seq.unsqueeze(-1)
        b_acc, b_vel, b_disp = self.freq_integration(bogie_acc, delta_t)

        # Stage 2: bogie/carbody fusion -> wheelset
        h2 = torch.zeros(B, self.hidden_size, device=car_acc.device)
        c2 = torch.zeros(B, self.hidden_size, device=car_acc.device)

        wheel_preds = []

        for t in range(L):
            car_vel_t = v[:, t, :]
            car_disp_t = d[:, t, :]

            bogie_acc_t = b_acc[:, t, 0].unsqueeze(1)
            bogie_vel_t = b_vel[:, t, 0].unsqueeze(1)
            bogie_disp_t = b_disp[:, t, 0].unsqueeze(1)

            # Stage 2 input order:
            # [zc_dot, zc, beta_dot, beta, a_b, v_b, d_b]
            stage2_input_t = torch.cat([
                car_vel_t[:, 0:1],  # carbody bounce velocity zc_dot
                car_disp_t[:, 0:1],  # carbody bounce displacement zc
                car_vel_t[:, 1:2],  # carbody pitch velocity beta_dot
                car_disp_t[:, 1:2],  # carbody pitch displacement beta
                bogie_acc_t,  # bogie acceleration
                bogie_vel_t,  # bogie velocity
                bogie_disp_t  # bogie displacement
            ], dim=1)

            h2, c2, wheel_pred_t = self.cell_stage2(stage2_input_t, h2, c2)
            wheel_preds.append(wheel_pred_t.unsqueeze(1))

        wheel_preds_seq = torch.cat(wheel_preds, dim=1)
        wheel_pred = wheel_preds_seq[:, -1]

        return wheel_pred, total_bogie_loss, bogie_preds_seq, wheel_preds_seq

# ============================================================
# 5. Best Model Prediction and Summary
# ============================================================
def collect_predictions(model, loader, params):
    """Collect wheelset predictions after training; when bogie labels are absent, only the internal predictions in normalized scale are kept."""
    wheel_preds, wheel_targets = [], []
    bogie_preds, bogie_targets = [], []
    has_bogie_labels = None

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_seq, y_bogie_norm, y_zw_norm = unpack_batch(batch)
            x_seq = x_seq.to(params['device'])
            pred_zw_norm, _, bogie_pred_seq, _ = model(
                x_seq, bogie_true=None, delta_t=params['delta_t']
            )
            wheel_preds.extend(pred_zw_norm.cpu().numpy())
            wheel_targets.extend(y_zw_norm.cpu().numpy())
            bogie_preds.extend(bogie_pred_seq[:, -1].cpu().numpy())
            if has_bogie_labels is None:
                has_bogie_labels = y_bogie_norm is not None
            if y_bogie_norm is not None:
                bogie_targets.extend(y_bogie_norm[:, -1].cpu().numpy())

    return (np.asarray(wheel_preds), np.asarray(wheel_targets),
            np.asarray(bogie_preds),
            np.asarray(bogie_targets) if has_bogie_labels else None)


def save_run_summary(summary_file, run_results, params):
    """Save wheelset metrics; diagnostic/supervision metrics are recorded only when ground-truth bogie labels exist."""
    bogie_label = 'Bogie supervision MSE' if params['model_variant'] == 'La' else 'Bogie diagnostic MSE (not used for optimization)'
    metric_items = (
        ('Train MAE (mm)', 'train_mae', 1000.0),
        ('Train RMSE (mm)', 'train_rmse', 1000.0),
        ('Test MAE (mm)', 'test_mae', 1000.0),
        ('Test RMSE (mm)', 'test_rmse', 1000.0),
        ('Run time (s)', 'run_time', 1.0),
    )

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write(f"Run summary (mode: PINN-{params['model_variant']}; "
                f"condition: {params['speed']}km/h, {params['highcut']}Hz)\n")
        f.write('=' * 80 + '\n\n')
        for result in run_results:
            f.write(f"Run {result['run']}:\n")
            f.write(f"  Train wheelset displacement: MAE={result['train_mae'] * 1000:.2f}mm, "
                    f"RMSE={result['train_rmse'] * 1000:.2f}mm, "
                    f"R={result['train_corr']:.4f}, R\u00b2(Pearson)={result['train_r_squared']:.4f}\n")
            f.write(f"  Test wheelset displacement: MAE={result['test_mae'] * 1000:.2f}mm, "
                    f"RMSE={result['test_rmse'] * 1000:.2f}mm, "
                    f"R={result['test_corr']:.4f}, R\u00b2(Pearson)={result['test_r_squared']:.4f}\n")
            if result['bogie_labels_available']:
                f.write(f"  Train bogie acceleration ({'supervision' if params['model_variant'] == 'La' else 'diagnostic only'}): "
                        f"MAE={result['train_bogie_mae']:.4f}, "
                        f"RMSE={result['train_bogie_rmse']:.4f}, "
                        f"R={result['train_bogie_correlation']:.4f}\n")
                f.write(f"  Test bogie acceleration ({'supervision' if params['model_variant'] == 'La' else 'diagnostic only'}): "
                        f"MAE={result['bogie_mae']:.4f}, "
                        f"RMSE={result['bogie_rmse']:.4f}, "
                        f"R={result['bogie_correlation']:.4f}\n")
            else:
                f.write("  Bogie acceleration: no ground-truth labels provided; bogie evaluation metrics are not computed.\n")
            f.write(f"  Best epoch: {result['best_epoch']} | "
                    f"Run time: {result['run_time']:.2f}s\n")
            f.write(f"  Final train/val wheelset loss: "
                    f"{result['final_train_loss']:.6f} / {result['final_val_loss']:.6f}\n")
            if result['bogie_labels_available']:
                f.write(f"  Final train/val {bogie_label}: "
                        f"{result['final_train_bogie_loss']:.6f} / "
                        f"{result['final_val_bogie_loss']:.6f}\n\n")
            else:
                f.write(f"  {bogie_label}: N/A (no ground-truth labels)\n\n")
        f.write('=' * 80 + '\nMean and Standard Deviation\n' + '=' * 80 + '\n')
        f.write(f"{'Metric':<20}{'Mean':<15}{'Std':<15}\n")
        for label, key, scale in metric_items:
            values = np.asarray([result[key] * scale for result in run_results])
            f.write(f'{label:<20}{np.mean(values):<15.4f}{np.std(values):<15.4f}\n')
# ============================================================
# 6. Single Run: Training, Evaluation and Result Saving
# ============================================================
def main_run(run_idx, base_dir, acc_data, dis_data, params):
    """Execute one independent run of the specified PINN-La / PINN-Lb mode.

    Args:
        run_idx: Index of the current independent run.
        base_dir: Root directory for the experiment results.
        acc_data: Raw data read from the acceleration file.
        dis_data: Raw data read from the displacement file.
        params: Configuration parameters of the current experiment.

    Returns:
        dict: Evaluation metrics, training records and related result
        information of this run.
    """
    run_start_time = time.time()
    use_bogie_supervision = params["model_variant"] == "La"

    # Get speed and highcut from the parameters
    speed = params['speed']
    highcut = params['highcut']

    # Create a subfolder named after the hyperparameters
    param_folder = f"{speed}km_h-{highcut}Hz"
    run_dir = os.path.join(base_dir, param_folder, f'run{run_idx}')
    os.makedirs(run_dir, exist_ok=True)

    # Raw channels (all column indices are controlled by CONFIG/params)
    zc = acc_data[:, params['zc_col']]  # vertical acceleration
    beta = acc_data[:, params['beta_col']]  # pitch acceleration
    bogie_col = resolve_bogie_column(acc_data, params)
    has_bogie_labels = bogie_col is not None
    bogie = acc_data[:, bogie_col] if has_bogie_labels else None  # used only for La supervision or Lb diagnostics
    print(f"Ground-truth bogie labels: {'available (La supervision)' if use_bogie_supervision else 'available (Lb diagnostic only)' if has_bogie_labels else 'unavailable (Lb, diagnostics not computed)'}")
    wheel_data = dis_data[:, params['wheel_col']]  # wheelset displacement

    total_len = len(zc)
    train_ratio = params['train_ratio']
    val_ratio = params['val_ratio']
    test_ratio = params['test_ratio']

    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)

    train_zc = zc[:train_end]
    val_zc = zc[train_end:val_end]
    test_zc = zc[val_end:]

    train_beta = beta[:train_end]
    val_beta = beta[train_end:val_end]
    test_beta = beta[val_end:]

    train_bogie = bogie[:train_end] if has_bogie_labels else None
    val_bogie = bogie[train_end:val_end] if has_bogie_labels else None
    test_bogie = bogie[val_end:] if has_bogie_labels else None

    train_wheel = wheel_data[:train_end]
    val_wheel = wheel_data[train_end:val_end]
    test_wheel = wheel_data[val_end:]

    print(f"Data split: train {train_ratio * 100:.1f}% ({train_end} points) | "
          f"val {val_ratio * 100:.1f}% ({val_end - train_end} points) | "
          f"test {test_ratio * 100:.1f}% ({total_len - val_end} points)")

    # Filtering and downsampling
    factor = int(
        params['fs'] /
        (params['highcut'] * params['downsample_nyquist_factor'])
    )

    if factor < 1:
        raise ValueError(
            "Downsampling factor < 1; please check fs, highcut and downsample_nyquist_factor."
        )

    def preprocess_signal(signal):
        filtered = butter_lowpass_filter(
            signal,
            params['highcut'],
            params['fs'],
            params['lowpass_order']
        )
        downsampled = resample_poly(filtered, up=1, down=factor)
        return downsampled

    train_zc_ds = preprocess_signal(train_zc)
    val_zc_ds = preprocess_signal(val_zc)
    test_zc_ds = preprocess_signal(test_zc)

    train_beta_ds = preprocess_signal(train_beta)
    val_beta_ds = preprocess_signal(val_beta)
    test_beta_ds = preprocess_signal(test_beta)

    train_bogie_ds = preprocess_signal(train_bogie) if has_bogie_labels else None
    val_bogie_ds = preprocess_signal(val_bogie) if has_bogie_labels else None
    test_bogie_ds = preprocess_signal(test_bogie) if has_bogie_labels else None

    train_wheel_ds = preprocess_signal(train_wheel)
    val_wheel_ds = preprocess_signal(val_wheel)
    test_wheel_ds = preprocess_signal(test_wheel)

    # Normalization: use training-set statistics only
    train_zc_norm, zc_mean, zc_std = normalize_data(train_zc_ds)
    train_beta_norm, beta_mean, beta_std = normalize_data(train_beta_ds)
    if has_bogie_labels:
        train_bogie_norm, bogie_mean, bogie_std = normalize_data(train_bogie_ds)
    else:
        train_bogie_norm, bogie_mean, bogie_std = None, None, None
    train_wheel_norm, wheel_mean, wheel_std = normalize_data(train_wheel_ds)

    # Validation and test sets use the training-set normalization parameters
    val_zc_norm = (val_zc_ds - zc_mean) / zc_std
    val_beta_norm = (val_beta_ds - beta_mean) / beta_std
    val_bogie_norm = (val_bogie_ds - bogie_mean) / bogie_std if has_bogie_labels else None
    val_wheel_norm = (val_wheel_ds - wheel_mean) / wheel_std

    test_zc_norm = (test_zc_ds - zc_mean) / zc_std
    test_beta_norm = (test_beta_ds - beta_mean) / beta_std
    test_bogie_norm = (test_bogie_ds - bogie_mean) / bogie_std if has_bogie_labels else None
    test_wheel_norm = (test_wheel_ds - wheel_mean) / wheel_std

    # Create datasets
    train_ds = BogieDataset(
        zc_ddot=train_zc_norm,
        beta_ddot=train_beta_norm,
        bogie_acc=train_bogie_norm,
        zw_true=train_wheel_norm,
        seq_len=params['seq_len']
    )

    val_ds = BogieDataset(
        zc_ddot=val_zc_norm,
        beta_ddot=val_beta_norm,
        bogie_acc=val_bogie_norm,
        zw_true=val_wheel_norm,
        seq_len=params['seq_len']
    )

    test_ds = BogieDataset(
        zc_ddot=test_zc_norm,
        beta_ddot=test_beta_norm,
        bogie_acc=test_bogie_norm,
        zw_true=test_wheel_norm,
        seq_len=params['seq_len']
    )

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=params['train_drop_last']
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=params['batch_size'],
        shuffle=False,
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params['batch_size'],
        shuffle=False,
        drop_last=params['val_drop_last']
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=params['batch_size'],
        shuffle=False,
        drop_last=params['test_drop_last']
    )

    # When the amount of data is too small and drop_last=True, avoid
    # confusing division-by-zero errors caused by empty loaders.
    for split_name, loader in (("training", train_loader), ("validation", val_loader), ("test", test_loader)):
        if len(loader) == 0:
            raise ValueError(
                f"The {split_name} set is too small to form a single batch after downsampling and sliding-window slicing."
                "Please check the data length, seq_len, batch_size and the corresponding drop_last setting."
            )

    # Model initialization
    model = TwoStagePhysLSTM(
        car_feat_dim=2,
        hidden_size=params['hidden_size'],
        dropout_rate=params['dropout_rate'],
        fft_dc_epsilon=params['fft_dc_epsilon']
    ).to(params['device'])

    # Optimizer and learning rate scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=params['scheduler_step_size'],
        gamma=params['scheduler_gamma']
    )

    # Training records
    train_losses = []  # wheelset displacement training loss
    val_losses = []  # wheelset displacement validation loss
    train_bogie_mse_history = []  # La: supervision MSE; Lb: diagnostic only; NaN when labels are missing
    val_bogie_mse_history = []  # not used for best-model selection in either mode

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1
    # La uses a decaying intermediate supervision weight; Lb keeps the same
    # architecture but disables intermediate supervision.
    for epoch in range(1, params['epochs'] + 1):
        if use_bogie_supervision:
            current_bogie_weight = max(
                params['bogie_weight_min'],
                params['bogie_weight_init'] * (params['bogie_decay_rate'] ** (epoch - 1))
            )

        model.train()
        run_zw_loss = 0.0
        run_bogie_loss = 0.0

        for batch in train_loader:
            x_seq, y_bogie_norm, y_zw_norm = unpack_batch(batch)
            x_seq = x_seq.to(params['device'])
            if y_bogie_norm is not None:
                y_bogie_norm = y_bogie_norm.to(params['device'])
            y_zw_norm = y_zw_norm.to(params['device'])

            optimizer.zero_grad()

            # La: the bogie supervision loss is included in optimization;
            # Lb: bogie labels are excluded from optimization and used only
            # for diagnostic evaluation.
            pred_zw_norm, bogie_loss, bogie_pred_seq, _ = model(
                x_seq,
                bogie_true=y_bogie_norm if use_bogie_supervision else None,
                delta_t=params['delta_t']
            )
            if not use_bogie_supervision and has_bogie_labels:
                with torch.no_grad():
                    bogie_loss = F.mse_loss(bogie_pred_seq.detach(), y_bogie_norm)  # diagnostic only
            # When label Lb is absent, bogie_loss is a zero placeholder returned by the model - not computed, let alone optimized.

            zw_loss = F.mse_loss(pred_zw_norm, y_zw_norm)

            l1_loss = torch.tensor(0.0, device=params['device'])
            for param in model.parameters():
                l1_loss += torch.norm(param, p=1)

            total_loss = zw_loss + params['l1_lambda'] * l1_loss
            if use_bogie_supervision:
                total_loss = total_loss + current_bogie_weight * bogie_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['grad_clip_max_norm'])
            optimizer.step()

            run_zw_loss += zw_loss.item()
            if has_bogie_labels:
                run_bogie_loss += bogie_loss.item()

        # Compute average losses
        train_zw_loss = run_zw_loss / len(train_loader)
        train_bogie_loss_avg = run_bogie_loss / len(train_loader) if has_bogie_labels else float('nan')

        train_losses.append(train_zw_loss)
        train_bogie_mse_history.append(train_bogie_loss_avg)

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_bogie_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x_seq, y_bogie_norm, y_zw_norm = unpack_batch(batch)
                x_seq = x_seq.to(params['device'])
                if y_bogie_norm is not None:
                    y_bogie_norm = y_bogie_norm.to(params['device'])
                y_zw_norm = y_zw_norm.to(params['device'])

                pred_zw_norm, bogie_loss, bogie_pred_seq, _ = model(
                    x_seq,
                    bogie_true=y_bogie_norm if use_bogie_supervision else None,
                    delta_t=params['delta_t']
                )
                if not use_bogie_supervision and has_bogie_labels:
                    bogie_loss = F.mse_loss(bogie_pred_seq, y_bogie_norm)  # diagnostic only

                zw_loss = F.mse_loss(pred_zw_norm, y_zw_norm)
                val_loss += zw_loss.item()
                if has_bogie_labels:
                    val_bogie_loss += bogie_loss.item()

        val_loss_avg = val_loss / len(val_loader)
        val_bogie_loss_avg = val_bogie_loss / len(val_loader) if has_bogie_labels else float('nan')

        val_losses.append(val_loss_avg)
        val_bogie_mse_history.append(val_bogie_loss_avg)

        if epoch % 10 == 0:
            bogie_log = (
                f" | Train bogie MSE ({'supervision' if use_bogie_supervision else 'diagnostic'}): "
                f"{train_bogie_loss_avg:.6f} | Val bogie MSE: {val_bogie_loss_avg:.6f}"
                if has_bogie_labels else " | Bogie metric: N/A (no labels)"
            )
            print(
                f"{params['model_variant']} | Run {run_idx} | Epoch {epoch}/{params['epochs']} | "
                f"Train wheel: {train_zw_loss:.6f} | "
                f"Val wheel: {val_loss_avg:.6f}{bogie_log}",
                flush=True
            )

        # Update the learning rate
        scheduler.step()

        # Save the best model by validation loss improvement
        # (does not affect the fixed number of epochs)
        if val_loss_avg < (best_val_loss - params['min_delta']):
            best_val_loss = val_loss_avg
            best_epoch = epoch

            best_model_state = copy.deepcopy(model.state_dict())

    # Use the model with the best validation wheelset loss and evaluate once
    # uniformly after training is complete.
    model.load_state_dict(best_model_state)
    best_train_preds, best_train_targets, best_train_bogie_preds, best_train_bogie_targets = collect_predictions(
        model, train_eval_loader, params
    )
    best_test_preds, best_test_targets, best_test_bogie_preds, best_test_bogie_targets = collect_predictions(
        model, test_loader, params
    )
    run_time = time.time() - run_start_time

    # Save the best model
    best_model_path = os.path.join(run_dir, 'best_model.pth')

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_variant': params['model_variant'],
        'bogie_supervision': use_bogie_supervision,
        'bogie_labels_available': has_bogie_labels,

        # Current operating condition
        'speed': params['speed'],
        'highcut': params['highcut'],
        'delta_t': params['delta_t'],
        'seq_len': params['seq_len'],
        'hidden_size': params['hidden_size'],

        # Best epoch
        'best_epoch': best_epoch,

        # Normalization parameters, used later for sensitivity analysis
        'zc_mean': zc_mean,
        'zc_std': zc_std,
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'bogie_mean': bogie_mean,  # None when labels are missing; de-normalization to physical units is not possible
        'bogie_std': bogie_std,
        'wheel_mean': wheel_mean,
        'wheel_std': wheel_std,

    }, best_model_path)

    print(f"Best model saved: {best_model_path}")

    # De-normalize the predictions
    train_preds_denorm = denormalize_data(best_train_preds, wheel_mean, wheel_std)
    train_true_denorm = denormalize_data(best_train_targets, wheel_mean, wheel_std)
    test_preds_denorm = denormalize_data(best_test_preds, wheel_mean, wheel_std)
    test_true_denorm = denormalize_data(best_test_targets, wheel_mean, wheel_std)

    # Compute metrics
    train_mae, train_mse, train_rmse, train_corr, train_r2 = calculate_metrics(
        train_true_denorm,
        train_preds_denorm
    )

    test_mae, test_mse, test_rmse, test_corr, test_r2 = calculate_metrics(
        test_true_denorm,
        test_preds_denorm
    )

    # De-normalize to physical units and evaluate only when ground-truth
    # bogie labels exist.
    # In the label-free mode, bogie normalization statistics must not be
    # fabricated, nor metrics based on ground truth computed.
    bogie_metrics = {
        'bogie_mae': None, 'bogie_rmse': None,
        'bogie_correlation': None, 'bogie_r_squared': None,
        'train_bogie_mae': None, 'train_bogie_rmse': None,
        'train_bogie_correlation': None, 'train_bogie_r_squared': None,
    }
    if has_bogie_labels:
        test_bogie_preds_denorm = denormalize_data(best_test_bogie_preds, bogie_mean, bogie_std)
        test_bogie_targets_denorm = denormalize_data(best_test_bogie_targets, bogie_mean, bogie_std)
        train_bogie_preds_denorm = denormalize_data(best_train_bogie_preds, bogie_mean, bogie_std)
        train_bogie_targets_denorm = denormalize_data(best_train_bogie_targets, bogie_mean, bogie_std)
        bogie_mae, _, bogie_rmse, bogie_corr, bogie_r2 = calculate_metrics(
            test_bogie_targets_denorm, test_bogie_preds_denorm
        )
        train_bogie_mae, _, train_bogie_rmse, train_bogie_corr, train_bogie_r2 = calculate_metrics(
            train_bogie_targets_denorm, train_bogie_preds_denorm
        )
        bogie_metrics.update({
            'bogie_mae': bogie_mae, 'bogie_rmse': bogie_rmse,
            'bogie_correlation': bogie_corr, 'bogie_r_squared': bogie_r2,
            'train_bogie_mae': train_bogie_mae, 'train_bogie_rmse': train_bogie_rmse,
            'train_bogie_correlation': train_bogie_corr, 'train_bogie_r_squared': train_bogie_r2,
        })

    print(f"Run {run_idx} - final metrics:")
    if has_bogie_labels:
        label = "supervision term" if use_bogie_supervision else "diagnostic only, excluded from optimization"
        print(f"  Test bogie acceleration ({label}): MAE={bogie_mae:.6f}, RMSE={bogie_rmse:.6f}, R={bogie_corr:.4f}, R\u00b2(Pearson)={bogie_r2:.4f}")
        print(f"  Train bogie acceleration ({label}): MAE={train_bogie_mae:.6f}, RMSE={train_bogie_rmse:.6f}, R={train_bogie_corr:.4f}, R\u00b2(Pearson)={train_bogie_r2:.4f}")
    else:
        print("  Bogie evaluation: no ground-truth labels provided; MAE/RMSE/R metrics are skipped. Internal predictions can still be saved (normalized scale).")
    print(f"  Test wheelset displacement: MAE={test_mae:.6f}, RMSE={test_rmse:.6f}, R={test_corr:.4f}, R\u00b2={test_r2:.4f}")
    print(f"  Train wheelset displacement: MAE={train_mae:.6f}, RMSE={train_rmse:.6f}, R={train_corr:.4f}, R\u00b2={train_r2:.4f}")
    print(f"  Final training loss: {train_losses[-1]:.6f}")
    print(f"  Final validation loss: {val_losses[-1]:.6f}")
    bogie_metric_label = "Bogie supervision MSE" if use_bogie_supervision else "Bogie diagnostic MSE (not used for training)"
    if has_bogie_labels:
        print(f"  Final training {bogie_metric_label}: {train_bogie_mse_history[-1]:.6f}")
        print(f"  Final validation {bogie_metric_label}: {val_bogie_mse_history[-1]:.6f}")
    else:
        print(f"  {bogie_metric_label}: N/A (no ground-truth labels)")
    print(f"  Epoch of best validation loss: {best_epoch}")

    # Save MAT numerical results (no plotting library required)
    train_preds_row = train_preds_denorm.reshape(1, -1)
    train_true_row = train_true_denorm.reshape(1, -1)
    test_preds_row = test_preds_denorm.reshape(1, -1)
    test_true_row = test_true_denorm.reshape(1, -1)

    train_mat_path = os.path.join(run_dir, 'train_results.mat')
    train_mat_data = {
        'train_true': train_true_row,
        'train_pred': train_preds_row,
        'train_loss': np.array(train_losses),  # wheel loss, consistent with the semantics of the original code
        'val_loss': np.array(val_losses),
        'train_bogie_mse': np.array(train_bogie_mse_history),
        'val_bogie_mse': np.array(val_bogie_mse_history),
        'bogie_labels_available': np.array(int(has_bogie_labels)),
    }
    if not has_bogie_labels:
        # Cannot be de-normalized to physical units; only the intermediate
        # model outputs in normalized scale are saved.
        train_mat_data['bogie_pred_normalized'] = best_train_bogie_preds.reshape(1, -1)
    sio.savemat(train_mat_path, train_mat_data)

    test_mat_path = os.path.join(run_dir, 'test_results.mat')
    test_mat_data = {
        'test_true': test_true_row,
        'test_pred': test_preds_row,
        'train_loss': np.array(train_losses),
        'val_loss': np.array(val_losses),
        'train_bogie_mse': np.array(train_bogie_mse_history),
        'val_bogie_mse': np.array(val_bogie_mse_history),
        'bogie_labels_available': np.array(int(has_bogie_labels)),
        'best_epoch': best_epoch,
        'delta_t': params['delta_t'],
    }
    if has_bogie_labels:
        test_mat_data.update({
            'bogie_true': test_bogie_targets_denorm.reshape(1, -1),
            'bogie_pred': test_bogie_preds_denorm.reshape(1, -1),
        })
    else:
        test_mat_data['bogie_pred_normalized'] = best_test_bogie_preds.reshape(1, -1)
    sio.savemat(test_mat_path, test_mat_data)

    return {
        'run': run_idx,
        'train_mae': train_mae,
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_corr': train_corr,
        'train_r_squared': train_r2,

        'test_mae': test_mae,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_corr': test_corr,
        'test_r_squared': test_r2,

        'bogie_labels_available': has_bogie_labels,
        **bogie_metrics,

        'run_time': run_time,

        'final_train_loss': train_losses[-1] if train_losses else 0,
        'final_val_loss': val_losses[-1] if val_losses else 0,
        'final_train_bogie_loss': train_bogie_mse_history[-1] if has_bogie_labels else None,
        'final_val_bogie_loss': val_bogie_mse_history[-1] if has_bogie_labels else None,

        'best_epoch': best_epoch,
    }


# ============================================================
# 7. Main Program Entry
# ============================================================
if __name__ == "__main__":
    params = CONFIG.copy()
    if params['model_variant'] not in ('La', 'Lb'):
        raise ValueError("CONFIG['model_variant'] must be either 'La' or 'Lb'.")

    # Check the data split ratios
    ratio_sum = (
            params['train_ratio'] +
            params['val_ratio'] +
            params['test_ratio']
    )
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1, currently {ratio_sum}"
        )

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Result root directory:
    # result_base_dir=None -> current script directory / result_folder_name
    # result_base_dir=full path -> use that full path directly
    if params['result_base_dir'] is None:
        base_save_dir = os.path.join(
            current_dir,
            params['result_folder_name']
        )
    else:
        base_save_dir = params['result_base_dir']
    base_save_dir = os.path.join(base_save_dir, f"PINN-{params['model_variant']}")
    os.makedirs(base_save_dir, exist_ok=True)

    print(f"Current training mode: PINN-{params['model_variant']} | "
          f"bogie supervision: {'enabled (ground truth required)' if params['model_variant'] == 'La' else 'disabled (ground truth optional, diagnostic only when available)'}")

    # All data file paths come from CONFIG
    input_file = os.path.join(
        params['data_dir'],
        params['input_filename'].format(speed=params['speed'])
    )

    output_file = os.path.join(
        params['data_dir'],
        params['output_filename'].format(speed=params['speed'])
    )

    print(f"Input acceleration file: {input_file}")
    print(f"Input displacement file: {output_file}")
    print(f"Results will be saved to: {base_save_dir}")

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Acceleration input file not found: {input_file}")
    if not os.path.isfile(output_file):
        raise FileNotFoundError(f"Displacement input file not found: {output_file}")

    # Load data
    acc_data = np.loadtxt(input_file)
    dis_data = np.loadtxt(output_file)
    if acc_data.ndim != 2 or dis_data.ndim != 2:
        raise ValueError("Both the acceleration and displacement files must be 2-D tables, one sample per row.")
    for key, data in (('zc_col', acc_data), ('beta_col', acc_data), ('wheel_col', dis_data)):
        col = params[key]
        if not isinstance(col, (int, np.integer)) or col < 0 or col >= data.shape[1]:
            raise ValueError(f"{key}={col} is invalid; the corresponding data file has only {data.shape[1]} columns.")
    if len(acc_data) != len(dis_data):
        raise ValueError("The acceleration and displacement files have different numbers of samples and cannot be aligned in time.")
    # Validate the La label requirement up front; missing columns in Lb must not raise an error.
    resolve_bogie_column(acc_data, params)

    # Compute derived parameters: the actual sampling frequency after
    # downsampling and delta_t
    factor = int(
        params['fs'] /
        (params['highcut'] * params['downsample_nyquist_factor'])
    )
    if factor < 1:
        raise ValueError(
            "Downsampling factor < 1; please check fs, highcut and downsample_nyquist_factor."
        )
    fs_new = params['fs'] / factor
    params['downsample_factor'] = factor
    params['fs_new'] = fs_new
    params['delta_t'] = 1.0 / fs_new

    # Create a subfolder based on the hyperparameters
    param_folder = f"{params['speed']}km_h-{params['highcut']}Hz"
    summary_dir = os.path.join(base_save_dir, param_folder, "runs_summary")
    os.makedirs(summary_dir, exist_ok=True)

    # Save the parameters to a file
    params_file = os.path.join(summary_dir, "parameters.txt")
    with open(params_file, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("Parameters\n")
        f.write("=" * 50 + "\n\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"Parameters saved to: {params_file}")

    # Run experiments
    run_results = []
    for run_idx in range(1, params['num_runs'] + 1):
        print(f"Starting run {run_idx}...")
        result = main_run(run_idx, base_save_dir, acc_data, dis_data, params)
        run_results.append(result)
        print(f"Finished run {run_idx} | time: {result['run_time']:.2f}s")
        print(
            f"Train set: MAE={result['train_mae'] * 1000:.2f}mm, RMSE={result['train_rmse'] * 1000:.2f}mm")
        print(
            f"Test set: MAE={result['test_mae'] * 1000:.2f}mm, RMSE={result['test_rmse'] * 1000:.2f}mm")
        if result['bogie_labels_available']:
            tag = 'supervision' if params['model_variant'] == 'La' else 'diagnostic only'
            print(f"Train bogie acceleration ({tag}): MAE={result['train_bogie_mae']:.4f}, RMSE={result['train_bogie_rmse']:.4f}")
            print(f"Test bogie acceleration ({tag}): MAE={result['bogie_mae']:.4f}, RMSE={result['bogie_rmse']:.4f}")
        else:
            print("Bogie acceleration: no ground-truth labels provided; evaluation skipped.")
        print("-" * 80)

    # Write the metrics and statistics of all runs into a single summary file.
    summary_file = os.path.join(summary_dir, "run_summary.txt")
    save_run_summary(summary_file, run_results, params)
    print(f"Summary saved to: {summary_file}")

    print("\nAll runs completed!")
    print(f"All results saved to: {base_save_dir}")
