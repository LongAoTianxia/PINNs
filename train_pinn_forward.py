"""
PINNs for solving the wave equation (forward problem).
Convert from notebook 5_4_PINNs_forward.ipynb

The goal is to compute the wave field p(x,y,t) that satisfies the wave equation.
All figures are saved to the figures/ directory.
"""

import os
# Fix OpenMP duplicate lib issue - must be set before importing numpy/torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import numpy as np
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.optim.lr_scheduler as lr_scheduler

import optuna
_HAS_OPTUNA = True

from PINNs_util.PINNs_fdiff import solver
from PINNs_util.PINNs_aux import (
    xyt_tensor, pde_residual, update_lambda, 
    rand_colloc, rand_colloc_mixed,rand_colloc_fixed, 
    rand_boundary, bc_residual_absorbing
)
from PINNs_util.PINNs_aux import FCN, FCN_with_Attention, FCN_Attention, LSTMAttention

# ============================================================================
# Configuration
# ============================================================================
class Config:
    """Configuration class for PINN training"""
    # Directories
    FIGURES_DIR = "./figures"
    MODEL_DIR = "./trained/forward/"
    
    # Domain parameters
    L = 5           # Domain size (km)
    T = 1.3         # Time duration (s)
    c0 = 3          # Max wave speed (km/s)
    
    # Grid parameters
    Nx = 50
    Ny = 50
    
    # Network architecture
    n_in = 3
    n_out = 1
    n_hidden = 64  # n_hidden % n_tokens == 0 且 (n_hidden / n_tokens) % num_heads == 0
    n_layers = 2
    n_ffeatures = 64

    n_tokens = 4
    num_heads = 4
    fusion_mode = 'self_attn'  # 'self_attn' or 'cross_attn'
    seq_mode = 'spatial_temporal' # 'dim' or 'spatial_temporal'
    use_attention = True  


    # Training parameters
    n_ini = 15              # Number of initial condition snapshots
    n_lamb_update = 70     # Lambda update frequency
    n_colloc = int(1e4)     # Number of collocation points
    n_causal = int(1.5e3)     # Causal training steps (set to int(2e3) for full training)
    learning_rate = 1e-4
    
    ratio_gaussian = 0.85    # for rand_colloc_fixed()

    # Boundary condition parameters
    n_bc = int(1e3)         # Number of boundary condition points per epoch
    bc_type = 'absorbing'   # Boundary condition type: 'absorbing', 'none' (可扩展其他类型)
    
    # Stage schedule (progress in [0,1])
    STAGE_A_END = 0.15
    STAGE_B_END = 0.70
 
    # BO (Bayesian Optimization) settings
    USE_STAGE_WEIGHTS = True
    RUN_BO = True                 # if False: use DEFAULT_HP directly
    BO_N_TRIALS = 25
    BO_PROXY_EPOCHS = 8000        # short-run budget per trial

    # Default stage HP (used when RUN_BO=False or as initial sanity run)
    DEFAULT_HP = {
        "A_end": STAGE_A_END,
        "B_end": STAGE_B_END,
        "w_pde_A": 1e-2,
        "w_pde_B": 5e-2,
        "w_pde_C": 5e-2,
        "w_ini_B": 1.0,
        "w_ini_C": 0.7,
        "w_bc_C_max": 5e-3,
        "bc_ramp_start": 0.75,
        "bc_ramp_len": 0.15,
    }
    
    # Flags
    TRAIN_NEW_MODEL = True  # Set to True to train a new model

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================================
# Physics Functions
# ============================================================================
def get_wave_speed(xx, yy, device, tensor=False):
    """
    Wave speed function c(x, y).
    Three-layer model with different wave speeds.
    """
    if tensor:
        c_val = 0.5 * torch.ones_like(xx, device=device)
    else:
        c_val = 0.5 * np.ones_like(xx)
    
    ind = yy >= 0.33
    c_val[ind] = 0.75
    ind = yy >= 0.66
    c_val[ind] = 1
    return c_val


def get_initial_condition(xx, yy):
    """
    Initial condition I(x, y).
    Gaussian pulse centered at (0.5, 0.5).
    """
    gpulse_std = 5e-2
    r_pulse = np.array([0.5, 0.5])
    I_val = np.exp(-0.5 * (((xx - r_pulse[0]) / gpulse_std)**2 +
                           ((yy - r_pulse[1]) / gpulse_std)**2))
    return I_val


# ============================================================================
# Plotting Functions
# ============================================================================
def save_wave_speed(c_field, L, save_path):
    """Save wave speed plot"""
    c_max = np.max(np.abs(c_field))
    c_min = np.min(np.abs(c_field))
    fig, ax = plt.subplots(figsize=(4, 3))
    img = ax.imshow(c_field, vmin=c_min, vmax=c_max, origin='lower', 
                    cmap='viridis', extent=[0, L, 0, L])
    fig.colorbar(img, ax=ax)
    ax.set_title('wave speed c(x,y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def save_field_animation(p, L, title, save_path):
    """Save wave field animation as GIF"""
    p_max = np.max(np.abs(p))
    fig, [ax1, ax2] = plt.subplots(1, 2, gridspec_kw={"width_ratios": [50, 1]}, figsize=(4, 3))
    cmap = matplotlib.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-p_max, vmax=p_max)
    matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, orientation='vertical')
    ax1.set_title(title)
    
    frames = []
    for i in range(p.shape[-1]):
        p_plot = p[:, :, i]
        img = ax1.imshow(p_plot, vmin=-p_max, vmax=p_max, origin='lower', 
                         cmap='seismic', extent=[0, L, 0, L], animated=True)
        frames.append([img])
    
    ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
    ani.save(save_path, writer='pillow', fps=5)
    plt.close()
    print(f"Saved: {save_path}")


def save_estimation_animation(p_ref, p_est, L, save_path):
    """Save reference vs estimation animation as GIF"""
    p_max = np.max(np.abs(p_ref))
    n_L = p_ref.shape[0]
    n_T = p_ref.shape[-1]
    p_est = p_est.reshape(n_L, n_L, n_T)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    ax1.set_title('Reference')
    ax2.set_title('Estimated')
    
    frames = []
    for i in range(n_T):
        img1 = ax1.imshow(p_ref[:, :, i], vmin=-p_max, vmax=p_max, origin='lower', 
                          cmap='seismic', extent=[-L/2, L/2, -L/2, L/2], animated=True)
        img2 = ax2.imshow(p_est[:, :, i], vmin=-p_max, vmax=p_max, origin='lower', 
                          cmap='seismic', extent=[-L/2, L/2, -L/2, L/2], animated=True)
        frames.append([img1, img2])
    
    ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
    ani.save(save_path, writer='pillow', fps=5)
    plt.close()
    print(f"Saved: {save_path}")


def save_train_log(loss, lamb, label, save_path):
    """Save loss and lambda training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.set_title('Loss')
    for i in range(len(loss) - 1, -1, -1):
        ax1.plot(np.asarray(loss[i]) * np.asarray(lamb[i]) / np.asarray(lamb[0]), label=label[i])
    ax1.legend()
    ax1.set_yscale("log")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Weighted Loss")
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Lambda')
    for i in range(len(lamb) - 1, -1, -1):
        ax2.plot(lamb[i], label=label[i])
    ax2.legend()
    ax2.set_yscale("log")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Lambda")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def save_comparison(p_ref, p_est, L, T, t, save_path):
    """Save comparison plot at different time steps"""
    n_L = p_ref.shape[0]
    n_T = p_ref.shape[-1]
    p_est = p_est.reshape(n_L, n_L, n_T)
    p_ref_max = np.max(p_ref)
    p_ref_min = -p_ref_max

    # 计算误差
    p_error = p_est - p_ref
    error_max = np.max(np.abs(p_error))

    time_indices = [0, int(n_T * 0.33), int(n_T * 0.66), n_T - 1]
    time_labels = [t[idx] * 5 / 3 for idx in time_indices]

    # 修改为 3 行：Reference, PINN, Error
    fig, axes = plt.subplots(3, 6, figsize=(22, 15), 
                             gridspec_kw={'width_ratios': [0.2, 1, 1, 1, 1, 0.2]})

    axes[0, 0].text(0.5, 0.5, 'Reference', fontsize=24, ha='center', va='center', rotation=0)
    axes[1, 0].text(0.5, 0.5, 'PINN', fontsize=24, ha='center', va='center', rotation=0)
    axes[2, 0].text(0.5, 0.5, 'Error', fontsize=24, ha='center', va='center', rotation=0)

    for ax in axes[:, 0]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[:, -1]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # 第一行：Reference
    for i, idx in enumerate(time_indices):
        ax = axes[0, i + 1]
        im_ref = ax.imshow(p_ref[:, :, idx], extent=[0, 5, 0, 5], origin='lower', 
                       cmap='seismic', vmin=p_ref_min, vmax=p_ref_max)
        ax.set_title(f"t={time_labels[i]:.2f} s", fontsize=22)
        ax.set_xlabel('x (km)', fontsize=20)
        ax.set_ylabel('y (km)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)

    # 第二行：PINN
    for i, idx in enumerate(time_indices):
        ax = axes[1, i + 1]
        im_est = ax.imshow(p_est[:, :, idx], extent=[0, 5, 0, 5], origin='lower', 
                       cmap='seismic', vmin=p_ref_min, vmax=p_ref_max)
        # ax.set_xlabel('x (km)', fontsize=20)
        ax.set_ylabel('y (km)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)

    # 第三行：Error，并在 title 上标注平均误差
    for i, idx in enumerate(time_indices):
        ax = axes[2, i + 1]
        error_slice = p_error[:, :, idx]
        mean_error = np.mean(np.abs(error_slice))  # 平均绝对误差
        im_err = ax.imshow(error_slice, extent=[0, 5, 0, 5], origin='lower', 
                       cmap='seismic', vmin=-error_max, vmax=error_max)
        ax.set_title(f"MAE={mean_error:.4f}", fontsize=22)
        ax.set_xlabel('x (km)', fontsize=20)
        ax.set_ylabel('y (km)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)

    # 添加两个 colorbar：一个用于 Reference/PINN，一个用于 Error
    cbar_ax1 = fig.add_axes([0.94, 0.4, 0.02, 0.5])
    cbar1 = fig.colorbar(im_est, cax=cbar_ax1)
    cbar1.set_label('Normalized pressure', fontsize=24)
    cbar1.ax.tick_params(labelsize=20)

    cbar_ax2 = fig.add_axes([0.94, 0.08, 0.02, 0.25])
    cbar2 = fig.colorbar(im_err, cax=cbar_ax2)
    cbar2.set_label('Error', fontsize=24)
    cbar2.ax.tick_params(labelsize=20)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Data Preparation
# ============================================================================
def prepare_domain(config, device):
    """Prepare the computational domain and reference solution"""
    # Scaling
    L = config.L / config.L  # Normalized to 1
    T = config.T * config.c0 / config.L
    
    # Create wave speed function with device closure
    def c_func(xx, yy, tensor=False):
        return get_wave_speed(xx, yy, device, tensor)
    
    # Solve reference solution
    print("Solving reference solution using finite difference...")
    p_ref, xx, yy, t, dt = solver(
        get_initial_condition, c_func, L, L, config.Nx, config.Ny, -1, T
    )
    
    # Prepare coordinate tensors
    xy = np.column_stack((np.reshape(xx, (-1, 1)), np.reshape(yy, (-1, 1))))
    r_ref = xyt_tensor(xy, t, device)
    # print("r_ref shape:", r_ref.shape)
    n_T = t.shape[0]
    n_L = xx.shape[0]
    
    return {
        'L': L, 'T': T,
        'p_ref': p_ref, 'xx': xx, 'yy': yy, 't': t,
        'xy': xy, 'r_ref': r_ref, 'n_T': n_T, 'n_L': n_L,
        'c_func': c_func
    }


def prepare_initial_condition(data, config, device):
    """Prepare initial condition data for training"""
    t_ini = data['t'][0:config.n_ini]
    r_ini = xyt_tensor(data['xy'], t_ini, device)
    p_ini = data['p_ref'][:, :, :config.n_ini].reshape(-1, 1)
    p_ini = torch.tensor(p_ini, device=device)
    
    return r_ini, p_ini


# ============================================================================
# Model Training
# ============================================================================
def get_stage_weights(progress, hp):
    # progress ∈ [0,1]
    if progress < hp["A_end"]:
        return (
            1.0,                     # w_ini
            hp["w_pde_A"],            # w_pde
            0.0                       # w_bc
        )
    elif progress < hp["B_end"]:
        return (
            hp["w_ini_B"],
            hp["w_pde_B"],
            0.0
        )
    else:
        r = (progress - hp["bc_ramp_start"]) / max(hp["bc_ramp_len"], 1e-6)
        r = min(max(r, 0.0), 1.0)
        return (
            hp["w_ini_C"],
            hp["w_pde_C"],
            hp["w_bc_C_max"] * r
        )

def _random_hp(rng: np.random.RandomState, config: Config):
    # log-uniform helpers
    def logu(a, b):
        return float(np.exp(rng.uniform(np.log(a), np.log(b))))
    hp = {
        "A_end": config.STAGE_A_END,
        "B_end": config.STAGE_B_END,
        "w_pde_A": logu(1e-3, 1e-1),
        "w_pde_B": logu(1e-2, 1.0),
        "w_pde_C": logu(1e-2, 1.0),
        "w_ini_B": float(rng.uniform(0.5, 2.0)),
        "w_ini_C": float(rng.uniform(0.2, 1.5)),
        "w_bc_C_max": logu(1e-4, 1e-1),
        "bc_ramp_start": float(rng.uniform(0.7, 0.9)),
        "bc_ramp_len": float(rng.uniform(0.05, 0.25)),
    }
    return hp


def create_model(config, device):
    """Create and initialize the PINN model"""
    #model = FCN(config.n_in, config.n_out, config.n_ffeatures, config.n_hidden, config.n_layers)
    #model = FCN_Attention(n_out=config.n_out, n_ffeatures=32, n_hidden=32, n_layers=2, num_heads=2, embed_dim=64, fusion_mode='cross_attn')
    #model = FCN_with_Attention(
    #    # n_hidden % n_tokens == 0 且 (n_hidden / n_tokens) % num_heads == 0
    #    n_in=config.n_in, n_out=config.n_out, n_ffeatures=config.n_ffeatures, n_hidden=config.n_hidden,   
    #    n_layers=config.n_layers, num_heads=config.num_heads, n_tokens=config.n_tokens,       # 拆分为4个token，每个16维
    #    mlp_ratio=4., use_attention=config.use_attention
    #)
    model = LSTMAttention(input_dim=config.n_in, output_dim=config.n_out, hidden_dim=config.n_hidden, num_layers=config.n_layers, dropout=0.0, 
                          use_attention=config.use_attention, seq_mode=config.seq_mode, n_ffeatures=config.n_ffeatures)
    model = model.to(device)
    return model


def train_model(model, data, r_ini, p_ini, config, device, use_adaptive_lambda=False, hp=None, max_epochs=None):
    """Train the PINN model"""
    #n_epochs = int(config.n_causal * (data['n_T'] - 1))
    n_epochs_full = int(config.n_causal * (data['n_T'] - 1))
    n_epochs = int(max_epochs) if max_epochs is not None else n_epochs_full

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    #lf = lambda x: ((1 + math.cos(x * math.pi / n_epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
    #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    mse_loss = nn.MSELoss()
    
    # History tracking
    loss_ini_hist, loss_pde_hist, loss_bc_hist = [], [], []
    lamb_ini_hist, lamb_pde_hist, lamb_bc_hist = [], [], []
    lamb = [1, 1, 1]  # [lamb_ini, lamb_pde, lamb_bc]
    
    # Check if BC is enabled
    use_bc = (config.bc_type == 'absorbing')
    
    # Early save flag
    early_saved = False
    loss_threshold = 0.075e-3

    if hp is None:
        hp = config.DEFAULT_HP
    print(f'Training PINN for {n_epochs} epochs... (use_adaptive_lambda={use_adaptive_lambda}, stage_weights={config.USE_STAGE_WEIGHTS})')
      
    i_causal = 0

    for i in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        
        # Initial condition loss
        p = model(r_ini)
        loss_ini = mse_loss(p, p_ini)

        if i % config.n_causal == 0:
            i_causal += 1
        
        # PDE loss
        #r_colloc = rand_colloc(config.n_colloc, data['L'], data['t'][i_causal], device)
        r_colloc = rand_colloc_mixed(config.n_colloc, data['L'], data['t'][i_causal], device, ratio_gaussian=1)
        r_colloc = rand_colloc_fixed(config.n_colloc, data['L'], data['t'][i_causal], device, ratio_gaussian=config.ratio_gaussian)
        
        c_colloc = data['c_func'](r_colloc[:, 0:1], r_colloc[:, 1:2], tensor=True)
        with torch.backends.cudnn.flags(enabled=False):
            p = model(r_colloc)
        pde_res = pde_residual(p, r_colloc, c_colloc)
        loss_pde = mse_loss(pde_res, torch.zeros_like(p))

        # BC loss (absorbing boundary condition)
        if use_bc:
            r_bc, side_bc = rand_boundary(config.n_bc, data['L'], data['t'][i_causal], device)
            c_bc = data['c_func'](r_bc[:, 0:1], r_bc[:, 1:2], tensor=True)
            with torch.backends.cudnn.flags(enabled=False):
                p_bc = model(r_bc)
            bc_res = bc_residual_absorbing(p_bc, r_bc, side_bc, c_bc)
            loss_bc = mse_loss(bc_res, torch.zeros_like(p_bc))
        else:
            loss_bc = torch.tensor(0.0, device=device)

        # Update lambda - 只更新有梯度的损失项
        if use_adaptive_lambda and i % config.n_lamb_update == 0:
            if use_bc:
                loss_lst = [loss_ini, loss_pde, loss_bc]
                lamb = update_lambda(model, loss_lst, lamb, 0.9)
            else:
                # BC 关闭时，只更新 ini 和 pde 的 lambda
                loss_lst = [loss_ini, loss_pde]
                lamb_update = update_lambda(model, loss_lst, lamb[:2], 0.9)
                lamb[0], lamb[1] = lamb_update[0], lamb_update[1]

        # Total loss: loss_ini + loss_pde * lamb_pde/lamb_ini + loss_bc * lamb_bc/lamb_ini
        if config.USE_STAGE_WEIGHTS:
            progress = i / max(n_epochs - 1, 1)
            w_ini, w_pde, w_bc = get_stage_weights(progress, hp)
            lamb = [w_ini, w_pde, w_bc]
            # If BC is disabled, force w_bc=0
            if not use_bc:
                w_bc = 0.0
            loss = w_ini * loss_ini + w_pde * loss_pde + w_bc * loss_bc
        else:
            # adaptive-lambda
            loss = loss_ini + loss_pde * lamb[1] / lamb[0] + loss_bc * lamb[2] / lamb[0]
        # Backpropagate
        loss.backward()
        optimizer.step()
        #scheduler.step()

        # Log
        loss_pde_hist.append(loss_pde.item())
        loss_ini_hist.append(loss_ini.item())
        loss_bc_hist.append(loss_bc.item() if hasattr(loss_bc, 'item') else loss_bc)
        lamb_ini_hist.append(lamb[0].item() if hasattr(lamb[0], 'item') else lamb[0])
        lamb_pde_hist.append(lamb[1].item() if hasattr(lamb[1], 'item') else lamb[1])
        lamb_bc_hist.append(lamb[2].item() if hasattr(lamb[2], 'item') else lamb[2])

        if i % 200 == 199:
            print(f'[{i + 1:5d}] loss: {loss.item() * 1e3:.3f}*1e-3')

            if not early_saved and loss.item() < loss_threshold:
                print(f'\n>>> Loss reached {loss.item():.6f} < {loss_threshold}, saving checkpoint...')
                loss_history = [loss_ini_hist, loss_pde_hist, loss_bc_hist]
                lamb_history = [lamb_ini_hist, lamb_pde_hist, lamb_bc_hist]
                save_bestmodel(model, loss_history, lamb_history, config)
                generate_all_bestfigures(model, data, loss_history, lamb_history, config, device)
                # early_saved = True
                print('>>> Checkpoint saved! Continuing training...\n')

        if i % 5000 == 4999:
            loss_history = [loss_ini_hist, loss_pde_hist, loss_bc_hist]
            lamb_history = [lamb_ini_hist, lamb_pde_hist, lamb_bc_hist]
            save_model(model, loss_history, lamb_history, config)
            generate_all_figures(model, data, loss_history, lamb_history, config, device)
            print('>>> Checkpoint saved! Continuing training...\n')

    loss_history = [loss_ini_hist, loss_pde_hist, loss_bc_hist]
    lamb_history = [lamb_ini_hist, lamb_pde_hist, lamb_bc_hist]
    
    return loss_history, lamb_history


def save_model(model, loss, lamb, config):
    """Save trained model and training history"""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_DIR + "model.pt")
    torch.save(loss, config.MODEL_DIR + "loss.pt")
    torch.save(lamb, config.MODEL_DIR + "lamb.pt")
    print(f"Model saved to {config.MODEL_DIR}")

def save_bestmodel(model, loss, lamb, config):
    """Save trained model and training history"""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_DIR + "best_model.pt")
    torch.save(loss, config.MODEL_DIR + "best_loss.pt")
    torch.save(lamb, config.MODEL_DIR + "best_lamb.pt")
    print(f"Model saved to {config.MODEL_DIR}")


def load_model(model, config, device):
    """Load pre-trained model"""
    print("Loading pre-trained model...")
    model.load_state_dict(torch.load(
        config.MODEL_DIR + "model.pt", 
        weights_only=True, 
        map_location=torch.device('cpu')
    ))
    model.eval()
    model.to(device)
    
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(f"  {param_tensor}\t{model.state_dict()[param_tensor].size()}")

    loss = torch.load(config.MODEL_DIR + "loss.pt", weights_only=False, 
                      map_location=torch.device('cpu'))
    lamb = torch.load(config.MODEL_DIR + "lamb.pt", weights_only=False, 
                      map_location=torch.device('cpu'))
    print("Model loaded!")
    
    return loss, lamb


def evaluate_model(model, data):
    with torch.no_grad():
        p_est = model(data['r_ref']).cpu().numpy()
    p_ref = data['p_ref']

    # 初值误差
    n_L = data["n_L"]
    n_T = data["n_T"]
    p_est = p_est.reshape(-1)  # (n_L*n_L*n_T,)
    p_est = p_est.reshape(n_L, n_L, n_T)  # (n_L, n_L, n_T)
    E_ini = np.linalg.norm(p_est[:, :, 0] - p_ref[:, :, 0]) / (np.linalg.norm(p_ref[:, :, 0]) + 1e-12)

    # 中后期结构误差
    idxs = [int(0.33*data['n_T']), int(0.66*data['n_T'])]
    E_mid = np.mean([
        np.linalg.norm(p_est[:, :, i] - p_ref[:, :, i]) / (np.linalg.norm(p_ref[:, :, i]) + 1e-12)
        for i in idxs
    ])

    # 能量塌缩惩罚
    En_est = np.mean(p_est[:, :, idxs[-1]]**2)
    En_ref = np.mean(p_ref[:, :, idxs[-1]]**2)
    collapse_penalty = max(0.0, np.log(En_ref / (En_est + 1e-12)) - 1.0)

    return 3*E_ini + E_mid + 2*collapse_penalty


# ============================================================================
# Visualization
# ============================================================================
def generate_all_figures(model, data, loss, lamb, config, device):
    """Generate and save all figures"""
    figures_dir = config.FIGURES_DIR
    os.makedirs(figures_dir, exist_ok=True)
    
    L, T, t = data['L'], data['T'], data['t']
    p_ref = data['p_ref']
    n_T, n_L = data['n_T'], data['n_L']
    
    # 1. Save wave speed
    save_wave_speed(
        data['c_func'](data['xx'], data['yy']), L,
        os.path.join(figures_dir, "wave_speed.png")
    )
    
    # 2. Save loss and lambda curves (with three curves: ini, pde, bc)
    label = ['ini', 'pde', 'bc']
    save_train_log(loss, lamb, label, os.path.join(figures_dir, "loss_lambda.png"))
    
    # 3. Generate estimation
    print("Generating reference vs estimation animation...")
    with torch.backends.cudnn.flags(enabled=False):
        p_est = model(data['r_ref'])
    p_est = p_est.cpu().detach().numpy()
    save_estimation_animation(p_ref, p_est, L, 
                              os.path.join(figures_dir, "estimation_animation.gif"))
    
    # 4. Save comparison plot
    save_comparison(p_ref, p_est, L, T, t, 
                    os.path.join(figures_dir, "comparison.png"))
    
    # 5. High resolution estimation
    print("Generating high resolution estimation...")
    t_hr = np.linspace(0, T, n_T * 2).astype(np.float32)
    r_hr = np.linspace(0, L, n_L * 2).astype(np.float32)
    
    x_hr, y_hr = np.meshgrid(r_hr, r_hr)
    r_hr_tensor = np.column_stack((np.reshape(x_hr, (-1, 1)), np.reshape(y_hr, (-1, 1))))
    r_hr_tensor = xyt_tensor(r_hr_tensor, t_hr, device)
    
    # 分批推理，防止显存溢出
    batch_size = 4096  # 可根据显存调整
    p_est_hr_list = []
    with torch.no_grad():
        for i in range(0, r_hr_tensor.shape[0], batch_size):
            batch = r_hr_tensor[i:i+batch_size]
            with torch.backends.cudnn.flags(enabled=False):
                p_est_hr_list.append(model(batch).cpu())
    p_est_hr = torch.cat(p_est_hr_list, dim=0).numpy()
    p_est_hr = np.reshape(p_est_hr, (n_L * 2, n_L * 2, n_T * 2))
    save_field_animation(p_est_hr, L, 'High Res Estimation', 
                         os.path.join(figures_dir, "high_res_estimation.gif"))
    
    print("\n" + "=" * 50)
    print(f"All figures saved to: {figures_dir}")
    print("=" * 50)


def generate_all_bestfigures(model, data, loss, lamb, config, device):
    """Generate and save all figures"""
    figures_dir = config.FIGURES_DIR
    os.makedirs(figures_dir, exist_ok=True)
    
    L, T, t = data['L'], data['T'], data['t']
    p_ref = data['p_ref']
    n_T, n_L = data['n_T'], data['n_L']
    
    # 1. Save wave speed
    save_wave_speed(
        data['c_func'](data['xx'], data['yy']), L,
        os.path.join(figures_dir, "best_wave_speed.png")
    )
    
    # 2. Save loss and lambda curves (with three curves: ini, pde, bc)
    label = ['ini', 'pde', 'bc']
    save_train_log(loss, lamb, label, os.path.join(figures_dir, "best_loss_lambda.png"))
    
    # 3. Generate estimation
    print("Generating reference vs estimation animation...")
    with torch.backends.cudnn.flags(enabled=False):
        p_est = model(data['r_ref'])
    p_est = p_est.cpu().detach().numpy()
    save_estimation_animation(p_ref, p_est, L, 
                              os.path.join(figures_dir, "best_estimation_animation.gif"))
    
    # 4. Save comparison plot
    save_comparison(p_ref, p_est, L, T, t, 
                    os.path.join(figures_dir, "best_comparison.png"))
    
    # 5. High resolution estimation
    print("Generating high resolution estimation...")
    t_hr = np.linspace(0, T, n_T * 2).astype(np.float32)
    r_hr = np.linspace(0, L, n_L * 2).astype(np.float32)
    
    x_hr, y_hr = np.meshgrid(r_hr, r_hr)
    r_hr_tensor = np.column_stack((np.reshape(x_hr, (-1, 1)), np.reshape(y_hr, (-1, 1))))
    r_hr_tensor = xyt_tensor(r_hr_tensor, t_hr, device)
    
    # 分批推理，防止显存溢出
    batch_size = 4096  # 可根据显存调整
    p_est_hr_list = []
    with torch.no_grad():
        for i in range(0, r_hr_tensor.shape[0], batch_size):
            batch = r_hr_tensor[i:i+batch_size]
            with torch.backends.cudnn.flags(enabled=False):
                p_est_hr_list.append(model(batch).cpu())
    p_est_hr = torch.cat(p_est_hr_list, dim=0).numpy()
    p_est_hr = np.reshape(p_est_hr, (n_L * 2, n_L * 2, n_T * 2))
    save_field_animation(p_est_hr, L, 'High Res Estimation', 
                         os.path.join(figures_dir, "best_high_res_estimation.gif"))
    
    print("\n" + "=" * 50)
    print(f"All figures saved to: {figures_dir}")
    print("=" * 50)

# ============================================================================
# Main Function
# ============================================================================
def main():
    """Main entry point for PINN training and evaluation"""
    set_seed(0)
    # Configuration
    config = Config()
    
    # Device setup
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    # Prepare domain and reference solution
    data = prepare_domain(config, device)
    
    # Prepare initial condition
    r_ini, p_ini = prepare_initial_condition(data, config, device)
    
    # Create model
    model = create_model(config, device)
    
    # Train or load model
    if config.TRAIN_NEW_MODEL:
        #loss, lamb = train_model(model, data, r_ini, p_ini, config, device)
        best_hp = config.DEFAULT_HP

        # -----------------------------
        # (Optional) BO search for stage hp
        # -----------------------------
        if config.RUN_BO and config.USE_STAGE_WEIGHTS:
            print(f"Running BO (trials={config.BO_N_TRIALS}, proxy_epochs={config.BO_PROXY_EPOCHS})...")

            def run_one(hp):
                set_seed(0)
                m = create_model(config, device)
                 # BO stage: disable adaptive lambdas to avoid bc domination
                train_model(
                     m, data, r_ini, p_ini, config, device,
                     use_adaptive_lambda=False,
                     hp=hp,
                     max_epochs=config.BO_PROXY_EPOCHS
                 )
                return evaluate_model(m, data)
 
            if _HAS_OPTUNA:
                def objective(trial):
                    hp = {
                        "A_end": config.STAGE_A_END,
                        "B_end": config.STAGE_B_END,
                        "w_pde_A": trial.suggest_float("w_pde_A", 1e-3, 1e-1, log=True),
                        "w_pde_B": trial.suggest_float("w_pde_B", 1e-2, 1.0, log=True),
                        "w_pde_C": trial.suggest_float("w_pde_C", 1e-2, 1.0, log=True),
                        "w_ini_B": trial.suggest_float("w_ini_B", 0.5, 2.0),
                        "w_ini_C": trial.suggest_float("w_ini_C", 0.2, 1.5),
                        "w_bc_C_max": trial.suggest_float("w_bc_C_max", 1e-4, 1e-1, log=True),
                        "bc_ramp_start": trial.suggest_float("bc_ramp_start", 0.7, 0.9),
                        "bc_ramp_len": trial.suggest_float("bc_ramp_len", 0.05, 0.25),
                    }
                    return run_one(hp)

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=config.BO_N_TRIALS, show_progress_bar=True)
                best_hp = study.best_params
                best_hp["A_end"] = config.STAGE_A_END
                best_hp["B_end"] = config.STAGE_B_END
                print("BO best hp:", best_hp)
            else:
                # Fallback: random search
                rng = np.random.RandomState(0)
                best_val = float("inf")
                for k in range(config.BO_N_TRIALS):
                    hp = _random_hp(rng, config)
                    val = run_one(hp)
                    if val < best_val:
                        best_val = val
                        best_hp = hp
                    print(f"[RS] trial {k+1}/{config.BO_N_TRIALS}  val={val:.6f}  best={best_val:.6f}")
                print("Random-search best hp:", best_hp)

        # -----------------------------
        # Full training with best_hp
        # -----------------------------
        loss, lamb = train_model(
            model, data, r_ini, p_ini, config, device,
            use_adaptive_lambda=False if config.USE_STAGE_WEIGHTS else True,
            hp=best_hp,
            max_epochs=None
        )     
        save_model(model, loss, lamb, config)
    
    else:
        loss, lamb = load_model(model, config, device)
    
    # Generate all figures
    generate_all_figures(model, data, loss, lamb, config, device)


if __name__ == "__main__":
    main()

