# === Script 1: bohmian_viewer_export.py ===
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, label
from matplotlib import cm
import os

# === Configurable Parameters ===
PARTICLE_MARKER_SIZE = 0.8
PARTICLE_ALPHA = 0.1
PARTICLE_COLOR = 'k'
Q_VMIN = -75
Q_VMAX = 75
Q_MASK_THRESHOLD = 1e-4
Q_SMOOTH_SIGMA = 1.5
Q_MIN_REGION_AREA = 16
XMIN, XMAX = -17.0, 17.0
YMIN, YMAX = -7, 7
ħ = 1.0
m = 1.0
eps = 1e-12

# === Load data ===
psi_data = np.load("output/psi_data.npz")
psi = psi_data["psi"]
x = psi_data["x"]
y = psi_data["y"]
dt = psi_data["dt"]
Nt = psi.shape[2]

traj_data = np.load("output/bohm_trajectories.npz")
trajectories = traj_data["trajectories"]
n_particles = trajectories.shape[0]

Ny, Nx = psi.shape[:2]
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)
time = np.arange(Nt) * dt

# === Quantum Potential computation ===
def compute_Q(R):
    R = np.maximum(R, eps)
    laplacian = (
        -4 * R +
        np.roll(R, 1, axis=0) + np.roll(R, -1, axis=0) +
        np.roll(R, 1, axis=1) + np.roll(R, -1, axis=1)
    ) / (dx * dy)
    return - (ħ ** 2) / (2 * m) * laplacian / R

print("Precomputing Q potential...")
Q_array = np.stack([compute_Q(np.abs(psi[:, :, t])) for t in range(Nt)])
print("Done computing Q.")

def build_clean_mask(amp, threshold=1e-8, sigma=1.0, min_area=16):
    smoothed = gaussian_filter(amp, sigma=sigma)
    raw_mask = smoothed > threshold
    labeled, num_features = label(raw_mask)
    clean_mask = np.zeros_like(raw_mask, dtype=bool)
    for i in range(1, num_features + 1):
        region = (labeled == i)
        if np.sum(region) >= min_area:
            clean_mask |= region
    return clean_mask

# === Setup Figure ===
fig = plt.figure(figsize=(7, 9))
gs = GridSpec(3, 2, width_ratios=[1, 0.03], height_ratios=[1, 1, 1], hspace=0.25, wspace=0.05)
ax_traj = fig.add_subplot(gs[0, 0])
ax_psi  = fig.add_subplot(gs[1, 0], sharex=ax_traj, sharey=ax_traj)
ax_q    = fig.add_subplot(gs[2, 0], sharex=ax_traj, sharey=ax_traj)
cax_psi = fig.add_subplot(gs[1, 1])
cax_q   = fig.add_subplot(gs[2, 1])

for ax in [ax_traj, ax_psi, ax_q]:
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)

ax_traj.set_title(f"Bohmian Trajectories (N = {n_particles})")
particle_plot, = ax_traj.plot([], [], 'o', color=PARTICLE_COLOR, markersize=PARTICLE_MARKER_SIZE, alpha=PARTICLE_ALPHA)

psi_img = ax_psi.imshow(np.abs(psi[:, :, 0]), extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='turbo')
ax_psi.set_title("|\u03c8(x, y)|")
plt.colorbar(psi_img, cax=cax_psi)

seismic_with_nan = cm.get_cmap('seismic').copy()
seismic_with_nan.set_bad(color='white')
q_img = ax_q.imshow(np.full_like(Q_array[0], np.nan), extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap=seismic_with_nan, vmin=Q_VMIN, vmax=Q_VMAX)
ax_q.set_title("Quantum Potential Q(x, y)")
plt.colorbar(q_img, cax=cax_q)

# === Frame update and export ===
def update_frame(t):
    amp = np.abs(psi[:, :, t])
    psi_img.set_data(amp)
    psi_img.set_clim(0, np.max(amp))
    particle_plot.set_data(trajectories[:, t, 0], trajectories[:, t, 1])
    Q = Q_array[t]
    support_mask = build_clean_mask(amp, threshold=Q_MASK_THRESHOLD, sigma=Q_SMOOTH_SIGMA, min_area=Q_MIN_REGION_AREA)
    Q_masked = np.where(support_mask, Q, np.nan)
    q_img.set_data(Q_masked)
    q_img.set_clim(Q_VMIN, Q_VMAX)
    ax_traj.set_title(f"Bohmian Trajectories (N = {n_particles}) at t = {time[t]:.2f}")
    filename = f"exports/bohm_snapshot_t{t:04d}.png"
    fig.savefig(filename, dpi=200)
    print(f"Saved frame {t}/{Nt} → {filename}")

os.makedirs("exports", exist_ok=True)
for t in range(Nt):
    update_frame(t)

print("All frames written. Done.")
# === Script 1 END ===