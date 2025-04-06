import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, label
from matplotlib import colormaps as mplcm
import os

# === Configurable Parameters ===
PARTICLE_MARKER_SIZE = 1.0  # Size of each Bohmian particle marker
PARTICLE_ALPHA = 0.4        # Transparency (alpha) for particle markers
SHOW_PARTICLES = True

#===Quantum potential ============
Q_VMIN = -75
Q_VMAX = 75

#=============================
EXPORT_SNAPSHOT_BUTTON = True
SHOW_POTENTIAL_OVERLAY = False

# === Axes limits ===
XMIN, XMAX = -17.0, 17.0
YMIN, YMAX = -7, 7

# === Quantum Potential Mask Settings ===
Q_MASK_THRESHOLD = 1e-4
Q_SMOOTH_SIGMA = 1.5
Q_MIN_REGION_AREA = 16

# === Physical Constants ===
ħ = 1.0
m = 1.0
eps = 1e-12

# === Load Data ===
psi_data = np.load("output/psi_data.npz")
psi = psi_data["psi"]
x = psi_data["x"]
y = psi_data["y"]
dt = psi_data["dt"]
Nt = psi.shape[2]

traj_data = np.load("output/bohm_trajectories.npz")
trajectories = traj_data["trajectories"]
n_particles = trajectories.shape[0]

# === Color by Initial x Position ===
initial_x = trajectories[:, 0, 0]
x_min = -15
x_max = -5
normed = np.clip((initial_x - x_min) / (x_max - x_min), 0, 1)
cmap = mplcm.get_cmap("coolwarm")
particle_colors = cmap(normed)

# === Grid ===
Ny, Nx = psi.shape[:2]
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)
time = np.arange(Nt) * dt

# === Optional Potential ===
if SHOW_POTENTIAL_OVERLAY:
    a = 0.01
    V = a * X**2 + a * Y**2

# === Quantum Potential ===
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

# === Masking ===
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

# === Plot Setup ===
fig = plt.figure(figsize=(7, 9))
gs = GridSpec(3, 2, width_ratios=[1, 0.03], height_ratios=[1, 1, 1], hspace=0.25, wspace=0.05)
ax_traj = fig.add_subplot(gs[0, 0])
ax_psi  = fig.add_subplot(gs[1, 0], sharex=ax_traj, sharey=ax_traj)
ax_q    = fig.add_subplot(gs[2, 0], sharex=ax_traj, sharey=ax_traj)
cax_psi = fig.add_subplot(gs[1, 1])
cax_q   = fig.add_subplot(gs[2, 1])

# === Trajectories Panel ===
ax_traj.set_title(f"Bohmian Trajectories (N = {n_particles})")
ax_traj.set_xlim(XMIN, XMAX)
ax_traj.set_ylim(YMIN, YMAX)
ax_traj.set_aspect('auto')

initial_dummy_positions = np.zeros((n_particles, 2))
particle_plot = ax_traj.scatter(
    initial_dummy_positions[:, 0],
    initial_dummy_positions[:, 1],
    s=PARTICLE_MARKER_SIZE**2,
    c=particle_colors,
    alpha=PARTICLE_ALPHA
)
particle_plot.set_offsets(np.empty((0, 2)))
particle_plot.set_visible(True)

# === |ψ| Panel ===
psi_img = ax_psi.imshow(np.abs(psi[:, :, 0]), extent=[x[0], x[-1], y[0], y[-1]],
                        origin='lower', cmap='turbo', aspect='auto')
ax_psi.set_title("|ψ(x, y)|")
ax_psi.set_xlim(XMIN, XMAX)
ax_psi.set_ylim(YMIN, YMAX)
plt.colorbar(psi_img, cax=cax_psi)

if SHOW_POTENTIAL_OVERLAY:
    ax_psi.contour(X, Y, V, levels=10, linewidths=0.5, colors='k', alpha=0.3)

# === Q Panel ===
seismic_with_nan = plt.colormaps.get_cmap('seismic').copy()
seismic_with_nan.set_bad(color='white')
q_img = ax_q.imshow(np.full_like(Q_array[0], np.nan), extent=[x[0], x[-1], y[0], y[-1]],
                    origin='lower', cmap=seismic_with_nan, aspect='auto', vmin=Q_VMIN, vmax=Q_VMAX)
ax_q.set_title("Quantum Potential Q(x, y)")
ax_q.set_xlim(XMIN, XMAX)
ax_q.set_ylim(YMIN, YMAX)
plt.colorbar(q_img, cax=cax_q)

# === Controls ===
ax_slider = plt.axes([0.16, 0.03, 0.50, 0.03])
slider = Slider(ax_slider, 't', 0, Nt - 1, valinit=0, valstep=1)
ax_button = plt.axes([0.72, 0.01, 0.1, 0.04])
button = Button(ax_button, 'Play')
playing = [False]

if EXPORT_SNAPSHOT_BUTTON:
    ax_save = plt.axes([0.84, 0.01, 0.14, 0.04])
    save_button = Button(ax_save, 'Save Frame')

animation_initialized = [False]

# === Update Function ===
def update_frame(t):
    t = int(t)
    particle_plot.set_offsets(trajectories[:, t, :])
    ax_traj.set_title(f"Bohmian Trajectories (N = {n_particles}) at t = {time[t]:.2f}")

    amp = np.abs(psi[:, :, t])
    psi_img.set_data(amp)
    psi_img.set_clim(0, np.max(amp))

    Q = Q_array[t]
    support_mask = build_clean_mask(amp, threshold=Q_MASK_THRESHOLD,
                                    sigma=Q_SMOOTH_SIGMA,
                                    min_area=Q_MIN_REGION_AREA)
    Q_masked = np.where(support_mask, Q, np.nan)
    q_img.set_data(Q_masked)
    q_img.set_clim(Q_VMIN, Q_VMAX)

    slider.eventson = False
    slider.set_val(t)
    slider.eventson = True

    return [particle_plot, psi_img, q_img]

# === Slider Callback ===
def on_slider_change(val):
    if not animation_initialized[0]:
        particle_plot.set_visible(True)
        animation_initialized[0] = True
    update_frame(int(val))

slider.on_changed(on_slider_change)

# === Play/Pause Callback ===
def on_button_clicked(event):
    playing[0] = not playing[0]
    button.label.set_text("Pause" if playing[0] else "Play")
    if not animation_initialized[0]:
        particle_plot.set_visible(True)
        animation_initialized[0] = True
    update_frame(int(slider.val))

button.on_clicked(on_button_clicked)

# === Save Frame Callback ===
def on_save_clicked(event):
    frame_idx = int(slider.val)
    os.makedirs("exports", exist_ok=True)
    filename = f"exports/bohm_snapshot_t{frame_idx:04d}.png"
    fig.savefig(filename, dpi=200)
    print(f"Saved snapshot to {filename}")

if EXPORT_SNAPSHOT_BUTTON:
    save_button.on_clicked(on_save_clicked)

# === Animation ===
current_frame = [0]
def animate(_):
    if playing[0]:
        current_frame[0] = (current_frame[0] + 1) % Nt
        return update_frame(current_frame[0])
    return []

ani = FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)

# === Optional: Automatically export a range of frames for movie ===
EXPORT_MOVIE_FRAMES = False # If you want to use bohmian_movie_maker set this to true.  
FRAMES_TO_EXPORT = range(0, 400)  # or `range(Nt)` for all frames

if EXPORT_MOVIE_FRAMES:
    os.makedirs("exports", exist_ok=True)
    for t in FRAMES_TO_EXPORT:
        update_frame(t)
        filename = f"exports/bohm_snapshot_t{t:04d}.png"
        fig.savefig(filename, dpi=200)
        print(f"Saved frame {t} -> {filename}")


# === Launch Viewer ===
plt.show()
