import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, label
import os

# === Configurable Parameters ===
SHOW_PARTICLES = True                  # Toggle particle visibility
PARTICLE_MARKER_SIZE = 0.8             # Size of each Bohmian particle marker
PARTICLE_ALPHA = 0.1                   # Transparency (alpha) for particle markers
PARTICLE_COLOR = 'k'                   # Particle color ('k' = black)
Q_VMIN = -75                           # Minimum value for Q color scale
Q_VMAX = 75                            # Maximum value for Q color scale
EXPORT_SNAPSHOT_BUTTON = True          # Whether to show "Save Frame" button
SHOW_POTENTIAL_OVERLAY = False         # Toggle plotting of potential contours

# === Color of Bohmian Particles ====
THRESHOLD_X = -10.335  # Threshold for initial x-position coloring. 
                    # All particles begin with the same kinetic energy
                    # but some have greater potential energy.  The particles
                    # with greater total energy pass over the barrier and the  
                    # others are caught in the quantum potential and get guided 
                    # back.  For the variable values chosen, this division
                    # occurs at x=-10.335.   
                    # As a challenge try to write code to figure out this value. 

# === Quantum Potential Mask Settings ===
Q_MASK_THRESHOLD = 1e-4                # Threshold for smoothed |ψ| to include region in Q mask
Q_SMOOTH_SIGMA = 1.5                   # Gaussian smoothing width for mask
Q_MIN_REGION_AREA = 16                 # Minimum region area (in pixels) to retain in mask

# === Axes limits ===
XMIN, XMAX = -17.0, 17.0               # Domain limits in x
YMIN, YMAX = -7, 7                     # Domain limits in y

ħ = 1.0                                # Planck constant (set to 1 for units)
m = 1.0                                # Mass (set to 1 for units)
eps = 1e-12                            # Small value to avoid division by zero

# === Load data ===
psi_data = np.load("output/psi_data.npz")
psi = psi_data["psi"]                 # Wavefunction array, shape (Ny, Nx, Nt)
x = psi_data["x"]
y = psi_data["y"]
dt = psi_data["dt"]
Nt = psi.shape[2]                     # Number of time steps

traj_data = np.load("output/bohm_trajectories.npz")
trajectories = traj_data["trajectories"]  # Particle trajectories, shape (n_particles, Nt, 2)
n_particles = trajectories.shape[0]


# Determine particle colors based on initial x-position
initial_positions = trajectories[:, 0, :]
particle_colors = np.where(initial_positions[:, 0] < THRESHOLD_X, 'red', 'blue')

# === Grid setup ===
Ny, Nx = psi.shape[:2]
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)
time = np.arange(Nt) * dt            # Time array in physical units

# === Optional potential overlay (for debugging or visual context) ===
if SHOW_POTENTIAL_OVERLAY:
    a = 0.01
    V = a * X**2 + a * Y**2          # Example: harmonic potential (not used in dynamics here)

# === Quantum Potential computation ===
def compute_Q(R):
    R = np.maximum(R, eps)           # Avoid division by zero
    laplacian = (
        -4 * R +
        np.roll(R, 1, axis=0) + np.roll(R, -1, axis=0) +
        np.roll(R, 1, axis=1) + np.roll(R, -1, axis=1)
    ) / (dx * dy)
    return - (ħ ** 2) / (2 * m) * laplacian / R

# === Precompute Q for all time steps ===
print("Precomputing Q potential...")
Q_array = np.stack([compute_Q(np.abs(psi[:, :, t])) for t in range(Nt)])
print("Done computing Q.")

# === Utility function: build cleaned mask based on smoothed |ψ| amplitude ===
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


# === Set up figure and layout ===
fig = plt.figure(figsize=(7, 9))     # Overall window size (in inches)

# Define a 3x2 grid: 3 vertical rows (ψ, Q, trajectories), colorbars on the right
gs = GridSpec(3, 2, width_ratios=[1, 0.03], height_ratios=[1, 1, 1], hspace=0.25, wspace=0.05)

# === Main panel axes ===
ax_traj = fig.add_subplot(gs[0, 0])   # Top panel: particle trajectories
ax_psi  = fig.add_subplot(gs[1, 0], sharex=ax_traj, sharey=ax_traj)  # Middle: |ψ|
ax_q    = fig.add_subplot(gs[2, 0], sharex=ax_traj, sharey=ax_traj)  # Bottom: Q

# === Colorbar axes (small side panels) ===
cax_psi = fig.add_subplot(gs[1, 1])
cax_q   = fig.add_subplot(gs[2, 1])

# === Trajectories panel ===
ax_traj.set_title(f"Bohmian Trajectories (N = {n_particles})")
ax_traj.set_xlim(XMIN, XMAX)
ax_traj.set_ylim(YMIN, YMAX)
ax_traj.set_aspect('auto')

# Create an initially empty scatter plot for particle positions
#particle_plot, = ax_traj.plot([], [], 'o',
#    color=PARTICLE_COLOR, markersize=PARTICLE_MARKER_SIZE, alpha=PARTICLE_ALPHA)

# Dummy initial positions (will be replaced immediately)
# Dummy initial positions (to match number of particles for coloring)
initial_dummy_positions = np.zeros((n_particles, 2))

particle_plot = ax_traj.scatter(initial_dummy_positions[:, 0],
                                initial_dummy_positions[:, 1],
                                s=PARTICLE_MARKER_SIZE**2,
                                c=particle_colors,
                                alpha=PARTICLE_ALPHA)

# Hide particles until animation starts
particle_plot.set_offsets(np.empty((0, 2)))



particle_plot.set_visible(True)

# === |ψ| panel ===
psi_img = ax_psi.imshow(np.abs(psi[:, :, 0]), extent=[x[0], x[-1], y[0], y[-1]],
                        origin='lower', cmap='turbo', aspect='auto')
ax_psi.set_title("|ψ(x, y)|")
ax_psi.set_xlim(XMIN, XMAX)
ax_psi.set_ylim(YMIN, YMAX)
plt.colorbar(psi_img, cax=cax_psi)

# Optional contour overlay for potential
if SHOW_POTENTIAL_OVERLAY:
    ax_psi.contour(X, Y, V, levels=10, linewidths=0.5, colors='k', alpha=0.3)

# === Q panel (with masked colormap) ===
from matplotlib import cm
seismic_with_nan = plt.colormaps.get_cmap('seismic').copy()
seismic_with_nan.set_bad(color='white')  # Masked areas show as white

# Initialize with blank (NaN-filled) array
q_img = ax_q.imshow(np.full_like(Q_array[0], np.nan), extent=[x[0], x[-1], y[0], y[-1]],
                    origin='lower', cmap=seismic_with_nan, aspect='auto',
                    vmin=Q_VMIN, vmax=Q_VMAX)
ax_q.set_title("Quantum Potential Q(x, y)")
ax_q.set_xlim(XMIN, XMAX)
ax_q.set_ylim(YMIN, YMAX)
plt.colorbar(q_img, cax=cax_q)

# === Controls ===

# ax_slider defines the position of the time slider bar:
# [left, bottom, width, height] — all values are in figure-relative coordinates (0 to 1)
ax_slider = plt.axes([0.16, 0.03, 0.50, 0.03])  # Centered slider bar near bottom
slider = Slider(ax_slider, 't', 0, Nt - 1, valinit=0, valstep=1)

# ax_button defines the "Play/Pause" button:
# [left, bottom, width, height]
ax_button = plt.axes([0.72, 0.01, 0.1, 0.04])
button = Button(ax_button, 'Play')
playing = [False]  # Mutable container so the toggle state persists

# ax_save defines the "Save Frame" button:
# [left, bottom, width, height]
if EXPORT_SNAPSHOT_BUTTON:
    ax_save = plt.axes([0.84, 0.01, 0.14, 0.04])
    save_button = Button(ax_save, 'Save Frame')

animation_initialized = [False]

# === Frame update logic ===
def update_frame(t):
    t = int(t)
    #print(f"Rendering Q at t = {t} (t_phys = {time[t]:.4f})")

    # Update particles
    #particle_plot.set_data(trajectories[:, t, 0], trajectories[:, t, 1])

    particle_plot.set_offsets(trajectories[:, t, :])


    ax_traj.set_title(f"Bohmian Trajectories (N = {n_particles}) at t = {time[t]:.2f}")

    # Update |ψ|
    amp = np.abs(psi[:, :, t])
    psi_img.set_data(amp)
    psi_img.set_clim(0, np.max(amp))

    # Update Q with masked regions
    Q = Q_array[t]
    support_mask = build_clean_mask(amp, threshold=Q_MASK_THRESHOLD,
                                    sigma=Q_SMOOTH_SIGMA,
                                    min_area=Q_MIN_REGION_AREA)
    Q_masked = np.where(support_mask, Q, np.nan)
    q_img.set_data(Q_masked)
    q_img.set_clim(Q_VMIN, Q_VMAX)

    # Sync slider
    slider.eventson = False
    slider.set_val(t)
    slider.eventson = True

    return [particle_plot, psi_img, q_img]

# === Slider callback ===
def on_slider_change(val):
    if not animation_initialized[0]:
        particle_plot.set_visible(True)
        animation_initialized[0] = True
    update_frame(int(val))

slider.on_changed(on_slider_change)

# === Play/Pause button callback ===
def on_button_clicked(event):
    playing[0] = not playing[0]
    button.label.set_text("Pause" if playing[0] else "Play")
    if not animation_initialized[0]:
        particle_plot.set_visible(True)
        animation_initialized[0] = True
    update_frame(int(slider.val))

button.on_clicked(on_button_clicked)

# === Save Frame button callback ===
def on_save_clicked(event):
    frame_idx = int(slider.val)
    os.makedirs("exports", exist_ok=True)
    filename = f"exports/bohm_snapshot_t{frame_idx:04d}.png"
    fig.savefig(filename, dpi=200)
    print(f"Saved snapshot to {filename}")

if EXPORT_SNAPSHOT_BUTTON:
    save_button.on_clicked(on_save_clicked)

# === Animation engine ===
current_frame = [0]
def animate(_):
    if playing[0]:
        current_frame[0] = (current_frame[0] + 1) % Nt
        return update_frame(current_frame[0])
    return []

ani = FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)

# === Launch viewer ===
plt.show()
