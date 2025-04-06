# tdse_visualizer.py
# Interactive visualizer for |psi(x,y,t)| with slider and play/pause (no threading)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

# === Load saved wavefunction ===
data = np.load("output/psi_data.npz")
psi = data["psi"]
x = data["x"]
y = data["y"]
dt = data["dt"]
Nt = psi.shape[2]

X, Y = np.meshgrid(x, y)
time = np.arange(Nt) * dt

# === Setup figure and axes ===
fig, ax = plt.subplots(figsize=(6, 5))
plt.subplots_adjust(left=0.1, bottom=0.25)
cax = ax.imshow(np.abs(psi[:, :, 0]), extent=[x[0], x[-1], y[0], y[-1]],
                 origin='lower', cmap='turbo')
title = ax.set_title(f"|psi(x, y)| at t = {time[0]:.2f}")
ax.set_xlabel("x")
ax.set_ylabel("y")
cb = fig.colorbar(cax, ax=ax)

# === Slider setup ===
ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 't', 0, Nt-1, valinit=0, valstep=1)

# === Play/Pause Button ===
ax_button = plt.axes([0.8, 0.1, 0.1, 0.04])
button = Button(ax_button, 'Play')
playing = [False]  # Mutable flag

# === Animation state ===
current_frame = [0]
updating_slider = [False]

# === Update frame ===
def update_frame(frame):
    amp = np.abs(psi[:, :, frame])
    cax.set_data(amp)
    cax.set_clim(0, np.max(amp))  # adaptive caxis
    title.set_text(f"|ψ(x, y)| at t = {time[frame]:.2f}")
    updating_slider[0] = True
    slider.set_val(frame)
    updating_slider[0] = False
    fig.canvas.draw_idle()

# === Slider callback ===
def on_slider_change(val):
    if not updating_slider[0]:
        update_frame(int(val))

# === Play/pause logic ===
def on_button_clicked(event):
    playing[0] = not playing[0]
    button.label.set_text('Pause' if playing[0] else 'Play')

# === Animation function ===
def animate(_):
    if playing[0]:
        current_frame[0] = (current_frame[0] + 1) % Nt
        update_frame(current_frame[0])

# Setup callbacks
slider.on_changed(on_slider_change)
button.on_clicked(on_button_clicked)
ani = FuncAnimation(fig, animate, interval=50, cache_frame_data=False)

plt.show()
