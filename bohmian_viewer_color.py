# bohm_trajectories_calculator.py
# Computes Bohmian trajectories from a time-dependent wavefunction ψ(x, y, t)

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from scipy.interpolate import RegularGridInterpolator
from multiprocessing import get_context, cpu_count
from functools import partial
import os
import sys
import matplotlib.pyplot as plt

# === Parameters ===
ħ = 1.0
m = 1.0
n_particles_to_add = 3000
sampling_threshold = 1e-7
output_file = "output/bohm_trajectories.npz"
append_trajectories = True
max_workers = 16  # or None to use all cores
initial_threshold = -10  # Threshold for particle color assignment

def integrate_trajectory(pos, dt, Nt, interpolators):
    traj = [pos.copy()]  # pos is [x, y]
    for t in range(Nt - 1):
        vx_interp, vy_interp = interpolators[t]

        def v(p):
            x, y = p
            return np.array([float(vx_interp([y, x])), float(vy_interp([y, x]))])

        r = traj[-1]
        k1 = v(r) * dt
        k2 = v(r + 0.5 * k1) * dt
        k3 = v(r + 0.5 * k2) * dt
        k4 = v(r + k3) * dt
        r_next = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        traj.append(r_next)
    return np.stack(traj)

def main():
    # === Load wavefunction data ===
    data = np.load("output/psi_data.npz")
    psi = data["psi"]
    x = data["x"]
    y = data["y"]
    dt = data["dt"]
    Nt = psi.shape[2]

    Ny, Nx = psi.shape[:2]
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    print(f"Loaded ψ with shape {psi.shape} (Ny={Ny}, Nx={Nx}, Nt={Nt}), dt = {dt}")

    # === Spectral derivative setup ===
    kx = 2 * np.pi * fftfreq(Nx, d=dx)
    ky = 2 * np.pi * fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')

    def spectral_gradient(psi_t):
        psi_hat = fft2(psi_t)
        dψ_dx = ifft2(1j * KX * psi_hat)
        dψ_dy = ifft2(1j * KY * psi_hat)
        return dψ_dx, dψ_dy

    def compute_velocity_field(psi_t):
        dψ_dx, dψ_dy = spectral_gradient(psi_t)
        with np.errstate(divide='ignore', invalid='ignore'):
            v_x = ħ / m * np.imag(dψ_dx / psi_t)
            v_y = ħ / m * np.imag(dψ_dy / psi_t)
            v_x[~np.isfinite(v_x)] = 0.0
            v_y[~np.isfinite(v_y)] = 0.0
        return v_x.real, v_y.real

    # === Sample continuous initial positions from |ψ|² ===
    R = np.abs(psi[:, :, 0])
    P = R**2
    P[R < sampling_threshold] = 0
    P /= P.sum()

    indices = np.random.choice(Nx * Ny, size=n_particles_to_add, p=P.ravel())
    iy, ix = np.unravel_index(indices, (Ny, Nx))

    x0 = x[ix] + np.random.uniform(-dx / 2, dx / 2, size=n_particles_to_add)
    y0 = y[iy] + np.random.uniform(-dy / 2, dy / 2, size=n_particles_to_add)
    initial_positions = np.stack([x0, y0], axis=1)

    # === Assign colors based on distance from the origin ===
    distances = np.sqrt(x0**2 + y0**2)  # Calculate distance from origin for each particle
    colors = ['red' if dist > initial_threshold else 'blue' for dist in distances]

    # === Build velocity field interpolators ===
    interpolators = []
    for t in range(Nt):
        v_x, v_y = compute_velocity_field(psi[:, :, t])
        interp_vx = RegularGridInterpolator((y, x), v_x, bounds_error=False, fill_value=0.0)
        interp_vy = RegularGridInterpolator((y, x), v_y, bounds_error=False, fill_value=0.0)
        interpolators.append((interp_vx, interp_vy))

    # === Parallel RK4 integration ===
    n_workers = min(cpu_count(), max_workers) if max_workers is not None else cpu_count()
    try:
        print(f"Launching trajectory integration with {n_workers} cores...")
        with get_context("spawn").Pool(processes=n_workers, maxtasksperchild=1) as pool:
            f = partial(integrate_trajectory, dt=dt, Nt=Nt, interpolators=interpolators)
            trajectories = pool.map(f, initial_positions)
    except KeyboardInterrupt:
        print("\nInterrupted by user — exiting cleanly.")
        sys.exit(0)

    trajectories = np.stack(trajectories)  # shape: (n_particles, Nt, 2)

    # === Save results ===
    os.makedirs("output", exist_ok=True)

    if append_trajectories and os.path.exists(output_file):
        existing = np.load(output_file)
        old_traj = existing["trajectories"]
        all_traj = np.concatenate([old_traj, trajectories], axis=0)
        np.savez(output_file, trajectories=all_traj, x=x, y=y, dt=dt)
        print(f"Appended trajectories ({old_traj.shape[0]} → {all_traj.shape[0]})")
    else:
        np.savez(output_file, trajectories=trajectories, x=x, y=y, dt=dt)
        print(f"Saved new trajectory file with {n_particles_to_add} particles")

    # === Plot particles with colors based on initial position ===
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Bohmian Trajectories')
    ax.set_xlim(-17.0, 17.0)
    ax.set_ylim(-7, 7)
    
    # Scatter plot for particles, using the colors array
    ax.scatter(initial_positions[:, 0], initial_positions[:, 1], c=colors, s=5, alpha=0.6)
    plt.show()

if __name__ == "__main__":
    main()
