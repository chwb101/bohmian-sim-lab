# tdse_solver.py
# Time-dependent Schrödinger equation solver (2D, split-step Fourier method)

import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
import os

# === Simulation parameters ===
use_slits = False
use_bump = True

Lx, Ly = 30.0, 20.0
Nx, Ny = 640, 640
dx = 2 * Lx / Nx
dy = 2 * Ly / Ny
x = np.linspace(-Lx + dx/2, Lx - dx/2, Nx)
y = np.linspace(-Ly + dy/2, Ly - dy/2, Ny)
X, Y = np.meshgrid(x, y)

# === K-space grid ===
dkx = 2 * np.pi / (Nx * dx)
dky = 2 * np.pi / (Ny * dy)
kx = fft.fftshift(np.arange(-Nx//2, Nx//2) * dkx)
ky = fft.fftshift(np.arange(-Ny//2, Ny//2) * dky)
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2

# === Potential: harmonic + optional bump or slits ===
a = 0.01
V_harm = a * X**2 + a * Y**2

x0, y0 = -10.0, 0.0
px0, py0 = 5.0, 0.0

E0 = 0.5 * px0**2 + a * x0**2
A = E0
b_bump = 4.0

if use_slits:
    c = 1.0
    n = 6
    M = np.ones_like(Y)
    region = np.abs(c * Y) < np.pi
    M[region] = np.cos(c * Y[region])**(2 * n)
else:
    M = np.ones_like(Y)

if use_bump:
    V_bump = A * np.exp(-b_bump * X**2) * M
else:
    V_bump = np.zeros_like(X)

V = V_harm + V_bump

# === Initial wavefunction ===
omega = np.sqrt(2 * a)
sigma = 1.0 / np.sqrt(omega)
psi0 = np.exp(-((X - x0)**2) / (2 * sigma**2) - ((Y - y0)**2) / (2 * sigma**2))
psi0 = psi0 * np.exp(1j * (px0 * (X - x0) + py0 * (Y - y0)))
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx * dy)

# === Time evolution ===
dt = 0.01
T_final = 4.0
Nt = int(T_final / dt)

V_half = np.exp(-1j * V * dt / 2)
T_full = np.exp(-1j * K2 * dt / 2)

psi = psi0.copy()
psi_abs_save = np.zeros((Ny, Nx, Nt), dtype=np.float32)
psi_save = np.zeros((Ny, Nx, Nt), dtype=np.complex64)

for t in range(Nt):
    psi = V_half * psi
    psi_k = fft.fft2(psi)
    psi_k = T_full * psi_k
    psi = fft.ifft2(psi_k)
    psi = V_half * psi

    psi_abs_save[:, :, t] = np.abs(psi)**2
    psi_save[:, :, t] = psi

# === Save output ===
os.makedirs("output", exist_ok=True)
np.savez_compressed("output/psi_data.npz", 
                   psi=psi_save, 
                   psi_abs=psi_abs_save,
                   x=x, y=y, dt=dt, T_final=T_final)

print("Saved wavefunction data to output/psi_data.npz")
