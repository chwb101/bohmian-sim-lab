# Bohmian Simulation Lab

This project provides a Python-based simulation toolkit for visualizing the evolution of a wavefunction and the corresponding Bohmian particle trajectories as they interact with a potential barrier.

## Getting Started

1. Run `tdse_solver.py` to simulate the evolution of the wavefunction.
2. Inspect the output using `tdse_visualizer.py`.
3. Generate Bohmian particle trajectories with `bohm_trajectories_calculator.py`.
4. View the system dynamically with `bohmian_viewer.py`.
5. Optionally export the simulation as a movie with `bohmian_viewer_export.py` and `bohmian_movie_maker.py`.

## File Structure

| File                         | Description |
|------------------------------|-------------|
| `tdse_solver.py`             | Solves the time-dependent Schrödinger equation. |
| `tdse_visualizer.py`         | Visualizes the evolved wavefunction. |
| `bohm_trajectories_calculator.py` | Computes Bohmian trajectories. |
| `bohmian_viewer.py`          | Interactive viewer for particles, amplitude, and quantum potential. |
| `bohmian_viewer_export.py`   | Exports simulation frames as PNGs. |
| `bohmian_movie_maker.py`     | Compiles frames into a movie. |
| `pyproject.toml` + `poetry.lock` | Environment and dependencies. |

## Notes

- Output files (`.npz`, `.mp4`) are not included. Students are expected to run the simulations themselves.
- For AI assistance using VS Code, see `prompts/intro.prompt.md`.