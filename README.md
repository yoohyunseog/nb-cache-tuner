nb-cache-tuner

A lightweight real-time dashboard to visualize NB_MAX and NB_MIN scores over a rolling window using BIT helpers. It samples system activity (CPU by default) and shows:
- NB_MAX/NB_MIN time series (Matplotlib)
- Realtime Top 10 windows of NB ranks
- System specs (CPU/GPU) and Top Processes (Windows)

Features
- Dual Qt backend: PyQt5 preferred, falls back to PySide6 automatically
- Matplotlib embedding for smooth plotting
- psutil optional (CPU sampling). Falls back to a synthetic signal if unavailable
- Windows only: Top Processes section via PowerShell (best-effort)

Requirements
- Python 3.9+
- One of: PyQt5 OR PySide6
- matplotlib
- Optional: psutil (CPU sampling), torch (GPU name)

Quick start
- Install dependencies
  pip install PyQt5 matplotlib psutil
  # or
  pip install PySide6 matplotlib psutil
- Run
  python ui_main.py

Notes
- BIT helpers live in nb_bit.py and are imported as BIT_MAX_NB / BIT_MIN_NB
- On non-Windows systems, process list may be unavailable
- The app is defensive; most optional features fail gracefully

License
- TBD
