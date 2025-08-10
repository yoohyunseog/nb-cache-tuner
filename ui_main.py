from __future__ import annotations

import sys
import time
from typing import List

# BIT helpers
try:
    from .nb_bit import BIT_MAX_NB, BIT_MIN_NB  # type: ignore
except Exception:
    from nb_bit import BIT_MAX_NB, BIT_MIN_NB  # type: ignore

# psutil optional
try:
    import psutil as _psutil  # type: ignore
except Exception:
    _psutil = None

# Qt + Matplotlib
try:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPlainTextEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
        QMainWindow,
    )
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    from PySide6.QtCore import QTimer  # type: ignore
    from PySide6.QtWidgets import (  # type: ignore
        QApplication,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPlainTextEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
        QMainWindow,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # type: ignore

from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, width: float = 7.0, height: float = 3.5, dpi: int = 100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)


class Dashboard(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NB Speed Dashboard (BIT_MAX_NB / BIT_MIN_NB)")
        self.resize(1000, 560)

        # State
        self._series: List[float] = []
        self._nb_max_hist: List[float] = []
        self._nb_min_hist: List[float] = []

        # Controls
        self.lbl_interval = QLabel("Interval(s):")
        self.le_interval = QLineEdit("1.0")
        self.le_interval.setFixedWidth(60)
        self.lbl_win = QLabel("Window(s):")
        self.le_win = QLineEdit("60")
        self.le_win.setFixedWidth(60)

        self.btn_start = QPushButton("START")
        self.btn_end = QPushButton("END")
        self.btn_shot = QPushButton("Screenshot")
        self.btn_rank = QPushButton("Rank/Spec")

        top = QHBoxLayout()
        for w in (self.lbl_interval, self.le_interval, self.lbl_win, self.le_win, self.btn_start, self.btn_end, self.btn_shot, self.btn_rank):
            top.addWidget(w)
        top.addStretch(1)

        # Figure
        self.canvas = MplCanvas(width=8, height=3)
        self.ax = self.canvas.fig.add_subplot(111)
        (self.line_max,) = self.ax.plot([], [], color="#d62728", label="NB_MAX")
        (self.line_min,) = self.ax.plot([], [], color="#1f77b4", label="NB_MIN")
        self.ax.set_xlabel("samples")
        self.ax.set_ylabel("score (0..100)")
        self.ax.legend(loc="upper left")
        self.ax.grid(True, alpha=0.2)

        # Layout + Right panel
        root = QWidget()
        v = QVBoxLayout(root)
        v.addLayout(top)
        mid = QHBoxLayout()
        mid.addWidget(self.canvas, 2)
        self.rank_out = QPlainTextEdit()
        self.rank_out.setReadOnly(True)
        mid.addWidget(self.rank_out, 3)
        v.addLayout(mid)
        self.setCentralWidget(root)

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)

        # Hooks
        self.btn_start.clicked.connect(self.on_start)
        self.btn_end.clicked.connect(self.on_end)
        self.le_interval.editingFinished.connect(self.apply_interval)
        self.btn_shot.clicked.connect(self.take_screenshot_to_clipboard)
        self.btn_rank.clicked.connect(self.show_rank_and_specs)

        # psutil baseline
        if _psutil:
            try:
                _psutil.cpu_percent(interval=None)
            except Exception:
                pass

        # Real-time NB ranking timer (updates right panel frequently)
        self.rank_nb_timer = QTimer(self)
        self.rank_nb_timer.setInterval(1000)
        self.rank_nb_timer.timeout.connect(self._update_rank_realtime)
        self.rank_nb_timer.start()
        QTimer.singleShot(200, self._update_rank_realtime)

        # Finalization flag
        self._finalized = False
        self._session_start_ts: float | None = None
        self._finalized_ts: float | None = None

    def take_screenshot_to_clipboard(self) -> None:
        try:
            pixmap = self.grab()
            app = QApplication.instance()
            cb = app.clipboard()  # type: ignore[attr-defined]
            cb.setPixmap(pixmap)  # type: ignore[attr-defined]
        except Exception:
            pass

    # START/END
    def on_start(self) -> None:
        self.apply_interval()
        if not self.timer.isActive():
            self.timer.start()
        # reset session timing/finalization state
        try:
            self._session_start_ts = time.time()
        except Exception:
            self._session_start_ts = None
        self._finalized = False
        self._finalized_ts = None

    def on_end(self) -> None:
        if self.timer.isActive():
            self.timer.stop()

    def apply_interval(self) -> None:
        try:
            sec = float(self.le_interval.text().strip() or "1.0")
        except Exception:
            sec = 1.0
        self.timer.setInterval(max(50, int(sec * 1000)))

    # Sampling
    def _sample_cpu_speed(self) -> float:
        if _psutil:
            try:
                return float(_psutil.cpu_percent(interval=None))
            except Exception:
                pass
        t = time.time()
        return 50.0 + 30.0 * __import__("math").sin(t)

    def _compute_bit_scores(self) -> tuple[float, float]:
        try:
            win_sec = int(float(self.le_win.text().strip() or "60"))
        except Exception:
            win_sec = 60
        if win_sec <= 1:
            win_sec = 60
        series = self._series[-win_sec:]
        if len(series) < 2:
            # allow minimal series; attempt BIT and fallback
            try:
                smax = float(BIT_MAX_NB(series))
            except Exception:
                smax = float(series[-1]) if series else 0.0
            try:
                smin = float(BIT_MIN_NB(series))
            except Exception:
                smin = float(series[-1]) if series else 0.0
            return smax, smin
        try:
            smax = float(BIT_MAX_NB(series))
        except Exception:
            smax = 0.0
        try:
            smin = float(BIT_MIN_NB(series))
        except Exception:
            smin = 0.0
        return smax, smin

    def _build_nb_rank_lines(self) -> tuple[list[str], list[tuple[int, float]], list[tuple[int, float]], list[float], list[float]]:
        s = list(self._series)
        try:
            win_ui = int(float(self.le_win.text().strip() or "60"))
        except Exception:
            win_ui = 60
        win = max(1, min(win_ui, len(s) if s else 1, 600))
        stride = max(1, win // 5)
        idxs = list(range(win, len(s) + 1, stride)) or ([len(s)] if len(s) >= 1 else [])
        top_max: list[tuple[int, float]] = []
        top_min: list[tuple[int, float]] = []
        vals_max: list[float] = []
        vals_min: list[float] = []
        for i in idxs:
            seg = s[i - win:i]
            try:
                mx = float(BIT_MAX_NB(seg))
            except Exception:
                mx = float(seg[-1]) if seg else 0.0
            try:
                mn = float(BIT_MIN_NB(seg))
            except Exception:
                mn = float(seg[-1]) if seg else 0.0
            top_max.append((i - 1, mx))
            top_min.append((i - 1, mn))
            vals_max.append(mx)
            vals_min.append(mn)
        top_max.sort(key=lambda x: x[1], reverse=True)
        top_min.sort(key=lambda x: x[1])
        lines: list[str] = []
        lines.append("Top 10 NB_MAX (high):")
        for r, (idx, v) in enumerate(top_max[:10], 1):
            lines.append(f"  {r}. idx {idx} → {v:.2f}")
        lines.append("")
        lines.append("Top 10 NB_MIN (low):")
        for r, (idx, v) in enumerate(top_min[:10], 1):
            lines.append(f"  {r}. idx {idx} → {v:.2f}")
        return lines, top_max, top_min, vals_max, vals_min

    def _update_rank_realtime(self) -> None:
        try:
            nb_lines, top_max, top_min, _, _ = self._build_nb_rank_lines()
            existing = self.rank_out.toPlainText() if hasattr(self, 'rank_out') else ''
            tail = ''
            if existing:
                marker = "Top Processes (CPU):"
                pos = existing.find(marker)
                if pos != -1:
                    tail = "\n" + existing[pos:]
            self.rank_out.setPlainText("\n".join(nb_lines) + ("\n\n" + tail if tail else ""))

            # Auto finalize when we have full Top 10 for both
            if (not self._finalized) and (len(top_max) >= 10) and (len(top_min) >= 10):
                self._finalized = True
                # stop sampling and ranking timers
                try:
                    self.on_end()
                except Exception:
                    pass
                try:
                    self.rank_nb_timer.stop()
                except Exception:
                    pass
                # show final result with specs/process list
                try:
                    self._finalized_ts = time.time()
                except Exception:
                    self._finalized_ts = None
                self.show_rank_and_specs()
                # annotate as final
                try:
                    self.rank_out.appendPlainText("\n[Finalized: auto END]")
                except Exception:
                    pass
        except Exception:
            pass

    def show_rank_and_specs(self) -> None:
        # Build NB ranking and process/spec info
        try:
            lines, _, _, vals_max, vals_min = self._build_nb_rank_lines()
            lines.append("")
            # session elapsed and averages
            try:
                if self._finalized_ts and self._session_start_ts:
                    elapsed = self._finalized_ts - self._session_start_ts
                elif self._session_start_ts:
                    elapsed = time.time() - self._session_start_ts
                else:
                    elapsed = None
            except Exception:
                elapsed = None
            if elapsed is not None:
                lines.append(f"Elapsed(s): {elapsed:.2f}")
            # averages
            try:
                if vals_max:
                    avg_max = sum(vals_max) / len(vals_max)
                    lines.append(f"AVG NB_MAX: {avg_max:.2f}")
                if vals_min:
                    avg_min = sum(vals_min) / len(vals_min)
                    lines.append(f"AVG NB_MIN: {avg_min:.2f}")
            except Exception:
                pass

            # process list top 10 by CPU (Windows)
            proc_table = "(process list unavailable)"
            if sys.platform.startswith("win"):
                try:
                    import subprocess, json as _json
                    cmd = [
                        "powershell", "-NoProfile", "-Command",
                        "Get-Process | Sort-Object CPU -Descending | Select-Object -First 10 Id, ProcessName, CPU, WS | ConvertTo-Json -Depth 2 -Compress"
                    ]
                    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=2)
                    data = _json.loads(raw)
                    if isinstance(data, dict):
                        data = [data]
                    header = f"{'Id':>6}  {'Name':<22}  {'CPU(s)':>8}  {'WS(MB)':>8}"
                    rows = [header]
                    for it in data or []:
                        pid = int(it.get('Id', 0))
                        name = str(it.get('ProcessName', ''))[:22]
                        cpuv = float(it.get('CPU', 0.0))
                        wsv = float(it.get('WS', 0.0))
                        wsmb = wsv / (1024 * 1024) if wsv else 0.0
                        rows.append(f"{pid:6d}  {name:<22}  {cpuv:8.2f}  {wsmb:8.1f}")
                    proc_table = "\n".join(rows)
                except Exception:
                    proc_table = "(process list unavailable)"

            # CPU/GPU specs
            try:
                import platform
                cpu_name = platform.processor() or platform.machine()
            except Exception:
                cpu_name = "(cpu unknown)"
            try:
                import importlib
                torch = importlib.import_module("torch")
                if getattr(torch.cuda, "is_available", lambda: False)():
                    gpu_name = torch.cuda.get_device_name(0)
                else:
                    gpu_name = "(gpu unavailable)"
            except Exception:
                gpu_name = "(gpu unavailable)"

            lines.append("Top Processes (CPU):")
            lines.append(proc_table)
            lines.append("")
            lines.append(f"CPU: {cpu_name}")
            lines.append(f"GPU: {gpu_name}")
            self.rank_out.setPlainText("\n".join(lines))
        except Exception:
            self.rank_out.setPlainText("(ranking/specs failed)")

    # Tick
    def on_tick(self) -> None:
        v = self._sample_cpu_speed()
        self._series.append(v)
        if len(self._series) > 3600:
            self._series = self._series[-3600:]

        smax, smin = self._compute_bit_scores()
        self._nb_max_hist.append(smax)
        self._nb_min_hist.append(smin)
        if len(self._nb_max_hist) > 3600:
            self._nb_max_hist = self._nb_max_hist[-3600:]
            self._nb_min_hist = self._nb_min_hist[-3600:]

        xs = list(range(len(self._nb_max_hist)))
        self.line_max.set_data(xs, self._nb_max_hist)
        self.line_min.set_data(xs, self._nb_min_hist)
        if xs:
            self.ax.set_xlim(xs[0], xs[-1] if xs[-1] > xs[0] else xs[0] + 1)
        self.ax.set_ylim(0.0, 100.0)
        self.canvas.draw_idle()


def main() -> None:
    app = QApplication(sys.argv)
    w = Dashboard()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

