import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QDoubleSpinBox, QSpinBox, QComboBox, QLabel, QGroupBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from Python_RFEA_Analysis_fcn import full_analysis

def resource_path(rel_path: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel_path)

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    w = MainWindow()
    w.resize(1100, 650)
    w.show()
    sys.exit(app.exec())

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.ax1 = fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()
        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RFEA Analyzer")
        self.setWindowIcon(QIcon("icon.ico"))

        self.csv_path: Path | None = None

        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        # Left panel (controls)
        controls = QVBoxLayout()
        main.addLayout(controls, 0)

        file_box = QGroupBox("File")
        fb = QVBoxLayout(file_box)
        self.btn_open = QPushButton("Open CSV…")
        self.lbl_file = QLabel("No file selected")
        self.lbl_file.setWordWrap(True)
        fb.addWidget(self.btn_open)
        fb.addWidget(self.lbl_file)
        controls.addWidget(file_box)

        param_box = QGroupBox("Parameters")
        pb = QVBoxLayout(param_box)

        self.pressure = QDoubleSpinBox()
        self.pressure.setRange(0.0, 1e9)
        self.pressure.setValue(1.0)
        self.pressure.setDecimals(4)
        pb.addWidget(QLabel("Pressure"))
        pb.addWidget(self.pressure)

        self.smooth_didv = QSpinBox()
        self.smooth_didv.setRange(1, 2000)
        self.smooth_didv.setValue(50)
        pb.addWidget(QLabel("SmoothFactordIdV"))
        pb.addWidget(self.smooth_didv)

        self.smooth_method = QComboBox()
        self.smooth_method.addItems(["Recursive", "Mavg", "SG"])
        pb.addWidget(QLabel("smoothfunctionIV"))
        pb.addWidget(self.smooth_method)

        # smoothIVparam widgets
        self.mavg_window = QSpinBox()
        self.mavg_window.setRange(1, 2000)
        self.mavg_window.setValue(20)

        self.sg_poly = QSpinBox()
        self.sg_poly.setRange(0, 10)
        self.sg_poly.setValue(1)
        self.sg_win = QSpinBox()
        self.sg_win.setRange(3, 2001)
        self.sg_win.setSingleStep(2)
        self.sg_win.setValue(13)

        pb.addWidget(QLabel("Mavg window"))
        pb.addWidget(self.mavg_window)
        pb.addWidget(QLabel("SG polyorder"))
        pb.addWidget(self.sg_poly)
        pb.addWidget(QLabel("SG window_length (odd)"))
        pb.addWidget(self.sg_win)

        controls.addWidget(param_box)

        # --- Sheath / tau_ratio parameters ---
        pb.addWidget(QLabel("Vp (V)"))
        self.vp = QDoubleSpinBox()
        self.vp.setRange(0.0, 1e4)
        self.vp.setDecimals(4)
        self.vp.setValue(20.0)  # default
        pb.addWidget(self.vp)

        pb.addWidget(QLabel("Mi (u)"))
        self.mi = QDoubleSpinBox()
        self.mi.setRange(0.1, 500.0)
        self.mi.setDecimals(4)
        self.mi.setValue(40.0)  # default (Ar+)
        pb.addWidget(self.mi)

        pb.addWidget(QLabel("f_rf (Hz)"))
        self.frf = QDoubleSpinBox()
        self.frf.setRange(0.0, 1e10)
        self.frf.setDecimals(2)
        self.frf.setValue(13.56e6)  # default
        pb.addWidget(self.frf)

        pb.addWidget(QLabel("Te (eV)"))
        self.te = QDoubleSpinBox()
        self.te.setRange(0.01, 1e3)
        self.te.setDecimals(4)
        self.te.setValue(3.0)  # default
        pb.addWidget(self.te)

        pb.addWidget(QLabel("alpha (sheath)"))
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.01, 100.0)
        self.alpha.setDecimals(4)
        self.alpha.setValue(3.0)  # default
        pb.addWidget(self.alpha)

        self.lbl_stats = QLabel("")
        self.lbl_stats.setAlignment(Qt.AlignTop)
        controls.addWidget(self.lbl_stats, 1)

        self.btn_run = QPushButton("Run / Update")
        controls.addWidget(self.btn_run)

        # Holds latest computed arrays for exporting
        self.last = {
            "Vavg": None,
            "Iavg": None,
            "Ismooth": None,
            "E": None,
            "dIdE": None,
        }

        self.btn_export_csv = QPushButton("Export data (CSV)…")
        self.btn_export_plot = QPushButton("Export plot (PNG/SVG)…")
        controls.addWidget(self.btn_export_csv)
        controls.addWidget(self.btn_export_plot)

        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_plot.clicked.connect(self.export_plot)

        self.btn_about = QPushButton("About")
        controls.addWidget(self.btn_about)
        self.btn_about.clicked.connect(self.show_about)

        # Right panel (plot)
        self.canvas = MplCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        main.addWidget(self.canvas, 1)
        plot_container = QVBoxLayout()
        main.addLayout(plot_container, 1)
        plot_container.addWidget(self.toolbar)
        plot_container.addWidget(self.canvas)
        self._cid_move = self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self._hover_text = self.canvas.ax1.text(
            0.02, 0.98, "", transform=self.canvas.ax1.transAxes,
            va="top", ha="left"
        )

        # Signals
        self.btn_open.clicked.connect(self.open_file)
        self.btn_run.clicked.connect(self.run)

        # Auto-update on changes
        for w in [self.pressure, self.smooth_didv, self.mavg_window, self.sg_poly, self.sg_win,
                  self.vp, self.mi, self.frf, self.te, self.alpha]:
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self.run)
            if hasattr(w, "currentIndexChanged"):
                w.currentIndexChanged.connect(self.run)

        self.sync_param_visibility()

        self.smooth_method.currentIndexChanged.connect(self.sync_param_visibility)

    def sync_param_visibility(self):
        method = self.smooth_method.currentText()
        self.mavg_window.setEnabled(method == "Mavg")
        self.sg_poly.setEnabled(method == "SG")
        self.sg_win.setEnabled(method == "SG")

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        self.csv_path = Path(path)
        self.lbl_file.setText(str(self.csv_path))
        self.run()

    def show_about(self):
        from PySide6.QtWidgets import QMessageBox
        from PySide6.QtCore import Qt

        name = "RFEA analysis by P. Hiret"
        company = "Universität Basel"
        email = "paul.hiret@unibas.ch"

        version = "1.0.0"
        text = (
            f"<b>{name}</b> <span style='color:gray'>(v{version})</span><br>"
            f"{company}<br>"
            f"<a href='mailto:{email}'>{email}</a>"
        )

        box = QMessageBox(self)
        box.setWindowTitle("About")
        box.setTextFormat(Qt.RichText)  # enables the mailto link
        box.setText(text)
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()

    def on_mouse_move(self, event):
        # Only if mouse is over the main axes
        if event.inaxes not in (self.canvas.ax1, self.canvas.ax2):
            return
        if event.xdata is None or event.ydata is None:
            return

        # Show coordinates (you can customize)
        x = event.xdata
        y = event.ydata

        # If you want "nearest point" readout, use stored arrays
        V = self.last.get("Vavg")
        I = self.last.get("Iavg")
        Is = self.last.get("Ismooth")
        E = self.last.get("E")
        dIdE = self.last.get("dIdE")
        if V is None or I is None or Is is None or E is None or dIdE is None:
            return

        # Decide which axis we're in and show relevant values
        if event.inaxes == self.canvas.ax1:
            # nearest in V
            idx = int(np.argmin(np.abs(V - x)))
            self._hover_text.set_text(
                f"V={V[idx]:.3g}  I={I[idx]:.3g}  Ismooth={Is[idx]:.3g}"
            )
        else:
            idx = int(np.argmin(np.abs(E - x)))
            self._hover_text.set_text(
                f"E={E[idx]:.3g}  dIdE={dIdE[idx]:.3g}"
            )

        self.canvas.draw_idle()

    def export_csv(self):
        import numpy as np
        import pandas as pd
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        Vavg = self.last.get("Vavg")
        Iavg = self.last.get("Iavg")
        Ismooth = self.last.get("Ismooth")
        E = self.last.get("E")
        dIdE = self.last.get("dIdE")

        if any(x is None for x in [Vavg, Iavg, Ismooth, E, dIdE]):
            QMessageBox.warning(self, "Nothing to export", "Run the analysis first.")
            return

        # Save As dialog (Explorer)
        default_name = "rfea_export.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save data as CSV", default_name, "CSV Files (*.csv)"
        )
        if not path:
            return

        # Align lengths (pad with NaN so everything is included)
        arrays = {
            "Vavg": np.asarray(Vavg).ravel(),
            "Iavg": np.asarray(Iavg).ravel(),
            "Ismooth": np.asarray(Ismooth).ravel(),
            "E": np.asarray(E).ravel(),
            "IEDF_dIdE": np.asarray(dIdE).ravel(),
        }
        maxlen = max(a.size for a in arrays.values())

        def pad(a):
            a = np.asarray(a).ravel()
            if a.size == maxlen:
                return a
            out = np.full(maxlen, np.nan, dtype=float)
            out[: a.size] = a
            return out

        df = pd.DataFrame({k: pad(v) for k, v in arrays.items()})
        try:
            df.to_csv(path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return

        QMessageBox.information(self, "Export complete", f"Saved:\n{path}")

    def export_plot(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        # Save As dialog (Explorer). Let user choose PNG or SVG.
        default_name = "rfea_plot.png"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save plot as…",
            default_name,
            "PNG Image (*.png);;SVG Vector (*.svg)",
        )
        if not path:
            return

        # Ensure extension matches selection
        if selected_filter.startswith("PNG") and not path.lower().endswith(".png"):
            path += ".png"
        if selected_filter.startswith("SVG") and not path.lower().endswith(".svg"):
            path += ".svg"

        try:
            # Save the current matplotlib figure embedded in the canvas
            self.canvas.figure.savefig(path, dpi=300, bbox_inches="tight")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return

        QMessageBox.information(self, "Export complete", f"Saved:\n{path}")

    def run(self):
        self.sync_param_visibility()
        if self.csv_path is None or not self.csv_path.exists():
            return

        pressure = float(self.pressure.value())
        SmoothFactordIdV = int(self.smooth_didv.value())
        method = self.smooth_method.currentText()

        smoothIVparam = None
        if method == "Mavg":
            smoothIVparam = int(self.mavg_window.value())
        elif method == "SG":
            win = int(self.sg_win.value())
            if win % 2 == 0:
                win += 1
            smoothIVparam = (int(self.sg_poly.value()), win)

        Vp = float(self.vp.value())
        Mi = float(self.mi.value())
        f_rf = float(self.frf.value())
        Te = float(self.te.value())
        alpha = float(self.alpha.value())

        (Eavg, flux, dIdE, E, ni,
         Electrode_Voltage, Ion_flux,
         Ismooth, Iavg, Epeak, tau_ratio,
         Vavg) = full_analysis(
            str(self.csv_path),
            pressure,
            boolplot=False,
            SmoothFactordIdV=SmoothFactordIdV,
            smoothfunctionIV=method,
            smoothIVparam=smoothIVparam,
            Mi=Mi,
            f_rf=f_rf,
            Vp=Vp,
            Te=Te,
            alpha=alpha,
        )

        tau_str = f"{tau_ratio:.6g}" if (tau_ratio is not None and np.isfinite(tau_ratio)) else "N/A"

        self.last["Vavg"] = np.asarray(Vavg).ravel()
        self.last["Iavg"] = np.asarray(Iavg).ravel()
        self.last["Ismooth"] = np.asarray(Ismooth).ravel()
        self.last["E"] = np.asarray(E).ravel()
        self.last["dIdE"] = np.asarray(dIdE).ravel()

        self.lbl_stats.setText(
            f"Eavg: {Eavg:.6g}\n"
            f"Flux: {flux:.6g}\n"
            f"ni: {ni:.6g}\n"
            f"Epeak: {Epeak:.6g}\n"
            f"Tau ratio (tau_i/tau_rf): {tau_str}\n"
            f"Electrode_Voltage: {Electrode_Voltage:.6g}\n"
            f"Ion_flux (imported): {Ion_flux:.6g}"
        )

        # ---- Plot update ----
        ax1, ax2 = self.canvas.ax1, self.canvas.ax2
        ax1.clear()
        ax2.clear()

        # --- Prepare data ---
        V = np.asarray(Vavg).ravel()
        I = np.asarray(Iavg).ravel()
        Is = np.asarray(Ismooth).ravel()
        E = np.asarray(E).ravel()
        dI = np.asarray(dIdE).ravel()

        # --- Ensure consistent lengths ---
        n_iv = min(len(V), len(I), len(Is))
        V, I, Is = V[:n_iv], I[:n_iv], Is[:n_iv]

        n_e = min(len(E), len(dI))
        E, dI = E[:n_e], dI[:n_e]

        # MATLAB-style trimming of IEDF
        if n_e > 10:
            E, dI = E[:-10], dI[:-10]

        # ----- Left axis: I–V -----
        ax1.plot(V, I, color="black", linewidth=2.5, label="I")
        ax1.plot(V, Is, color="green", linewidth=2.5, label="Ismooth")

        ax1.set_xlabel("Energy (eV)")
        ax1.set_ylabel("Current (A)", color="black")
        ax1.tick_params(axis="y", colors="black")
        ax1.spines["left"].set_color("black")
        ax1.grid(True)
        ax1.legend(loc="upper left")

        # ----- Right axis: IEDF -----
        # ----- Right axis: IEDF -----
        ax2.plot(E, dI, color="red", linewidth=2.5)

        ax2.set_ylabel("IEDF (a.u.)", color="red")

        # FORCE right-side placement (this fixes the issue)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()

        ax2.tick_params(axis="y", colors="red")
        ax2.spines["right"].set_color("red")

        ymax = np.nanmax(dI) * 1.5 if np.isfinite(np.nanmax(dI)) else 1.0
        ax2.set_ylim(0, ymax)

        ax1.set_title("I–V curve and IEDF")

        self.canvas.draw()


if __name__ == "__main__":
    main()
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1100, 650)
    w.show()
    sys.exit(app.exec())
