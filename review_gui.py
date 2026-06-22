#!/usr/bin/env python3
"""Quick flip-through viewer for a folder of review overlays.

Shows one image at a time, scaled to the window. No buttons / no verification --
just flip through and eyeball. Built for the cut-top recovery overlays
(build_recovery_review.py), but works on any folder of PNGs.

Keys:
  Right / Down / Space / PageDown / j   next
  Left  / Up   / PageUp   / k            previous
  Home / End                            first / last
  f                                     toggle fullscreen
  q / Esc                               quit

Usage:
  python tools/review_gui.py [DIR]     # DIR default: experiments/out/recovery_review
"""
import os, sys, glob

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow


class Viewer(QMainWindow):
    def __init__(self, files):
        super().__init__()
        self.files = files
        self.i = 0
        self.label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background:#202020;")
        self.setCentralWidget(self.label)
        self.resize(1200, 850)
        nxt, prv = lambda: self.step(+1), lambda: self.step(-1)
        for keys, fn in [(("Right", "Down", "Space", "PgDown", "J"), nxt),
                         (("Left", "Up", "PgUp", "K"), prv),
                         (("Home",), lambda: self.go(0)),
                         (("End",), lambda: self.go(len(self.files) - 1)),
                         (("Q", "Esc"), self.close),
                         (("F",), self.toggle_fs)]:
            for k in keys:
                QShortcut(QKeySequence(k), self, activated=fn)
        self.show_current()

    def step(self, d):
        self.go(self.i + d)

    def go(self, j):
        self.i = max(0, min(len(self.files) - 1, j))
        self.show_current()

    def toggle_fs(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def show_current(self):
        self.pix = QPixmap(self.files[self.i])
        self.setWindowTitle(f"[{self.i + 1}/{len(self.files)}]  {os.path.basename(self.files[self.i])}")
        self._render()

    def _render(self):
        if getattr(self, "pix", None) and not self.pix.isNull():
            self.label.setPixmap(self.pix.scaled(
                self.label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        self._render()
        super().resizeEvent(e)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join("experiments", "out", "recovery_review")
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    if not files:
        sys.exit(f"no overlays in {d!r} -- run tools/build_recovery_review.py first")
    app = QApplication(sys.argv)
    Viewer(files).show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
