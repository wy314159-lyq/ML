# dialogs.py

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QDialogButtonBox)
from mpl_canvas import MplCanvas

class ComparisonDialog(QDialog):
    """
    A dialog for showing a side-by-side comparison of two plots.
    Used to visualize data "before" and "after" a transformation.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Before vs. After Comparison")
        self.setMinimumSize(800, 400)

        # Main layout
        main_layout = QVBoxLayout(self)

        # Plotting canvases
        plot_layout = QHBoxLayout()
        self.before_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.after_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        plot_layout.addWidget(self.before_canvas)
        plot_layout.addWidget(self.after_canvas)
        main_layout.addLayout(plot_layout)

        # OK and Cancel buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        # Set titles for clarity
        self.before_canvas.axes.set_title("Before")
        self.after_canvas.axes.set_title("After") 