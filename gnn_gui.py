import sys
import os

# Add src to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import platform
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == "__main__":
    if platform.system() != "Windows":
        xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        if not xdg_runtime_dir or not os.access(xdg_runtime_dir, os.W_OK):
            fallback_dir = os.path.expanduser("~/.gnn_gui_runtime")
            os.makedirs(fallback_dir, exist_ok=True)
            os.environ["XDG_RUNTIME_DIR"] = fallback_dir

    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1000, 700)
    w.show()
    sys.exit(app.exec_())
