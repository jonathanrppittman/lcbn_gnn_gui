import os
import subprocess
from typing import Optional
from PyQt5.QtCore import QThread, pyqtSignal


class CommandRunner(QThread):
    output = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, command: str, working_dir: Optional[str] = None, env: Optional[dict] = None):
        super().__init__()
        self.command = command
        self.working_dir = working_dir
        self.env = env or os.environ.copy()

    def run(self) -> None:
        proc = subprocess.Popen([
            "/usr/bin/bash", "-c", self.command
        ], cwd=self.working_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=self.env, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            self.output.emit(line)
        proc.wait()
        self.finished.emit(proc.returncode)

