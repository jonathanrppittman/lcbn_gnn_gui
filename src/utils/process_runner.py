import os
import subprocess
from typing import Optional, Iterator, Tuple
from PyQt5.QtCore import QThread, pyqtSignal


class ProcessExecutor:
    """
    Handles the execution of a command in a subprocess and yields its output.
    This class is framework-agnostic and can be tested without a Qt event loop.
    """
    def __init__(self, command: str, working_dir: Optional[str] = None, env: Optional[dict] = None):
        self.command = command
        self.working_dir = working_dir
        self.env = env or os.environ.copy()

    def run(self) -> Iterator[Tuple[str, int]]:
        """
        Executes the command and yields output lines.
        The final yielded value will be an empty string and the return code.
        """
        try:
            proc = subprocess.Popen(
                ["/usr/bin/bash", "-c", self.command],
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=self.env,
                bufsize=1
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                yield line, -1  # -1 indicates the process is still running
            proc.wait()
            yield "", proc.returncode
        except Exception as e:
            # In case of an exception (e.g., command not found), yield the error and a non-zero exit code.
            yield str(e), 1


class CommandRunner(QThread):
    """A QThread that runs a command and emits signals for output and completion."""
    output = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, command: str, working_dir: Optional[str] = None, env: Optional[dict] = None):
        super().__init__()
        self.executor = ProcessExecutor(command, working_dir, env)

    def run(self) -> None:
        """
        Runs the command using the executor and emits signals.
        """
        for line, return_code in self.executor.run():
            if return_code == -1:
                self.output.emit(line)
            else:
                self.finished.emit(return_code)
                break