from typing import List
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QFileDialog, QMessageBox, QApplication,
    QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QLineEdit,
    QLabel, QComboBox, QTextEdit, QCheckBox
)
from PyQt5.QtCore import Qt

from utils.config import load_config, save_config
from utils.process_runner import CommandRunner
from utils.slurm import write_job_script, submit_job


def _detect_interpreter(script_path: str) -> str:
    if script_path.endswith(".py"):
        return f"python {script_path}"
    return script_path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GNN GUI")
        self.config = load_config()
        self.runner = None  # type: CommandRunner

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Conversion widgets
        conv_row = QHBoxLayout()
        root.addLayout(conv_row)
        conv_row.addWidget(QLabel("Conversion script:"))
        self.conv_script = QLineEdit(self.config["conversion"]["script_path"])
        conv_row.addWidget(self.conv_script, 1)
        btn_browse_conv = QPushButton("Browse")
        btn_browse_conv.clicked.connect(self._pick_conv_script)
        conv_row.addWidget(btn_browse_conv)

        add_files_row = QHBoxLayout()
        root.addLayout(add_files_row)
        self.btn_add_files = QPushButton("Add .mat files")
        self.btn_add_files.clicked.connect(self._add_mat_files)
        add_files_row.addWidget(self.btn_add_files)
        self.btn_add_dir = QPushButton("Add folder")
        self.btn_add_dir.clicked.connect(self._add_folder)
        add_files_row.addWidget(self.btn_add_dir)
        add_files_row.addWidget(QLabel("Output dir:"))
        self.out_dir = QLineEdit(self.config.get("workspace_dir", ""))
        add_files_row.addWidget(self.out_dir, 1)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._pick_out_dir)
        add_files_row.addWidget(btn_out)

        files_row = QHBoxLayout()
        root.addLayout(files_row)
        self.files_list = QListWidget()
        files_row.addWidget(self.files_list, 1)
        self.btn_convert = QPushButton("Convert to .pt")
        self.btn_convert.clicked.connect(self._run_conversion)
        files_row.addWidget(self.btn_convert)

        # Training widgets
        train_row1 = QHBoxLayout()
        root.addLayout(train_row1)
        train_row1.addWidget(QLabel("Training script:"))
        self.train_script = QLineEdit(self.config["training"]["script_path"])
        train_row1.addWidget(self.train_script, 1)
        btn_browse_train = QPushButton("Browse")
        btn_browse_train.clicked.connect(self._pick_train_script)
        train_row1.addWidget(btn_browse_train)

        train_row2 = QHBoxLayout()
        root.addLayout(train_row2)
        train_row2.addWidget(QLabel("Dataset dir (.pt):"))
        self.dataset_dir = QLineEdit("")
        train_row2.addWidget(self.dataset_dir, 1)
        btn_ds = QPushButton("Browse")
        btn_ds.clicked.connect(self._pick_dataset_dir)
        train_row2.addWidget(btn_ds)

        train_row3 = QHBoxLayout()
        root.addLayout(train_row3)
        train_row3.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["GCN", "GAT", "GATv2", "GraphSAGE", "GTransformer"])
        train_row3.addWidget(self.model_combo)
        train_row3.addWidget(QLabel("Extra args:"))
        self.train_args = QLineEdit(self.config["training"]["default_args"])
        train_row3.addWidget(self.train_args, 1)

        slurm_row = QHBoxLayout()
        root.addLayout(slurm_row)
        self.use_slurm = QCheckBox("Submit with SLURM (sbatch)")
        self.use_slurm.setChecked(self.config["slurm"].get("use_slurm_by_default", False))
        slurm_row.addWidget(self.use_slurm)
        self.btn_train = QPushButton("Run Training")
        self.btn_train.clicked.connect(self._run_training)
        slurm_row.addWidget(self.btn_train)

        # Output console
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(250)
        root.addWidget(self.console, 1)

        # Persist on close
        self.destroyed.connect(self._persist_config)

    # ------------- UI Handlers -------------
    def _pick_conv_script(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select conversion script")
        if path:
            self.conv_script.setText(path)

    def _add_mat_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select .mat files", filter="MAT (*.mat)")
        for f in files:
            if f and self.files_list.findItems(f, Qt.MatchExactly):
                continue
            if f:
                self.files_list.addItem(f)

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder containing .mat files")
        if folder:
            for name in sorted(os.listdir(folder)):
                if name.lower().endswith('.mat'):
                    self.files_list.addItem(os.path.join(folder, name))

    def _pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self.out_dir.setText(d)

    def _pick_train_script(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select training script")
        if path:
            self.train_script.setText(path)

    def _pick_dataset_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select dataset directory (.pt files)")
        if d:
            self.dataset_dir.setText(d)

    # ------------- Command builders -------------
    def _format_args(self, template: str, mapping: dict) -> str:
        result = template
        for k, v in mapping.items():
            result = result.replace("{" + k + "}", v)
        return result

    def _run_conversion(self):
        script = self.conv_script.text().strip()
        if not script:
            QMessageBox.warning(self, "Missing script", "Please select a conversion script.")
            return
        if self.files_list.count() == 0:
            QMessageBox.warning(self, "No files", "Please add .mat files to convert.")
            return
        out_dir = self.out_dir.text().strip() or self.config.get("workspace_dir", "")
        os.makedirs(out_dir, exist_ok=True)
        input_files: List[str] = [self.files_list.item(i).text() for i in range(self.files_list.count())]
        inputs_str = " ".join(f'"{p}"' for p in input_files)
        args_template = self.config["conversion"].get("default_args", "")
        args_filled = self._format_args(args_template, {"inputs": inputs_str, "output_dir": f'"{out_dir}"'})
        command = f"{_detect_interpreter(script)} {args_filled}".strip()
        self._start_command(command)

    def _run_training(self):
        script = self.train_script.text().strip()
        if not script:
            QMessageBox.warning(self, "Missing script", "Please select a training script.")
            return
        dataset = self.dataset_dir.text().strip()
        if not dataset:
            QMessageBox.warning(self, "Missing dataset", "Please select a dataset directory containing .pt files.")
            return
        model = self.model_combo.currentText()
        args_template = self.train_args.text().strip() or self.config["training"].get("default_args", "")
        args_filled = self._format_args(args_template, {"dataset_dir": f'"{dataset}"', "model": model})
        command = f"{_detect_interpreter(script)} {args_filled}".strip()
        if self.use_slurm.isChecked():
            script_path = write_job_script(command, self.config)
            result = submit_job(script_path)
            if result.returncode == 0:
                self._append_console(f"Submitted: {result.stdout}")
            else:
                self._append_console(f"SLURM submit failed: {result.stderr}")
        else:
            self._start_command(command)

    # ------------- Runner -------------
    def _start_command(self, command: str):
        if self.runner and self.runner.isRunning():
            QMessageBox.information(self, "Busy", "A job is already running. Please wait.")
            return
        self.console.clear()
        self._append_console(f"$ {command}\n")
        self.runner = CommandRunner(command, working_dir=self.config.get("workspace_dir"))
        self.runner.output.connect(self._append_console)
        self.runner.finished.connect(self._on_finished)
        self.runner.start()

    def _append_console(self, text: str):
        self.console.moveCursor(self.console.textCursor().End)
        self.console.insertPlainText(text)
        self.console.moveCursor(self.console.textCursor().End)

    def _on_finished(self, code: int):
        self._append_console(f"\nProcess finished with code {code}\n")

    def _persist_config(self):
        self.config["conversion"]["script_path"] = self.conv_script.text().strip()
        self.config["training"]["script_path"] = self.train_script.text().strip()
        save_config(self.config)


if __name__ == "__main__":
    app = QApplication([])
    w = MainWindow()
    w.resize(1000, 700)
    w.show()
    app.exec_()
