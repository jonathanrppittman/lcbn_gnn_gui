from typing import List, Dict
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QFileDialog, QMessageBox, QApplication,
    QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QLineEdit,
    QLabel, QComboBox, QTextEdit, QCheckBox, QGroupBox
)
from PyQt5.QtCore import Qt
import re

from utils.config import load_config, save_config
from utils.process_runner import CommandRunner
import shlex
from utils.slurm import update_slurm_script, submit_job
from ui.slurm_config_widget import SlurmConfigWidget


def _detect_interpreter(script_path: str) -> str:
    if script_path.endswith(".py"):
        return f"python {script_path}"
    return script_path


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GNN GUI")
        self.config = load_config()
        self.runner = None  # type: CommandRunner

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Theme switcher
        theme_row = QHBoxLayout()
        root.addLayout(theme_row)
        theme_row.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()

        self.theme_combo.addItems(["Light", "Dark", "Dark Colorful", "Wake Forest"])
        self.theme_combo.currentTextChanged.connect(self._change_theme)
        theme_row.addWidget(self.theme_combo)
        theme_row.addStretch(1)
        self._load_theme()

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
        self.btn_add_files = QPushButton("Add .mat inputs")
        self.btn_add_files.clicked.connect(self._add_mat_files)
        add_files_row.addWidget(self.btn_add_files)
        self.btn_add_labels = QPushButton("Add .mat labels")
        self.btn_add_labels.clicked.connect(self._add_label_files)
        add_files_row.addWidget(self.btn_add_labels)
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

        inputs_col = QVBoxLayout()
        files_row.addLayout(inputs_col)
        inputs_col.addWidget(QLabel("Input Files:"))
        self.files_list = QListWidget()
        self.files_list.setSelectionMode(QListWidget.ExtendedSelection)
        inputs_col.addWidget(self.files_list)
        self.btn_remove_file = QPushButton("Remove Selected")
        self.btn_remove_file.clicked.connect(self._remove_selected_input_file)
        inputs_col.addWidget(self.btn_remove_file)

        labels_col = QVBoxLayout()
        files_row.addLayout(labels_col)
        labels_col.addWidget(QLabel("Label File(s):"))
        self.labels_list = QListWidget()
        self.labels_list.setSelectionMode(QListWidget.ExtendedSelection)
        labels_col.addWidget(self.labels_list)
        self.btn_remove_label = QPushButton("Remove Selected")
        self.btn_remove_label.clicked.connect(self._remove_selected_label_file)
        labels_col.addWidget(self.btn_remove_label)

        actions_row = QHBoxLayout()
        root.addLayout(actions_row)
        self.use_slurm_conversion = QCheckBox("Submit with SLURM (sbatch)")
        self.use_slurm_conversion.setChecked(self.config.get("slurm_conversion", {}).get("use_slurm_by_default", False))
        self.use_slurm_conversion.toggled.connect(self._update_slurm_visibility)
        actions_row.addWidget(self.use_slurm_conversion)
        actions_row.addStretch(1)
        self.btn_convert = QPushButton("Convert to .pt")
        self.btn_convert.clicked.connect(self._run_conversion)
        actions_row.addWidget(self.btn_convert)

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
        train_row2.addWidget(QLabel("Dataset file (.pt):"))
        self.dataset_file_input = QLineEdit("")
        train_row2.addWidget(self.dataset_file_input, 1)
        btn_ds = QPushButton("Browse")
        btn_ds.clicked.connect(self._pick_dataset_file)
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

        # Slurm config for conversion
        self.slurm_conversion_group = QGroupBox("SLURM Configuration for Conversion")
        slurm_conversion_layout = QVBoxLayout(self.slurm_conversion_group)
        self.slurm_conversion_config_widget = SlurmConfigWidget(self.config, "slurm_conversion")
        slurm_conversion_layout.addWidget(self.slurm_conversion_config_widget)
        root.addWidget(self.slurm_conversion_group)
        self.slurm_conversion_group.setVisible(False)

        # Slurm config for training
        self.slurm_training_group = QGroupBox("SLURM Configuration for Training")
        slurm_training_layout = QVBoxLayout(self.slurm_training_group)
        self.slurm_training_config_widget = SlurmConfigWidget(self.config, "slurm_training")
        slurm_training_layout.addWidget(self.slurm_training_config_widget)
        root.addWidget(self.slurm_training_group)
        self.slurm_training_group.setVisible(False)

        slurm_row = QHBoxLayout()
        root.addLayout(slurm_row)
        self.use_slurm = QCheckBox("Submit with SLURM (sbatch)")
        self.use_slurm.setChecked(self.config.get("slurm_training", {}).get("use_slurm_by_default", False))
        self.use_slurm.toggled.connect(self._update_slurm_visibility)
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

        self._update_slurm_visibility()

    # ------------- UI Handlers -------------
    def _update_slurm_visibility(self) -> None:
        self.slurm_conversion_group.setVisible(self.use_slurm_conversion.isChecked())
        self.slurm_training_group.setVisible(self.use_slurm.isChecked())

    def _pick_conv_script(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select conversion script")
        if path:
            self.conv_script.setText(path)

    def _add_mat_files(self) -> None:
        self._add_files_to_list(self.files_list, "Select .mat input files")

    def _add_label_files(self) -> None:
        self._add_files_to_list(self.labels_list, "Select .mat label files")

    def _add_files_to_list(self, list_widget: QListWidget, title: str) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, title, filter="MAT (*.mat)")
        for f in files:
            if f and list_widget.findItems(f, Qt.MatchExactly):
                continue
            if f:
                list_widget.addItem(f)

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder containing .mat files")
        if folder:
            for name in sorted(os.listdir(folder)):
                if name.lower().endswith('.mat'):
                    self.files_list.addItem(os.path.join(folder, name))

    def _pick_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self.out_dir.setText(d)

    def _pick_train_script(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select training script")
        if path:
            self.train_script.setText(path)

    def _pick_dataset_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select dataset file", filter="PyTorch data (*.pt);;All files (*)"
        )
        if path:
            self.dataset_file_input.setText(os.path.basename(path))

    def _remove_selected_input_file(self) -> None:
        self._remove_selected_from_list(self.files_list)

    def _remove_selected_label_file(self) -> None:
        self._remove_selected_from_list(self.labels_list)

    def _remove_selected_from_list(self, list_widget: QListWidget) -> None:
        selected_items = list_widget.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            list_widget.takeItem(list_widget.row(item))

    # ------------- Command builders -------------
    def _format_args(self, template: str, mapping: Dict[str, str]) -> str:
        result = template
        for k, v in mapping.items():
            result = result.replace("{" + k + "}", v)
        return result

    def _run_conversion(self) -> None:
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

        label_files: List[str] = [self.labels_list.item(i).text() for i in range(self.labels_list.count())]
        labels_str = " ".join(f'"{p}"' for p in label_files)

        args_template = self.config["conversion"].get("default_args", "")
        args_filled = self._format_args(args_template, {
            "inputs": inputs_str,
            "labels": labels_str,
            "output_dir": f'"{out_dir}"'
        })
        command = f"{_detect_interpreter(script)} {args_filled}".strip()

        if self.use_slurm_conversion.isChecked():
            slurm_config = self.config.get("slurm_conversion", {})
            script_path = update_slurm_script("src/utils/MakeTorchGraphData.sh", command, slurm_config)
            result = submit_job(script_path)
            if result.returncode == 0:
                self._append_console(f"Submitted job: {result.stdout}")
            else:
                self._append_console(f"SLURM submit failed: {result.stderr}")
        else:
            command = f"{_detect_interpreter(script)} {args_filled}".strip()
            self._start_command(command)

    def _run_training(self) -> None:
        script = self.train_script.text().strip()
        if not script:
            QMessageBox.warning(self, "Missing script", "Please select a training script.")
            return
        dataset = self.dataset_file_input.text().strip()
        if not dataset:
            QMessageBox.warning(self, "Missing dataset", "Please select a dataset file (.pt).")
            return

        model_map = {
            "GCN": "GCNConv",
            "GAT": "GATConv",
            "GATv2": "GATv2Conv",
            "GraphSAGE": "SAGEConv",
            "GTransformer": "TransformerConv",
        }
        model_display_name = self.model_combo.currentText()
        model_script_name = model_map.get(model_display_name, model_display_name)

        args_text = self.train_args.text().strip() or self.config["training"].get("default_args", "")

        # Define the arguments from the GUI
        gui_args = {
            "--model": model_script_name,
            "--data": dataset
        }

        # Split the base arguments string into a list
        args_list = shlex.split(args_text)

        # Remove any existing instances of the GUI-controlled arguments
        for key in gui_args:
            if key in args_list:
                try:
                    # Find the index of the argument and remove it and its value
                    idx = args_list.index(key)
                    args_list.pop(idx)  # Remove the key
                    if idx < len(args_list) and not args_list[idx].startswith('-'):
                        args_list.pop(idx)  # Remove the value
                except ValueError:
                    # The key is not in the list, so we can ignore it
                    pass

        # Now, add the GUI arguments to the list
        for key, value in gui_args.items():
            args_list.append(key)
            args_list.append(value)

        # Join the arguments back into a string
        args_filled = " ".join(shlex.quote(arg) for arg in args_list)
        # If using SLURM, the command to run is a python script inside the sbatch script.
        # Otherwise, it's the script from the input field.
        if self.use_slurm.isChecked():
            # TODO: Make the python script name configurable
            python_script = "main_NCanda.py"
            command = f"python {python_script} {args_filled}".strip()
            slurm_config = self.config.get("slurm_training", {})
            script_path = update_slurm_script(script, command, slurm_config)
            result = submit_job(script_path)
            if result.returncode == 0:
                self._append_console(f"Submitted: {result.stdout}")
            else:
                self._append_console(f"SLURM submit failed: {result.stderr}")
        else:
            command = f"{_detect_interpreter(script)} {args_filled}".strip()
            self._start_command(command)

    # ------------- Runner -------------
    def _start_command(self, command: str) -> None:
        if self.runner and self.runner.isRunning():
            QMessageBox.information(self, "Busy", "A job is already running. Please wait.")
            return
        self.console.clear()
        self._append_console(f"$ {command}\n")
        self.runner = CommandRunner(command, working_dir=self.config.get("workspace_dir"))
        self.runner.output.connect(self._append_console)
        self.runner.finished.connect(self._on_finished)
        self.runner.start()

    def _append_console(self, text: str) -> None:
        self.console.moveCursor(self.console.textCursor().End)
        self.console.insertPlainText(text)
        self.console.moveCursor(self.console.textCursor().End)

    def _on_finished(self, code: int) -> None:
        self._append_console(f"\nProcess finished with code {code}\n")

    def _persist_config(self) -> None:
        self.config["conversion"]["script_path"] = self.conv_script.text().strip()
        self.config["training"]["script_path"] = self.train_script.text().strip()
        if "slurm_conversion" not in self.config:
            self.config["slurm_conversion"] = {}
        self.config["slurm_conversion"]["use_slurm_by_default"] = self.use_slurm_conversion.isChecked()
        if "slurm_training" not in self.config:
            self.config["slurm_training"] = {}
        self.config["slurm_training"]["use_slurm_by_default"] = self.use_slurm.isChecked()
        save_config(self.config)

    def _load_theme(self) -> None:
        theme = self.config.get("theme", "dark colorful")
        self.theme_combo.setCurrentText(theme.title())
        self._apply_theme(theme)

    def _change_theme(self, theme_name: str) -> None:
        theme = theme_name.lower()
        self._apply_theme(theme)
        self.config["theme"] = theme
        save_config(self.config)

    def _apply_theme(self, theme: str) -> None:
        p = os.path.dirname(__file__)
        if theme == "dark":
            qss_path = os.path.join(p, "dark_theme.qss")
        elif theme == "dark colorful":
            qss_path = os.path.join(p, "colorful_theme.qss")
        elif theme == "wake forest":
            qss_path = os.path.join(p, "wake_forest_theme.qss")
        else:  # light theme
            qss_path = ""

        if qss_path:
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())
        else:
            # Reset to default
            self.setStyleSheet("")


if __name__ == "__main__":
    app = QApplication([])
    w = MainWindow()
    w.resize(1000, 700)
    w.show()
    app.exec_()
