from typing import List, Dict
import os
import json
import platform
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QFileDialog, QMessageBox, QApplication,
    QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QLineEdit,
    QLabel, QComboBox, QTextEdit, QCheckBox, QGroupBox, QScrollArea, QFormLayout
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


def _format_label(text: str) -> str:
    if text == "--early_stopping":
        return "Early Stopping (epochs)"
    return text.replace("--", "").replace("_", " ").title()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GNN GUI")
        self.config = load_config()
        self.runner = None  # type: CommandRunner
        self.dataset_file_path = None
        self.param_widgets = {}
        self.is_submitting = False

        self.model_map = {
            "GCN": "GCNConv",
            "GAT": "GATConv",
            "GATv2": "GATv2Conv",
            "GraphSAGE": "SAGEConv",
            "GTransformer": "TransformerConv",
        }

        scroll = QScrollArea()
        self.setCentralWidget(scroll)
        scroll.setWidgetResizable(True)
        scroll_content = QWidget(scroll)
        scroll.setWidget(scroll_content)
        root = QVBoxLayout(scroll_content)

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

        # Environment config
        env_row = QHBoxLayout()
        root.addLayout(env_row)
        env_row.addWidget(QLabel("Conda Environment:"))
        self.conda_env = QLineEdit(self.config.get("conda_env", "NeuroGraph"))
        env_row.addWidget(self.conda_env)

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
        self.out_dir = QLineEdit(os.path.join("..", "NeuroGraph", "data", "NCanda", "raw"))
        add_files_row.addWidget(self.out_dir, 1)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._pick_out_dir)
        add_files_row.addWidget(btn_out)

        conv_opts_row = QHBoxLayout()
        root.addLayout(conv_opts_row)
        conv_opts_row.addWidget(QLabel("ROIs:"))
        self.num_rois = QLineEdit("500")
        conv_opts_row.addWidget(self.num_rois)
        conv_opts_row.addStretch(1)

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

        # Slurm config for conversion
        self.slurm_conversion_group = QGroupBox("SLURM Configuration for Conversion")
        slurm_conversion_layout = QVBoxLayout(self.slurm_conversion_group)
        self.slurm_conversion_config_widget = SlurmConfigWidget(
            self.config, "slurm_conversion", default_job_name="MakeTorchGraphData"
        )
        slurm_conversion_layout.addWidget(self.slurm_conversion_config_widget)
        root.addWidget(self.slurm_conversion_group)
        self.slurm_conversion_group.setVisible(False)

        convert_button_row = QHBoxLayout()
        root.addLayout(convert_button_row)
        convert_button_row.addStretch(1)
        self.btn_convert = QPushButton("Convert to .pt")
        self.btn_convert.clicked.connect(self._run_conversion)
        convert_button_row.addWidget(self.btn_convert)

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
        self.model_combo.addItems(list(self.model_map.keys()))
        train_row3.addWidget(self.model_combo)
        train_row3.addStretch(1)

        # Training parameters editor
        self.train_params_group = QGroupBox("Training Arguments")
        self.train_params_layout = QFormLayout(self.train_params_group)
        root.addWidget(self.train_params_group)

        # Slurm config for training
        self.slurm_training_group = QGroupBox("SLURM Configuration for Training")
        slurm_training_layout = QVBoxLayout(self.slurm_training_group)
        self.slurm_training_config_widget = SlurmConfigWidget(
            self.config, "slurm_training", default_job_name="LCBN_GNN_Training"
        )
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

        self._setup_training_params()
        self._update_slurm_visibility()

        if platform.system() == "Windows":
            self.use_slurm_conversion.setVisible(False)
            self.slurm_conversion_group.setVisible(False)
            self.use_slurm.setVisible(False)
            self.slurm_training_group.setVisible(False)

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
        default_path = self.config.get("default_dataset_path", os.path.expanduser("~"))
        path, _ = QFileDialog.getOpenFileName(
            self, "Select dataset file", default_path, filter="PyTorch data (*.pt);;All files (*)"
        )
        if path:
            self.dataset_file_path = path
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

    def _setup_training_params(self):
        try:
            with open("src/utils/training_args.json", "r") as f:
                params = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            QMessageBox.warning(self, "Could not load training params", f"Could not load or parse training_args.json: {e}")
            params = {}

        # Clear existing widgets
        for i in reversed(range(self.train_params_layout.count())):
            self.train_params_layout.itemAt(i).widget().setParent(None)
        self.param_widgets = {}

        for key, value in params.items():
            if key == "--path":
                continue

            if key == "--model":
                model_name = str(value)
                model_map_inv = {v: k for k, v in self.model_map.items()}
                self.model_combo.setCurrentText(model_map_inv.get(model_name, "GAT"))
            elif key == "--data":
                dataset_filename = str(value)
                self.dataset_file_input.setText(dataset_filename)
                if "--path" in params and dataset_filename:
                    self.dataset_file_path = os.path.join(params["--path"], dataset_filename)
            else:
                label = QLabel(_format_label(key))
                widget = QLineEdit(str(value))
                self.train_params_layout.addRow(label, widget)
                self.param_widgets[key] = widget

        # Set defaults for dedicated widgets if not in params
        if "--model" not in params:
            self.model_combo.setCurrentText("GAT")
        if "--data" not in params:
            self.dataset_file_input.setText("")

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

        label_files: List[str] = [self.labels_list.item(i).text() for i in range(self.labels_list.count())]

        # Ensure there's at least one label file, and take the first one.
        if not label_files:
            QMessageBox.warning(self, "No Label File", "Please add a .mat label file.")
            return

        label_file = label_files[0]  # The script expects a single label file.

        # Construct the command with proper arguments.
        command_parts = [
            _detect_interpreter(script),
            "--inputs", *[f'"{p}"' for p in input_files],
            "--labels", f'"{label_file}"',
            "--output_dir", f'"{out_dir}"',
            "--ROIs", self.num_rois.text().strip()
        ]

        command = " ".join(command_parts)

        if self.use_slurm_conversion.isChecked():
            slurm_config = self.config.get("slurm_conversion", {})
            slurm_config["conda_env"] = self.conda_env.text().strip()
            script_path = update_slurm_script(
                "src/utils/MakeTorchGraphData.sh", command, slurm_config, self.config["jobs_dir"], env_name
            )
            result = submit_job(script_path)
            if result.returncode == 0:
                self._append_console(f"Submitted job: {result.stdout}")
            else:
                self._append_console(f"SLURM submit failed: {result.stderr}")
        else:
            conda_env = self.conda_env.text().strip()
            if platform.system() == "Windows":
                conda_command = f"source activate {conda_env} && {command}"
            else:
                conda_command = f"conda activate {conda_env} && {command}"
            self._start_command(conda_command)

    def _run_training(self) -> None:
        if self.is_submitting:
            QMessageBox.warning(self, "Busy", "A job is already being submitted, please wait.")
            return

        try:
            self.is_submitting = True
            script = self.train_script.text().strip()
            if not script:
                QMessageBox.warning(self, "Missing script", "Please select a training script.")
                return

            # Gather parameters from the dynamically generated widgets
            params = {}
            for key, widget in self.param_widgets.items():
                value = widget.text().strip()
                # Attempt to convert to number if possible, else keep as string
                try:
                    if '.' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = value

            # Add parameters from dedicated widgets
            dataset = self.dataset_file_input.text().strip()
            if not dataset:
                QMessageBox.warning(self, "Missing dataset", "Please select a dataset file (.pt).")
                return
            params["--data"] = dataset

            model_display_name = self.model_combo.currentText()
            model_script_name = self.model_map.get(model_display_name, model_display_name)
            params["--model"] = model_script_name

            if self.dataset_file_path:
                params["--path"] = os.path.dirname(self.dataset_file_path)

            # Save the updated parameters back to the JSON file
            try:
                with open("src/utils/training_args.json", "w") as f:
                    json.dump(params, f, indent=4)
            except IOError as e:
                QMessageBox.warning(self, "Save failed", f"Could not save training arguments to file: {e}")
                return

            # Construct the command from the parameters
            args_list = []
            for key, value in params.items():
                args_list.append(str(key))
                args_list.append(str(value))

            args_filled = " ".join(shlex.quote(arg) for arg in args_list)

            if self.use_slurm.isChecked():
                python_script = "main_NCanda.py"
                command = f"python {python_script} {args_filled}".strip()
                slurm_config = self.config.get("slurm_training", {})
                slurm_config["conda_env"] = self.conda_env.text().strip()
                script_path = update_slurm_script(script, command, slurm_config, self.config["jobs_dir"])
                result = submit_job(script_path)
                if result.returncode == 0:
                    self._append_console(f"Submitted: {result.stdout}")
                else:
                    self._append_console(f"SLURM submit failed: {result.stderr}")
            else:
                command = f"{_detect_interpreter(script)} {args_filled}".strip()
                conda_env = self.conda_env.text().strip()
                if platform.system() == "Windows":
                    conda_command = f"source activate {conda_env} && {command}"
                else:
                    conda_command = f"conda activate {conda_env} && {command}"
                self._start_command(conda_command)
        finally:
            self.is_submitting = False

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
        self.config["conda_env"] = self.conda_env.text().strip()
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
