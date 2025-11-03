from PyQt5.QtWidgets import QWidget, QLineEdit, QSpinBox, QFormLayout, QLabel, QComboBox

class SlurmConfigWidget(QWidget):
    def __init__(self, config, config_key="slurm", default_job_name="MakeTorchGraphData", parent=None):
        super().__init__(parent)
        self.config = config
        self.config_key = config_key
        self.default_job_name = default_job_name

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.job_name = QLineEdit()
        self.output = QLineEdit()
        self.error = QLineEdit()
        self.partition = QComboBox()
        self.partition.addItems(["gpu", "gpu-h100", "gpu-amd"])
        self.gpus = QSpinBox()
        self.mem = QLineEdit()
        self.time = QLineEdit()
        self.additional = QLineEdit()
        self.env_activation = QLineEdit()
        self.conda_env = QLineEdit()

        layout.addRow(QLabel("Job Name:"), self.job_name)
        layout.addRow(QLabel("Output File:"), self.output)
        layout.addRow(QLabel("Error File:"), self.error)
        layout.addRow(QLabel("Partition:"), self.partition)
        layout.addRow(QLabel("GPUs:"), self.gpus)
        layout.addRow(QLabel("Memory (e.g., 700000M):"), self.mem)
        layout.addRow(QLabel("Time (HH:MM:SS):"), self.time)
        layout.addRow(QLabel("Additional SBATCH lines:"), self.additional)
        layout.addRow(QLabel("Env Activation:"), self.env_activation)
        layout.addRow(QLabel("Conda Env:"), self.conda_env)

        self._load_config()

    def _load_config(self):
        if self.config_key not in self.config:
            self.config[self.config_key] = {}
        slurm_config = self.config[self.config_key]
        self.job_name.setText(slurm_config.get("job_name", self.default_job_name))
        self.output.setText(slurm_config.get("output", "./logs/train_out.txt"))
        self.error.setText(slurm_config.get("error", "./logs/train_err.txt"))
        self.partition.setCurrentText(slurm_config.get("partition", "gpu-h100"))
        self.gpus.setValue(int(slurm_config.get("gpus", 2)))
        self.mem.setText(slurm_config.get("mem", "700000M"))
        self.time.setText(slurm_config.get("time", "04:00:00"))
        self.additional.setText(slurm_config.get("additional", ""))
        self.env_activation.setText(slurm_config.get("env_activation", ""))
        self.conda_env.setText(slurm_config.get("conda_env", "NeuroGraph"))

        self._connect_signals()

    def _connect_signals(self):
        self.job_name.textChanged.connect(lambda t: self._update_config("job_name", t))
        self.output.textChanged.connect(lambda t: self._update_config("output", t))
        self.error.textChanged.connect(lambda t: self._update_config("error", t))
        self.partition.currentTextChanged.connect(lambda t: self._update_config("partition", t))
        self.gpus.valueChanged.connect(lambda v: self._update_config("gpus", v))
        self.mem.textChanged.connect(lambda t: self._update_config("mem", t))
        self.time.textChanged.connect(lambda t: self._update_config("time", t))
        self.additional.textChanged.connect(lambda t: self._update_config("additional", t))
        self.env_activation.textChanged.connect(lambda t: self._update_config("env_activation", t))
        self.conda_env.textChanged.connect(lambda t: self._update_config("conda_env", t))

    def _update_config(self, key, value):
        if self.config_key not in self.config:
            self.config[self.config_key] = {}
        self.config[self.config_key][key] = value
