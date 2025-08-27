from PyQt5.QtWidgets import QWidget, QLineEdit, QSpinBox, QFormLayout, QLabel, QComboBox

class SlurmConfigWidget(QWidget):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.job_name = QLineEdit()
        self.output = QLineEdit()
        self.error = QLineEdit()
        self.account = QLineEdit()
        self.partition = QComboBox()
        self.partition.addItems(["gpu", "gpu-h100", "gpu-amd"])
        self.qos = QLineEdit()
        self.gpus = QSpinBox()
        self.mem = QLineEdit()
        self.time = QLineEdit()
        self.additional = QLineEdit()
        self.env_activation = QLineEdit()

        layout.addRow(QLabel("Job Name:"), self.job_name)
        layout.addRow(QLabel("Output File:"), self.output)
        layout.addRow(QLabel("Error File:"), self.error)
        layout.addRow(QLabel("Account:"), self.account)
        layout.addRow(QLabel("Partition:"), self.partition)
        layout.addRow(QLabel("QoS:"), self.qos)
        layout.addRow(QLabel("GPUs:"), self.gpus)
        layout.addRow(QLabel("Memory (e.g., 700000M):"), self.mem)
        layout.addRow(QLabel("Time (HH:MM:SS):"), self.time)
        layout.addRow(QLabel("Additional SBATCH lines:"), self.additional)
        layout.addRow(QLabel("Env Activation:"), self.env_activation)

        self._load_config()

    def _load_config(self):
        slurm_config = self.config.get("slurm", {})
        self.job_name.setText(slurm_config.get("job_name", "MakeTorchGraphData"))
        self.output.setText(slurm_config.get("output", ""))
        self.error.setText(slurm_config.get("error", ""))
        self.account.setText(slurm_config.get("account", ""))
        self.partition.setCurrentText(slurm_config.get("partition", "gpu-h100"))
        self.qos.setText(slurm_config.get("qos", ""))
        self.gpus.setValue(int(slurm_config.get("gpus", 2)))
        self.mem.setText(slurm_config.get("mem", "700000M"))
        self.time.setText(slurm_config.get("time", "24:00:00"))
        self.additional.setText(slurm_config.get("additional", ""))
        self.env_activation.setText(slurm_config.get("env_activation", ""))

        self._connect_signals()

    def _connect_signals(self):
        self.job_name.textChanged.connect(lambda t: self._update_config("job_name", t))
        self.output.textChanged.connect(lambda t: self._update_config("output", t))
        self.error.textChanged.connect(lambda t: self._update_config("error", t))
        self.account.textChanged.connect(lambda t: self._update_config("account", t))
        self.partition.currentTextChanged.connect(lambda t: self._update_config("partition", t))
        self.qos.textChanged.connect(lambda t: self._update_config("qos", t))
        self.gpus.valueChanged.connect(lambda v: self._update_config("gpus", v))
        self.mem.textChanged.connect(lambda t: self._update_config("mem", t))
        self.time.textChanged.connect(lambda t: self._update_config("time", t))
        self.additional.textChanged.connect(lambda t: self._update_config("additional", t))
        self.env_activation.textChanged.connect(lambda t: self._update_config("env_activation", t))

    def _update_config(self, key, value):
        if "slurm" not in self.config:
            self.config["slurm"] = {}
        self.config["slurm"][key] = value
