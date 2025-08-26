from PyQt5.QtWidgets import QWidget, QLineEdit, QSpinBox, QFormLayout, QLabel

class SlurmConfigWidget(QWidget):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.account = QLineEdit()
        self.partition = QLineEdit()
        self.qos = QLineEdit()
        self.gpus = QSpinBox()
        self.cpus = QSpinBox()
        self.mem = QLineEdit()
        self.time = QLineEdit()
        self.additional = QLineEdit()
        self.env_activation = QLineEdit()

        layout.addRow(QLabel("Account:"), self.account)
        layout.addRow(QLabel("Partition:"), self.partition)
        layout.addRow(QLabel("QoS:"), self.qos)
        layout.addRow(QLabel("GPUs:"), self.gpus)
        layout.addRow(QLabel("CPUs:"), self.cpus)
        layout.addRow(QLabel("Memory (e.g., 16G):"), self.mem)
        layout.addRow(QLabel("Time (HH:MM:SS):"), self.time)
        layout.addRow(QLabel("Additional SBATCH lines:"), self.additional)
        layout.addRow(QLabel("Env Activation:"), self.env_activation)

        self._load_config()

    def _load_config(self):
        slurm_config = self.config.get("slurm", {})
        self.account.setText(slurm_config.get("account", ""))
        self.partition.setText(slurm_config.get("partition", ""))
        self.qos.setText(slurm_config.get("qos", ""))
        self.gpus.setValue(int(slurm_config.get("gpus", 0)))
        self.cpus.setValue(int(slurm_config.get("cpus", 0)))
        self.mem.setText(slurm_config.get("mem", ""))
        self.time.setText(slurm_config.get("time", ""))
        self.additional.setText(slurm_config.get("additional", ""))
        self.env_activation.setText(slurm_config.get("env_activation", ""))

        self._connect_signals()

    def _connect_signals(self):
        self.account.textChanged.connect(lambda t: self._update_config("account", t))
        self.partition.textChanged.connect(lambda t: self._update_config("partition", t))
        self.qos.textChanged.connect(lambda t: self._update_config("qos", t))
        self.gpus.valueChanged.connect(lambda v: self._update_config("gpus", v))
        self.cpus.valueChanged.connect(lambda v: self._update_config("cpus", v))
        self.mem.textChanged.connect(lambda t: self._update_config("mem", t))
        self.time.textChanged.connect(lambda t: self._update_config("time", t))
        self.additional.textChanged.connect(lambda t: self._update_config("additional", t))
        self.env_activation.textChanged.connect(lambda t: self._update_config("env_activation", t))

    def _update_config(self, key, value):
        if "slurm" not in self.config:
            self.config["slurm"] = {}
        self.config["slurm"][key] = value
