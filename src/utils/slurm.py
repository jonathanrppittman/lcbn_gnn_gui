import os
import datetime as _dt
import subprocess
from typing import Dict, Any


def _make_sbatch_header(cfg: Dict[str, Any]) -> str:
    slurm = cfg.get("slurm", {})
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name=gnn_job",
        f"#SBATCH --output={cfg['logs_dir']}/%x-%j.out",
    ]
    if slurm.get("account"): lines.append(f"#SBATCH --account={slurm['account']}")
    if slurm.get("partition"): lines.append(f"#SBATCH --partition={slurm['partition']}")
    if slurm.get("qos"): lines.append(f"#SBATCH --qos={slurm['qos']}")
    if slurm.get("gpus", 0): lines.append(f"#SBATCH --gres=gpu:{int(slurm['gpus'])}")
    if slurm.get("cpus", 0): lines.append(f"#SBATCH --cpus-per-task={int(slurm['cpus'])}")
    if slurm.get("mem"): lines.append(f"#SBATCH --mem={slurm['mem']}")
    if slurm.get("time"): lines.append(f"#SBATCH --time={slurm['time']}")
    if slurm.get("additional"): lines.append(slurm["additional"])  # raw extra SBATCH lines
    lines.append("")
    if slurm.get("env_activation"): lines.append(slurm["env_activation"])  # e.g., conda activate
    lines.append("")
    return "\n".join(lines)


def write_job_script(command: str, cfg: Dict[str, Any]) -> str:
    jobs_dir = cfg["jobs_dir"]
    os.makedirs(jobs_dir, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(jobs_dir, f"gnn_job_{stamp}.sh")
    content = _make_sbatch_header(cfg) + "\n" + command + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(path, 0o750)
    return path


def submit_job(script_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(["sbatch", script_path], check=False, capture_output=True, text=True)

