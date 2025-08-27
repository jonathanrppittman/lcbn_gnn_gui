import os
import datetime as _dt
import subprocess
from typing import Dict, Any


def _make_sbatch_header(cfg: Dict[str, Any]) -> str:
    slurm = cfg.get("slurm", {})
    lines = [
        "#!/bin/bash -l",
        f"#SBATCH --job-name={slurm.get('job_name', 'MakeTorchGraphData')}",
        f"#SBATCH --output={slurm.get('output', '/isilon/datalake/lcbn_research/final/beach/JonathanP/MakeTorchGraphDataOutput.txt')}",
        f"#SBATCH --error={slurm.get('error', '/isilon/datalake/lcbn_research/final/beach/JonathanP/MakeTorchGraphDataError.txt')}",
        "#SBATCH --nodes 1",
    ]
    if slurm.get("partition"): lines.append(f"#SBATCH -p {slurm['partition']}")
    if slurm.get("gpus", 0): lines.append(f"#SBATCH --gpus {int(slurm['gpus'])}")
    if slurm.get("mem"): lines.append(f"#SBATCH --mem {slurm['mem']}")
    if slurm.get("time"): lines.append(f"#SBATCH --time {slurm['time']}")
    if slurm.get("account"): lines.append(f"#SBATCH --account={slurm['account']}")
    if slurm.get("qos"): lines.append(f"#SBATCH --qos={slurm['qos']}")
    if slurm.get("additional"): lines.append(slurm["additional"])  # raw extra SBATCH lines
    lines.append("")
    if slurm.get("env_activation"): lines.append(slurm["env_activation"])  # e.g., conda activate
    lines.append("")
    return "\n".join(lines)


def write_job_script(command: str, cfg: Dict[str, Any]) -> str:
    jobs_dir = cfg["jobs_dir"]
    os.makedirs(jobs_dir, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = cfg.get("slurm", {}).get("job_name", "gnn_job")
    path = os.path.join(jobs_dir, f"{job_name}_{stamp}.sh")
    content = _make_sbatch_header(cfg) + "\n" + command + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(path, 0o750)
    return path


def write_job_script_from_template(command: str, template_path: str, cfg: Dict[str, Any]) -> str:
    """
    Creates a SLURM job script from a template file, replacing the python command.
    """
    with open(template_path, "r") as f:
        template_lines = f.readlines()

    # Find the python execution line to replace it, ignoring commented lines
    command_line_idx = -1
    for i, line in enumerate(template_lines):
        if "python" in line and not line.strip().startswith("#"):
            command_line_idx = i
            break

    if command_line_idx != -1:
        template_lines[command_line_idx] = command + "\n"
    else:
        # If no python command found, append the new command
        template_lines.append("\n" + command + "\n")

    content = "".join(template_lines)

    jobs_dir = cfg["jobs_dir"]
    os.makedirs(jobs_dir, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(jobs_dir, f"gnn_job_{stamp}.sh")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(path, 0o750)
    return path


def submit_job(script_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(["sbatch", script_path], check=False, capture_output=True, text=True)

