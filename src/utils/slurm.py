import os
import datetime as _dt
import subprocess
from typing import Dict, Any, List
import re


def write_job_script(command: str, cfg: Dict[str, Any]) -> str:
    """
    Creates a SLURM job script from a template file, replacing SBATCH directives
    and the python command based on the provided configuration.
    """
    slurm_cfg = cfg.get("slurm", {})
    template_path = slurm_cfg.get("template_path")
    if not template_path or not os.path.exists(template_path):
        raise FileNotFoundError(f"SLURM template script not found at: {template_path}")

    with open(template_path, "r") as f:
        template_lines = f.readlines()

    # Mapping from config key to SBATCH directive
    sbatch_map = {
        "job_name": "--job-name",
        "output": "--output",
        "error": "--error",
        "partition": "-p",
        "gpus": "--gpus",
        "mem": "--mem",
        "time": "--time",
        "account": "--account",
        "qos": "--qos",
    }

    new_lines: List[str] = []
    python_cmd_replaced = False

    for line in template_lines:
        stripped_line = line.strip()
        if stripped_line.startswith("#SBATCH"):
            found_match = False
            for cfg_key, sbatch_key in sbatch_map.items():
                # Regex to match SBATCH directive, accommodating both space and '=' separators
                pattern = re.compile(rf"(#SBATCH\s+{re.escape(sbatch_key)})(?:[=\s])(.*)")
                match = pattern.match(stripped_line)
                if match:
                    if cfg_key in slurm_cfg:
                        new_value = slurm_cfg[cfg_key]
                        new_lines.append(f"{match.group(1)} {new_value}\n")
                        found_match = True
                        break
            if not found_match:
                new_lines.append(line)  # Keep original line if no config override
        elif "python" in stripped_line and not stripped_line.startswith("#"):
            # Replace the python command execution line
            # We assume the command includes 'srun' or similar, which is replaced entirely.
            new_lines.append(f"srun {command}\n")
            python_cmd_replaced = True
        else:
            new_lines.append(line)

    # If the python command was not found to be replaced, append it.
    if not python_cmd_replaced:
        new_lines.append(f"\nsrun {command}\n")

    # Add env activation if specified
    if slurm_cfg.get("env_activation"):
        # Insert before the command
        for i, line in enumerate(new_lines):
            if "srun" in line:
                new_lines.insert(i, f'{slurm_cfg["env_activation"]}\n')
                break

    content = "".join(new_lines)

    jobs_dir = cfg.get("jobs_dir", "./jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = slurm_cfg.get("job_name", "gnn_job")
    path = os.path.join(jobs_dir, f"{job_name}_{stamp}.sh")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(path, 0o750)
    return path


def submit_job(script_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(["sbatch", script_path], check=False, capture_output=True, text=True)

