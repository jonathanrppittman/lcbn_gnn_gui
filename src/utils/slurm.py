import os
import subprocess
from typing import Dict, Any, List
import re


def update_slurm_script(script_path: str, command: str, slurm_cfg: Dict[str, Any]) -> str:
    """
    Updates the specified SLURM script file with SBATCH directives and the
    python command from the provided configuration. This function overwrites the script.
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"SLURM script not found at: {script_path}")

    with open(script_path, "r") as f:
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
                    if cfg_key in slurm_cfg and slurm_cfg[cfg_key]:
                        new_value = slurm_cfg[cfg_key]
                        new_lines.append(f"{match.group(1)} {new_value}\n")
                        found_match = True
                        break
            if not found_match:
                new_lines.append(line)  # Keep original line if no config override
        elif "python" in stripped_line and not stripped_line.startswith("#"):
            # Replace the python command execution line
            new_lines.append(f"srun {command}\n")
            python_cmd_replaced = True
        else:
            new_lines.append(line)

    # If the python command was not found to be replaced, append it.
    if not python_cmd_replaced:
        new_lines.append(f"\nsrun {command}\n")

    content = "".join(new_lines)

    # Overwrite the original script
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(script_path, 0o750)

    return script_path


def submit_job(script_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(["sbatch", script_path], check=False, capture_output=True, text=True)

