import os
import subprocess
from typing import Dict, Any
import re

def update_slurm_script(script_path: str, command: str, slurm_cfg: Dict[str, Any]) -> str:
    """
    Updates the specified SLURM script file with a command by replacing a placeholder.
    This function overwrites the script.
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"SLURM script not found at: {script_path}")

    with open(script_path, "r") as f:
        content = f.read()

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

    # Update SBATCH directives
    for cfg_key, sbatch_key in sbatch_map.items():
        if cfg_key in slurm_cfg and slurm_cfg[cfg_key]:
            new_value = slurm_cfg[cfg_key]
            # Regex to match SBATCH directive, accommodating both space and '=' separators
            pattern = re.compile(rf"^(#SBATCH\s+{re.escape(sbatch_key)})(?:[=\s]).*", re.MULTILINE)
            if pattern.search(content):
                content = pattern.sub(rf"\1 {new_value}", content)

    # The command from the GUI is authoritative.
    # The srun part is added here to ensure it's always present.
    new_command_str = f"srun {command}"

    # Replace the placeholder with the new command
    content = content.replace("#COMMAND_PLACEHOLDER", new_command_str)

    # Overwrite the original script
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(script_path, 0o750)

    return script_path


def submit_job(script_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(["sbatch", script_path], check=False, capture_output=True, text=True)
