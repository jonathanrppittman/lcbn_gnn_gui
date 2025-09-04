import os
import shlex
import subprocess
from typing import Dict, Any, List, Tuple
import re


def parse_sbatch_settings(script_path: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Parses a shell script to extract SBATCH directives and environment setup commands.
    """
    sbatch_settings: Dict[str, str] = {}
    env_setup_lines: List[str] = []

    if not os.path.exists(script_path):
        return sbatch_settings, env_setup_lines

    sbatch_pattern = re.compile(r'#SBATCH\s+(--[\w-]+(?:=[\w\d]+)?|-[a-zA-Z])\s*([^\s]*)')

    with open(script_path, 'r') as f:
        for line in f:
            line = line.strip()
            sbatch_match = sbatch_pattern.match(line)
            if sbatch_match:
                key = sbatch_match.group(1).lstrip('-')
                value = sbatch_match.group(2)
                sbatch_settings[key] = value
            elif line.startswith("module") or line.startswith("conda"):
                env_setup_lines.append(line)

    return sbatch_settings, env_setup_lines


def create_slurm_script(
    script_path: str,
    command: str,
    sbatch_settings: Dict[str, Any],
    env_setup_lines: List[str]
) -> str:
    """
    Generates a new SLURM script file with the specified command and SBATCH directives.
    """
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash -l\n\n")

        # Write SBATCH directives
        for key, value in sbatch_settings.items():
            if value:
                f.write(f"#SBATCH --{key}={value}\n")
        f.write("\n")

        # Write environment setup commands
        if env_setup_lines:
            for line in env_setup_lines:
                f.write(f"{line}\n")
            f.write("\n")

        # Write the srun command
        parts = ["srun"] + shlex.split(command)
        multiline_command = " \\\n    ".join(parts) + "\n"
        f.write(multiline_command)

    os.chmod(script_path, 0o750)
    return script_path


def submit_job(script_path: str) -> subprocess.CompletedProcess:
    """
    Submits a job to SLURM using sbatch.
    """
    return subprocess.run(["sbatch", script_path], check=False, capture_output=True, text=True)

