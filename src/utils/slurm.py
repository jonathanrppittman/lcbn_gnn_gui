import os
import subprocess
from typing import Dict, Any, List
import re
import shlex


def update_slurm_script(script_path: str, command: str, slurm_cfg: Dict[str, Any]) -> str:
    """
    Updates the specified SLURM script file with SBATCH directives and the
    python command from the provided configuration. This function overwrites the script.
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

    # Regex to find and replace the python command, including multi-line commands
    # This looks for a line starting with 'srun python' or just 'python', not commented out,
    # and includes any subsequent lines connected by a trailing '\'.
    python_cmd_pattern = re.compile(r"^(?!#)\s*(?:srun\s+)?python.*(?:\\\s*\n.*)*", re.MULTILINE)

    # Find the base command and arguments
    try:
        command_parts = shlex.split(command)
    except ValueError:
        # Handle cases where shlex can't split the command
        command_parts = command.split()

    base_cmd_parts = []
    args_parts = []
    is_arg = False
    for part in command_parts:
        if part.startswith("-"):
            is_arg = True
        if is_arg:
            args_parts.append(part)
        else:
            base_cmd_parts.append(part)

    base_cmd = " ".join(base_cmd_parts)

    if args_parts:
        # Group arguments with their values
        grouped_args = []
        current_arg = []
        for arg in args_parts:
            if arg.startswith("-") and current_arg:
                grouped_args.append(" ".join(current_arg))
                current_arg = [arg]
            else:
                current_arg.append(arg)
        if current_arg:
            grouped_args.append(" ".join(current_arg))

        multiline_args = " \\\n  ".join(grouped_args)
        new_command_str = f"srun {base_cmd} {multiline_args}".strip()
    else:
        new_command_str = f"srun {base_cmd}".strip()


    content, num_replacements = python_cmd_pattern.subn(new_command_str, content, count=1)
    if num_replacements == 0:
        # If no python command was found, append the new one.
        content += f"\n{new_command_str}\n"

    # Overwrite the original script
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(script_path, 0o750)

    return script_path


def submit_job(script_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(["sbatch", script_path], check=False, capture_output=True, text=True)

