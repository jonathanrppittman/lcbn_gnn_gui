import os
import shlex
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

    # Process SBATCH directives
    new_lines: List[str] = []
    for line in template_lines:
        stripped_line = line.strip()
        if stripped_line.startswith("#SBATCH"):
            found_match = False
            for cfg_key, sbatch_key in sbatch_map.items():
                pattern = re.compile(rf"(#SBATCH\s+{re.escape(sbatch_key)})(?:[=\s])(.*)")
                match = pattern.match(stripped_line)
                if match:
                    if cfg_key in slurm_cfg and slurm_cfg[cfg_key]:
                        new_value = slurm_cfg[cfg_key]
                        new_lines.append(f"{match.group(1)} {new_value}\n")
                        found_match = True
                        break
            if not found_match:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Build the srun command
    parts = ["srun"] + shlex.split(command)
    multiline_command = " \\\n    ".join(parts) + "\n"

    # Find the command placeholder and replace it. If not found, append.
    final_lines = []
    command_inserted = False
    placeholder = "# <<< COMMAND HERE >>>"
    for line in new_lines:
        if placeholder in line:
            final_lines.append(multiline_command)
            command_inserted = True
        else:
            final_lines.append(line)

    # If the placeholder was not found, fall back to the old method of appending.
    # This ensures backward compatibility with the conversion script.
    if not command_inserted:
        # First, remove any existing command block to prevent duplication
        cleaned_lines = []
        in_command_block = False
        for line in final_lines:
            stripped = line.strip()
            # A simple heuristic to detect a command block to remove
            is_command_line = not stripped.startswith("#") and ('srun' in stripped or 'python' in stripped)

            if not in_command_block and is_command_line:
                in_command_block = True

            if in_command_block:
                if not stripped.endswith("\\"):
                    in_command_block = False # End of block
                continue # Skip line

            cleaned_lines.append(line)

        final_lines = cleaned_lines
        if final_lines and not final_lines[-1].endswith('\n'):
            final_lines.append('\n')
        final_lines.append(multiline_command)

    content = "".join(final_lines)

    # Overwrite the original script
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(script_path, 0o750)

    return script_path


def submit_job(script_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(["sbatch", script_path], check=False, capture_output=True, text=True)

