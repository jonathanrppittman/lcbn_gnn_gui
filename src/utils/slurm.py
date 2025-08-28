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

    # Now, find and remove the old command block, then append the new one.
    final_lines: List[str] = []
    in_command_block = False
    for line in new_lines:
        stripped = line.strip()

        # Detect start of the command block
        if not in_command_block and not stripped.startswith("#") and ('srun' in stripped or 'python' in stripped):
            in_command_block = True

        # If we are in the command block, we skip the line.
        # We also check if the block ends here.
        if in_command_block:
            if not stripped.endswith("\\"):
                in_command_block = False  # End of the block
            continue  # Skip the line

        # Only append lines that are not part of the old command block
        final_lines.append(line)

    # Append the new command at the end of the script
    # Ensure there's a newline before it if the script isn't empty
    if final_lines and not final_lines[-1].endswith('\n'):
        final_lines.append('\n')
    final_lines.append(f"srun {command}\n")

    content = "".join(final_lines)

    # Overwrite the original script
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(script_path, 0o750)

    return script_path


def submit_job(script_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(["sbatch", script_path], check=False, capture_output=True, text=True)

