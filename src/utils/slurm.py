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

    # This regex finds the srun command, capturing the python part and its arguments
    srun_pattern = re.compile(r"srun\s+(python\s+\S+\.py)\s+(.*)", re.DOTALL)
    srun_match = srun_pattern.search(content)

    # This regex finds the python command if srun is not present.
    python_pattern = re.compile(r"^(?!#)\s*(python\s+\S+\.py)\s+(.*)", re.MULTILINE | re.DOTALL)
    python_match = python_pattern.search(content)

    base_command = ""
    existing_args_str = ""

    if srun_match:
        base_command = srun_match.group(1)
        existing_args_str = srun_match.group(2)
    elif python_match:
        # If no srun, but a python command is found, use it as the base.
        base_command = python_match.group(1)
        existing_args_str = python_match.group(2)

    # Parse existing arguments from the script
    # shlex.split is used to handle quoted arguments correctly.
    # The existing arguments are processed first.
    existing_args = shlex.split(existing_args_str.replace(' \\', ''))

    # The command from the GUI is authoritative, so its arguments will overwrite existing ones.
    # It's also split to handle complex arguments.
    gui_args = shlex.split(command)

    # The command from the GUI is authoritative. Find where the arguments start.
    if gui_args:
        arg_start_index = -1
        for i, arg in enumerate(gui_args):
            if arg.startswith('-'):
                arg_start_index = i
                break

        # If arguments are found, update the base command and the list of GUI arguments.
        if arg_start_index != -1:
            base_command = " ".join(gui_args[:arg_start_index])
            gui_args_list = gui_args[arg_start_index:]
        else:
            # If no arguments, the whole command is the base command.
            base_command = " ".join(gui_args)
            gui_args_list = []
    else:
        gui_args_list = []

    # A dictionary to hold the final set of arguments.
    # This will ensure that any argument from the GUI overwrites the script's default.
    args_dict = {}

    # Process existing arguments
    i = 0
    while i < len(existing_args):
        if existing_args[i].startswith('--'):
            key = existing_args[i]
            # Check if the next item is a value or another flag
            if i + 1 < len(existing_args) and not existing_args[i+1].startswith('--'):
                args_dict[key] = existing_args[i+1]
                i += 2
            else:
                args_dict[key] = None  # Flag without a value
                i += 1
        else:
            i += 1

    # Process GUI arguments, overwriting existing ones
    i = 0
    while i < len(gui_args_list):
        if gui_args_list[i].startswith('--'):
            key = gui_args_list[i]
            if i + 1 < len(gui_args_list) and not gui_args_list[i+1].startswith('--'):
                args_dict[key] = gui_args_list[i+1]
                i += 2
            else:
                args_dict[key] = None
                i += 1
        else:
            i += 1

    # Reconstruct the arguments string for the script
    # Each argument is on a new line for readability.
    args_list = []
    # List of arguments that should always be quoted
    quoted_args = ["--model", "--data", "--device", "--trip_net_num"]

    for key, value in args_dict.items():
        if value is not None:
            # Check if the key is in the list of args to be quoted, or if the value contains a space.
            if key in quoted_args or ' ' in str(value):
                # Using single quotes as requested by the user.
                args_list.append(f"{key} '{value}'")
            else:
                args_list.append(f'{key} {value}')
        else:
            # For flags without values
            args_list.append(key)

    # Joining with " \\n " ensures each argument is on a new line.
    multiline_args = " \\\n  ".join(args_list)

    # The new command to be placed in the script.
    new_command_str = f"srun {base_command} \\\n  {multiline_args}"

    # Replace the old command block with the new one
    lines = content.split('\n')
    start_pattern = re.compile(r"^(?!#)\s*(srun\s+)?python.*")

    start_index = -1
    for i, line in enumerate(lines):
        if start_pattern.match(line):
            start_index = i
            break

    if start_index != -1:
        # Found the start, now find the end of the block.
        # A block continues if the line is empty or indented.
        end_index = start_index
        for i in range(start_index + 1, len(lines)):
            line = lines[i]
            if line.strip() == '' or line.startswith(' ') or line.startswith('\t'):
                end_index = i
            else:
                # First non-indented, non-empty line marks the end.
                break

        # Reconstruct the content with the new command
        pre_block = lines[:start_index]
        post_block = lines[end_index+1:]
        new_lines = pre_block + [new_command_str] + post_block
        content = "\n".join(new_lines)
    else:
        # If no python command was found, append it to the end
        content += f"\n{new_command_str}\n"

    # Overwrite the original script
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(script_path, 0o750)

    return script_path


def submit_job(script_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(["sbatch", script_path], check=False, capture_output=True, text=True)

