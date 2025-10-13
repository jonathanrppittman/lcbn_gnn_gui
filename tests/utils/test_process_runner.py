import pytest
import os
from src.utils.process_runner import ProcessExecutor

def test_executor_success():
    """Test that ProcessExecutor executes a command successfully and yields correct output."""
    command = 'echo "Success!"'
    executor = ProcessExecutor(command)

    output_lines = []
    return_code = -1

    for line, code in executor.run():
        if code == -1:
            output_lines.append(line)
        else:
            return_code = code

    assert "Success!" in "".join(output_lines)
    assert return_code == 0

def test_executor_failure():
    """Test that ProcessExecutor handles a failing command and captures the error code."""
    command = '>&2 echo "An error occurred" && exit 123'
    executor = ProcessExecutor(command)

    output_lines = []
    return_code = -1

    for line, code in executor.run():
        if code == -1:
            output_lines.append(line)
        else:
            return_code = code

    assert "An error occurred" in "".join(output_lines)
    assert return_code == 123

def test_executor_working_directory(tmp_path):
    """Test that the command is executed in the specified working directory."""
    test_file = tmp_path / "new_file.txt"
    command = f'touch "{test_file.name}"'

    assert not test_file.exists()

    executor = ProcessExecutor(command, working_dir=str(tmp_path))

    # Run the executor to completion
    list(executor.run())

    assert test_file.exists()

def test_executor_command_not_found():
    """Test that ProcessExecutor handles a command that doesn't exist."""
    command = 'this_command_does_not_exist_12345'
    executor = ProcessExecutor(command)

    output_lines = []
    return_code = -1

    for line, code in executor.run():
        if code == -1:
            output_lines.append(line)
        else:
            return_code = code

    # The shell will output an error message
    assert "command not found" in "".join(output_lines).lower()
    # The exit code from the shell for command not found is typically 127
    assert return_code == 127