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
    # We use python -c to simulate a failure and stderr output, instead of relying on shell operators like && and redirect
    command = "python3 -c \"import sys; print('An error occurred', file=sys.stderr); sys.exit(123)\""
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
            # If the process fails to start, the exception is yielded with exit code 1
            if line:
                output_lines.append(line)
            return_code = code

    # Without the shell, Popen itself will raise a FileNotFoundError
    # which is caught and yielded as a string in our executor.
    output_str = "".join(output_lines).lower()
    assert "no such file or directory" in output_str or "not found" in output_str
    # The exit code returned by our exception handler is 1
    assert return_code == 1