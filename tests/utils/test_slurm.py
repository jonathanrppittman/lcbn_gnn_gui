import os
import tempfile
from src.utils.slurm import update_slurm_script

def test_update_slurm_script_injection():
    # Create a temporary template
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write("#COMMAND_PLACEHOLDER\n#CONDA_ACTIVATION_PLACEHOLDER\n")
        template_path = f.name

    jobs_dir = tempfile.mkdtemp()

    slurm_cfg = {"conda_env": "malicious_env; echo 'hacked'"}
    command = "python script.py --arg 'malicious; rm -rf /'"
    conda_env = "malicious_env; echo 'hacked'"

    try:
        script_path = update_slurm_script(template_path, command, slurm_cfg, jobs_dir, conda_env)

        with open(script_path, "r") as f:
            content = f.read()

        # The variables should be quoted so they don't break out into their own commands
        assert "srun conda run -n 'malicious_env; echo '\"'\"'hacked'\"'\"'' python script.py --arg 'malicious; rm -rf /'" in content, content
        assert "conda activate 'malicious_env; echo '\"'\"'hacked'\"'\"''" in content, content

    finally:
        os.remove(template_path)
        for file in os.listdir(jobs_dir):
            os.remove(os.path.join(jobs_dir, file))
        os.rmdir(jobs_dir)
