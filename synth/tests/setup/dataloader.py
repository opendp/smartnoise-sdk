import os
import subprocess
import sys
from dataloader.db import download_data_files


git_root_dir = (
    subprocess.check_output("git rev-parse --show-toplevel".split(" "))
    .decode("utf-8")
    .strip()
)
setup_path = os.path.abspath(os.path.join(git_root_dir, "sql", "tests", "setup"))
sys.path.insert(0, setup_path)


download_data_files()
