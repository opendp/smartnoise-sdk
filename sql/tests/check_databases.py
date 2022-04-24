import os
import subprocess
import sys

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
setup_path = os.path.abspath(
    os.path.join(
        git_root_dir, 
        "sql", 
        "tests", 
        "setup"
    )
)

sys.path.insert(0, setup_path)

from dataloader.db import DbCollection, download_data_files

download_data_files()
test_databases = DbCollection()
print(test_databases)