import json
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

from dataloader.factories.bigquery import BigQueryFactory


def test_load_service_account_info_from_inline_json():
    info = {"project_id": "demo-project", "type": "service_account"}

    assert BigQueryFactory._load_service_account_info(json.dumps(info)) == info


def test_load_service_account_info_from_file_path(tmp_path):
    info = {"project_id": "demo-project", "type": "service_account"}
    credentials_path = tmp_path / "gha-creds.json"
    credentials_path.write_text(json.dumps(info), encoding="utf-8")

    assert BigQueryFactory._load_service_account_info(str(credentials_path)) == info
