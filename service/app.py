import argparse
import os
import subprocess
import sys

# Add the utils directory to the path
root_url = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
sys.path.append(os.path.join(root_url, "utils"))

from service_utils import run_app  # NOQA


if __name__ == "__main__":
    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=str, default="swagger.yml",
                        help="the service api specification")
    parser.add_argument("--port", type=int, default=5000, help="the port number for the service")
    parser.add_argument("--debug", type=bool, default=True, help="whether to run in debug mode")
    args = parser.parse_args()

    run_app(__name__, spec_file=args.spec, port=args.port, debug_mode=args.debug)
