import os
import subprocess
import sys

from threading import Thread



def get_git_root_dir():
    # Add the utils directory to the path
    return subprocess.check_output("git rev-parse --show-toplevel".split(" "))    .decode("utf-8").strip()

root_url = get_git_root_dir()


def run_app(name, spec_file, port, debug_mode=True, run_in_background=False):
    import connexion
    specification_dir = os.path.join(root_url, "service")
    sys.path.insert(0, specification_dir)

    # Create the application instance
    app = connexion.App(__name__, specification_dir=specification_dir)

    # Read the open api .yml file to configure the endpoints
    # TODO allow for specifying multiple swagger files
    app.add_api(os.path.join(root_url, "openapi", spec_file))

    app.run(host='0.0.0.0', port=port, debug=debug_mode)

    port = int(os.environ.get("BURDOCK_SERVICE_PORT", port))
    return app
