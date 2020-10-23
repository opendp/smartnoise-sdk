import argparse
import os
import subprocess
import sys

from connexion import FlaskApp as Flask

specification_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, specification_dir)


# Create the application instance
app=Flask(__name__, specification_dir=specification_dir)

                        # Read the open api .yml file to configure the
                        # endpoints
                        # TODO allow for specifying multiple swagger files
app.add_api(os.path.join("openapi", "swagger.yml"))


port = int(os.environ.get("SMARTNOISE_SERVICE_PORT", 5000))
os.environ["SMARTNOISE_SERVICE_PORT"] = str(port)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)

# flask app
app = app.app
