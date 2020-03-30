import argparse
import os
import subprocess
import sys
from flask import jsonify
import mlflow

from connexion import FlaskApp as Flask
from exceptions import InvalidUsage

specification_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, specification_dir)

os.environ["GIT_PYTHON_REFRESH"] = "quiet"


# Create the application instance
app=Flask(__name__, specification_dir=specification_dir)

                        # Read the open api .yml file to configure the
                        # endpoints
                        # TODO allow for specifying multiple swagger files
app.add_api(os.path.join("openapi", "swagger.yml"))


port = int(os.environ.get("WHITENOISE_SERVICE_PORT", 5000))
os.environ["WHITENOISE_SERVICE_PORT"] = str(port)

if __name__ == "__main__":
    mlflow.set_tracking_uri(os.path.abspath("./mlruns"))
    @app.app.errorhandler(InvalidUsage)
    def handle_invalid_usage(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    app.run(host='0.0.0.0', port=port)

os.environ["WHITENOISE_SERVICE_URL"] = "http://whitenoise.azurewebsites.net"
# flask app
app = app.app

