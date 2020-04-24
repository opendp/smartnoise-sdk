"""Setup file for differential privacy package."""
from setuptools import setup, find_namespace_packages
import os
import shutil

_major = "0.1"
_minor = "0.1"

VERSION = "{}.{}".format(_major, _minor)
SELFVERSION = VERSION
if os.path.exists("patch.version"):
    with open("patch.version", "rt") as bf:
        _patch = str(bf.read()).strip()
        SELFVERSION = "{}.{}".format(VERSION, _patch)

DEPENDENCIES = [
    "numpy",
    "pandas",
    "msrest",
    "scipy",
    "antlr4-python3-runtime==4.8",
    "pandasql",
    "statsmodels",
    "pyyaml"
]

EXTRAS = { }

with open("README.md", "r", encoding="utf-8") as f:
    README = f.read()
inline_license = ""

setup(
    name="opendp-whitenoise",

    version=SELFVERSION,

    description="",
    long_description=README,
    long_description_content_type="text/x-rst",
    author="opendp-whitenoise",
    license=inline_license,
    packages=find_namespace_packages(include=['opendp.*'],
                                     exclude=["*.tests"]),
    install_requires=DEPENDENCIES,

    include_package_data=True,

    extras_require=EXTRAS,

    data_files=[],
    zip_safe=False
)
