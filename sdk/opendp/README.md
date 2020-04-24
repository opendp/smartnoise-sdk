# Packaging note

- Please do not put an `__init__.py` file in this directory--the `opendp' directory
- The "opendp" namespace is shared with another package generated from the [whitenoise-core](https://github.com/opendifferentialprivacy/whitenoise-core) repository.
- For reference, see this python packaging document:
  - https://packaging.python.org/guides/packaging-namespace-packages/#native-namespace-packages
    - Note the line: `    # No __init__.py here.`
