[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![CI](https://github.com/opendp/opendp-documentation/actions/workflows/main.yml/badge.svg)

# SmartNoise Documentation

This folder contains the source for building the top-level SmartNoise documentation, [docs.smartnoise.org](https://docs.smartnoise.org), landing page and general information about deploying differential privacy.  Detailed documentation about the SQL and Synthesizer functionality is built from the `docs` folders in their respective project folders.

If you are extending or altering the SDK, you should update documentation as appropriate.  To quickly run a test build of the documentation, you can run `source make_docs.sh` from this folder.

## Building the Docs

The steps below assume the use of [Homebrew] on a Mac.

[Homebrew]: https://brew.sh

```shell
pip install -r requirements.txt
make html
open build/html/index.html
```

## Deployment

Docs are deployed to http://docs.smartnoise.org using GitHub Actions.


## Join the Discussion

You are very welcome to join us on [GitHub Discussions][]!

[GitHub Discussions]: https://github.com/opendp/opendp/discussions
