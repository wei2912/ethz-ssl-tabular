# ethz-ssl-tabular

## Installation

This project requires [the installation of Poetry](https://python-poetry.org/docs/#installation), which is used to manage dependencies.

Afterwards, run the following commands:
```bash
$ poetry shell # enter the virtualenv created by Poetry
$ poetry install
$ poe force-cuda118 # install PyTorch w/ Cuda 11.8
```

## `pre-commit` Setup

To enable the use of Git [pre-commit](https://pre-commit.com/) hooks, run the following command:

$ pre-commit install
