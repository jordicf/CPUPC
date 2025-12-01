# `CPUPC` Development

`CPUPC` is both a Python library (the `cpupc` package and its subpackages) and a set of tools
accessible from the `cpupc` command-line utility.

## Installation for development

The following instructions will install both the `cpupc` Python package (and subpackages) and the
`cpupc` command-line utility.

First, install [Python 3](https://www.python.org/downloads/) and [Git](https://git-scm.com/download/).
Then, open a terminal and execute the following commands, depending on your operating system:

### Linux or macOS

```console
git clone https://github.com/jordicf/CPUPC.git
cd CPUPC
python -m venv venv
source venv/bin/activate
pip install -e '.[mypy,jupyter]'
```

### Windows

```console
git clone https://github.com/jordicf/CPUPC.git
cd CPUPC
python -m venv venv
.\venv\Scripts\activate
pip install -e '.[mypy,jupyter]'
```

### PyCharm configuration

If you use PyCharm, to configure the project Python interpreter go to File | Settings... |
Project: FRAME | Python Interpreter.
Then click the gears icon, Add..., and choose Existing environment and select Interpreter as the one
in the `CPUPC/venv` folder.

Mypy is also used to check the types. To integrate it with PyCharm,
install the following plugin:
[https://plugins.jetbrains.com/plugin/11086-mypy].

Code inspection can be executed in PyCharm going to Code | Inspect Code.... Creating a Custom Scope
including only the CPUPC code can be helpful to speed up the inspection and not get errors about
third-party code.

```console
file[CPUPC]:frame//*.py
file[CPUPC]:frame//*.ipynb
file[CPUPC]:tests//*.py
file[CPUPC]:tools//*.py
file[CPUPC]:tests//*.ipynb
file:setup.py
```

## Testing

To run all the tests, execute the following command from the project folder:

```console
python -m unittest discover -v -t . -s tests
```

## Extending

### Adding third-party dependencies

To add a third-party dependency, add the package name in the `dependencies` list of the `[project]`
section of the [`pyproject.toml` file](pyproject.toml) and re-execute `pip install -e '.[mypy,jupyter]'`
from the top-level project folder. Note that the package name should be the one that appears in the
[Python Package Index](https://pypi.org/).

### Adding a new subpackage

To add a new subpackage to the `cpupc` Python package, create a new directory inside the
[`cpupc` directory](cpupc). This new folder should contain an empty `__init__.py`
file, and all the Python code of the new subpackage. Then, add the subpackage name (prefixed with
`cpupc.`) in the `packages` list of the `[tool.setuptools]` of the
[`pyproject.toml` file](pyproject.toml). Finally, re-execute `pip install -e '.[mypy,jupyter]'` from the
top-level project folder.

To add unit tests for the new subpackage, create a new directory inside the
[`tests/frame` folder](tests/frame) with the name of the subpackage. This folder should
contain an empty `__init__.py` file too, and the scripts defining the unit tests using the
[`unittest` unit testing framework](https://docs.python.org/3/library/unittest.html).

### Adding a new tool

To add a new tool to the `cpupc` command-line utility, create a new directory inside the
[`tools` directory](tools). This new folder should contain an empty `__init__.py` file, and all the
code of the new tool. In particular, the main function of the tool should have the following
signature:

```python
main(prog: str | None = None, args: list[str] | None = None) -> None
```

where `prog` will be the name of the tool to be used in the command-line, and `args` is the list
of arguments passed to the tool. These arguments should be parsed using the
[`argparse` module](https://docs.python.org/3/library/argparse.html). Then, add the tool name
(prefixed with `tools.`) in the `packages` list of the `[tool.setuptools]` of the
[`pyproject.toml` file](pyproject.toml), and specify the tool name and the main function to call in
the `TOOLS` dictionary in [`tools/cpupc.py`](tools/frame.py). Finally, re-execute
`pip install -e '.[mypy,jupyter]'` from the top-level project folder.

To add unit tests for the new tool, create a new directory inside the
[`tests` folder](tests) with the name of the tool. This folder should
contain an empty `__init__.py` file too, and the scripts defining the unit tests using the
[`unittest` unit testing framework](https://docs.python.org/3/library/unittest.html).
