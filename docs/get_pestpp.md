# Install the PEST++ software suite

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Command-line interface](#command-line-interface)
  - [Using the `get-pestpp` command](#using-the-get-pestpp-command)
  - [Using `get_pestpp.py` as a script](#using-get_pestpppy-as-a-script)
- [pyEMU module](#pyemu-module)
- [Where to install?](#where-to-install)
- [Selecting a distribution](#selecting-a-distribution)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

pyEMU includes a `get-pestpp` utility to install the PEST++ software suite for Windows, Mac or Linux. If pyEMU is installed, the utility is available in the Python environment as a `get-pestpp` command. The script `pyEMU/utils/get_pestpp.py` has no dependencies and can be invoked independently.

The utility uses the [GitHub releases API](https://docs.github.com/en/rest/releases) to download versioned archives containing PEST++ executables. The utility is able to match the binary archive to the operating system and extract the console programs to a user-defined directory. A prompt can also be used to help the user choose where to install programs.

## Command-line interface

### Using the `get-pestpp` command

When pyEMU is installed, a `get-pestpp` (or `get-pestpp.exe` for Windows) program is installed, which is usually installed to the PATH (depending on the Python setup). From a console:

```console
$ get-pestpp --help
usage: get-pestpp [-h]
...
```

### Using `get_pestpp.py` as a script

The script requires Python 3.6 or later and does not have any dependencies, not even pyEMU. It can be downloaded separately and used the same as the console program, except with a different invocation. For example:

```console
$ wget https://raw.githubusercontent.com/modflowpy/pyEMU/develop/pyEMU/utils/get_pestpp.py
$ python3 get_pestpp.py --help
usage: get_pestpp.py [-h]
...
```

## pyEMU module

The same functionality of the command-line interface is available from the pyEMU module, as demonstrated below:

```python
from pathlib import Path
import pyemu

bindir = Path("/tmp/bin")
bindir.mkdir(exist_ok=True)
pymu.utils.get_pestpp(bindir)
list(bindir.iterdir())

# Or use an auto-select option
pyemu.utils.get_pestpp(":pyemu")
```

## Where to install?

A required `bindir` parameter must be supplied to the utility, which specifies where to install the programs. This can be any existing directory, usually which is on the users' PATH environment variable.

To assist the user, special values can be specified starting with the colon character. Use a single `:` to interactively select an option of paths.

Other auto-select options are only available if the current user can write files (some may require `sudo` for Linux or macOS):
 - `:prev` - if this utility was run by pyEMU more than once, the first option will be the previously used `bindir` path selection
 - `:pyemu` - special option that will create and install programs for pyEMU
 - `:python` - use Python's bin (or Scripts) directory
 - `:home` - use `$HOME/.local/bin`
 - `:system` - use `/usr/local/bin`
 - `:windowsapps` - use `%LOCALAPPDATA%\Microsoft\WindowsApps`

## Selecting a distribution

By default the distribution from the [`usgs/pestpp` repository](https://github.com/usgs/pestpp) is installed. The utility can also install from forks of the main [PEST++ 6 repo](https://github.com/usgs/pestpp) or other repo distributions. These repos include the following executables:

- `pestpp-da`
- `pestpp-glm`
- `pestpp-ies`
- `pestpp-mou`
- `pestpp-opt`
- `pestpp-sen`
- `pestpp-sqp`
- `pestpp-swp`

To select a distribution, specify a repository name with the `--repo` command line option or the `repo` function argument.

The repository owner can also be configured with the `--owner` option. This can be useful for installing from unreleased PEST++ software suite feature branches still in development &mdash; the only compatibility requirement is that release assets be named identically to those on the official repositories.
