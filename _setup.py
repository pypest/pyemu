import os
import sys

from setuptools import setup

sys.path.append(os.path.dirname(__file__))

import versioneer  # noqa: E402

# see pyproject.toml for static project metadata
setup(
    name="pyemu",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
