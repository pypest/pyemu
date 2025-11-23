#!/usr/bin/env python3
import os
import platform
import subprocess
from pathlib import Path
import shutil


bin_path = os.path.join("..", "bin")
assert os.path.isdir(bin_path), f"Bin path {bin_path} does not exist"
if "linux" in platform.system().lower():
    bin_path = os.path.join(bin_path, "linux")
elif "darwin" in platform.system().lower():
    bin_path = os.path.join(bin_path, "mac")
else:
    bin_path = os.path.join(bin_path, "win")

# --- CONFIG ---
ENV_FILE = "environment.yml"   # path to your environment.yml
BIN_DIR = bin_path  # directory to add to PATH (use Windows format on Windows)
# ----------------


# Detect available executable
if shutil.which("mamba"):
    manager = "mamba"
elif shutil.which("conda"):
    manager = "conda"
else:
    raise RuntimeError("Neither 'mamba' nor 'conda' is installed.")
print(f">>> Using {manager} as package manager.")


# Load environment name from environment.yml
ENV_NAME = None
with open(ENV_FILE) as f:
    for line in f:
        line = line.strip()
        if line.startswith("name:"):
            ENV_NAME = line.split(":", 1)[1].strip()
            break
if not ENV_NAME:
    raise ValueError("Could not find 'name:' in environment.yml")
# Create environment
print(f">>> Creating {manager} environment: {ENV_NAME}")
subprocess.run([manager, "env", "create", "-f", ENV_FILE], check=True)

# Determine environment path
base_path = subprocess.check_output([manager, "info", "--base"], text=True).split(":")[-1].strip()
base_path = Path(base_path).resolve()
env_path = (base_path / "envs" / ENV_NAME).resolve(strict=True)

# Directories for activation hooks
activate_d = Path(env_path / "etc" / "conda" / "activate.d").resolve(strict=True)
deactivate_d = Path(env_path / "etc" / "conda" / "deactivate.d").resolve(strict=True)
activate_d.mkdir(parents=True, exist_ok=True)
deactivate_d.mkdir(parents=True, exist_ok=True)

system = platform.system()
if system == "Windows":
    # Windows: .bat scripts
    activate_file = activate_d / "env_path.bat"
    deactivate_file = deactivate_d / "env_path.bat"

    activate_file.write_text(f"""@echo off
set "PATH={BIN_DIR};%PATH%"
""")
    deactivate_file.write_text(f"""@echo off
set "PATH=%PATH:{BIN_DIR};=%"
""")
else:
    # macOS / Linux: .sh scripts
    activate_file = activate_d / "env_path.sh"
    deactivate_file = deactivate_d / "env_path.sh"

    activate_file.write_text(f"""#!/bin/bash
export PATH="{BIN_DIR}:$PATH"
""")
    deactivate_file.write_text(f"""#!/bin/bash
export PATH="${{PATH//{BIN_DIR}:/}}"
""")
    # Make sure scripts are executable
    activate_file.chmod(0o755)
    deactivate_file.chmod(0o755)

print(f">>> Environment {ENV_NAME} created successfully.")
print(f">>> PATH will include {BIN_DIR} when you activate this environment.")

# download the pestpp binaries
subprocess.run(["../pyemu/utils/get_pestpp.py", BIN_DIR], check=True)

print(f">>> PEST++ binaries downloaded to {BIN_DIR}.")
print(">>> Setup complete.")