# Test Environment Setup Guide

## Why use this script?

The `setup_test_env.py` script automates the setup of a the testing environment for `pyemu`. It:

- Creates a conda environment with all required dependencies
- Downloads and configures PEST++ binaries for your operating system
- Sets up PATH variables so the binaries are automatically available when the environment is activated

This ensures you have a consistent, working environment for running pyemu tests and examples.

## Prerequisites

- **Conda or Mamba**: You need either `conda` or `mamba` installed on your system
  - The script will automatically detect and use `mamba` if available (faster)
  - Falls back to `conda` if `mamba` is not found

## How to run

1. **Navigate to the etc directory**:
   ```bash
   cd etc
   ```

2. **Run the setup script**:
   ```bash
   python setup_test_env.py
   ```
3. **Answer the promtps in the terminal/command line**

4. **Activate the new environment**:
   ```bash
   conda activate pyemu-issues
   ```
5. **Call `pestpp-ies` in the terminal/comand line to check**



## What happens during setup?

1. **Environment Creation**: Creates a conda environment named `pyemu` with all dependencies from `environment.yml`
2. **Binary Download**: Downloads PEST++ executables appropriate for your OS (Linux, macOS, or Windows) 
3. **PATH Configuration**: Sets up activation hooks so PEST++ binaries are available in your PATH when the environment is active

## Troubleshooting

- **"Neither 'mamba' nor 'conda' is installed"**: Install conda or mamba first
- **Permission errors**: Make sure you have write permissions in the current directory
- **Network issues**: The script downloads binaries from the internet, so ensure you have a stable connection

## After setup

Once complete, you can run pyemu tests and examples with all required dependencies and binaries available in your environment.