from pathlib import Path
import pytest
import shutil
import platform
# from pst_from_tests import setup_freyberg_mf6

pytest_plugins = ["modflow_devtools.fixtures"]

collect_ignore = [
    # "emulator_tests.py",
    # "en_tests.py",
    # "get_pestpp_tests.py",
    # "la_tests.py",
    # "mat_tests.py",
    # "mc_tests_ignore.py",
    # "metrics_tests.py",
    # "plot_tests.py",
    # "pst_from_tests.py",
    # "pst_tests.py",
    # "transformer_tests.py",
    # "utils_tests.py",
    # "verf_test.py",
]

def get_project_root_path():
    """
    Get the root path of the project.
    """
    return Path(__file__).parent.parent


def get_exe_path(exe_name, forgive=True):
    """
    Get the absolute path to an executable in the project.
    """
    if platform.system() == "Windows":
        exe_name = f"{exe_name}.exe"
    if shutil.which(exe_name) is not None:
        print(f"Found {exe_name} in system PATH")
        return exe_name
    # else look in local project bin/<platform>
    root_path = get_project_root_path()
    exe_path = root_path / "bin"
    if not (exe_path / exe_name).exists():
        if "linux" in platform.system().lower():
            exe_path = Path(exe_path, "linux")
        elif "darwin" in platform.system().lower():
            exe_path = Path(exe_path, "mac")
        else:
            exe_path = Path(exe_path, "win")
    # if is isn't in bin/<platform> either, give up
    if not (exe_path / exe_name).exists():
        if forgive:
            print(f"Executable {exe_name} not found in {exe_path}, returning None")
        else:
            raise FileNotFoundError(f"Executable {exe_name} not found in system PATH or fallback path:"
                                    f" {exe_path}")
        return None
    return (exe_path / exe_name).as_posix()


def full_exe_ref_dict():
    """
    Get a dictionary of executable references for the project.
    """
    d = {}
    for exe_name in [
        "mfnwt", "mfusg_gsi", "mf6",
        "pestpp-ies", "pestpp-sen", "pestpp-opt", "pestpp-glm",
        "pestpp-mou", "pestpp-da", "pestpp-sqp", "pestpp-swp"
    ]:
        exe_path = get_exe_path(exe_name)
        d[exe_name] = exe_path
    return d


@pytest.fixture(autouse=True)
def _ch2testdir(monkeypatch):
    testdir = Path(__file__).parent
    monkeypatch.chdir(testdir)

@pytest.fixture(autouse=True)
def _use_plt_agg_backend():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        plt.switch_backend("Agg")