from pathlib import Path
import pytest
# from pst_from_tests import setup_freyberg_mf6

pytest_plugins = ["modflow_devtools.fixtures"]

collect_ignore = [
    "utils_tests.py",
    "pst_tests.py",
    "pst_tests_2.py",
    "pst_from_tests.py",
    # "en_tests.py",
    "full_meal_deal_tests_2.py",
    # "la_tests.py",
    "plot_tests.py",
    "metrics_tests.py",
    "moouu_tests.py",
    "mat_tests.py",
    "da_tests.py",
    # "get_pestpp_tests.py"
]

@pytest.fixture(autouse=True)
def _ch2testdir(monkeypatch):
    testdir = Path(__file__).parent
    monkeypatch.chdir(testdir)