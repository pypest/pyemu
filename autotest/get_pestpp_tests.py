"""Test get-pestpp utility."""
import os
import platform
import sys
from os.path import expandvars
from pathlib import Path
from platform import system
from urllib.error import HTTPError

import pytest
from flaky import flaky
from modflow_devtools.markers import requires_github
from modflow_devtools.misc import run_py_script
from pyemu.utils import get_pestpp
from pyemu.utils.get_pestpp import get_release, get_releases, select_bindir

rate_limit_msg = "rate limit exceeded"
get_pestpp_script = (
    Path(__file__).parent.parent / "pyemu" / "utils" / "get_pestpp.py"
)
bindir_options = {
    "pyemu": Path(expandvars(r"%LOCALAPPDATA%\pyemu")) / "bin"
    if system() == "Windows"
    else Path.home() / ".local" / "share" / "pyemu" / "bin",
    "python": Path(sys.prefix)
    / ("Scripts" if system() == "Windows" else "bin"),
    "home": Path.home() / ".local" / "bin",
}
owner_options = [
    "usgs", "pestpp"
]
repo_options = {
    "pestpp": [
        "pestpp-da",
        "pestpp-glm",
        "pestpp-ies",
        "pestpp-mou",
        "pestpp-opt",
        "pestpp-sen",
        "pestpp-sqp",
        "pestpp-swp",
    ],
    "pestpp-nightly-builds": [
        "pestpp-da",
        "pestpp-glm",
        "pestpp-ies",
        "pestpp-mou",
        "pestpp-opt",
        "pestpp-sen",
        "pestpp-sqp",
        "pestpp-swp"
    ]
}

if system() == "Windows":
    bindir_options["windowsapps"] = Path(
        expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps")
    )
else:
    bindir_options["system"] = Path("/usr") / "local" / "bin"


@pytest.fixture
def downloads_dir(tmp_path_factory):
    downloads_dir = tmp_path_factory.mktemp("Downloads")
    return downloads_dir


@pytest.fixture(autouse=True)
def create_home_local_bin():
    # make sure $HOME/.local/bin exists for :home option
    home_local = Path.home() / ".local" / "bin"
    home_local.mkdir(parents=True, exist_ok=True)


def run_get_pestpp_script(*args):
    return run_py_script(get_pestpp_script, *args, verbose=True)


def append_ext(path: str):
    if system() == "Windows":
        return f"{path}{'.exe'}"
    elif system() == "Darwin":
        return f"{path}{''}"
    elif system() == "Linux":
        return f"{path}{''}"


@pytest.mark.parametrize("per_page", [-1, 0, 101, 1000])
def test_get_releases_bad_page_size(per_page):
    with pytest.raises(ValueError):
        get_releases(repo="pestpp", per_page=per_page)


@flaky
@requires_github
@pytest.mark.parametrize("repo", repo_options.keys())
def test_get_releases(repo):
    releases = get_releases(repo=repo)
    assert "latest" in releases


@flaky
@requires_github
@pytest.mark.parametrize("repo", repo_options.keys())
def test_get_release(repo):
    tag = "latest"
    release = get_release(repo=repo, tag=tag)
    assets = release["assets"]
    if len(release["body"]) > 0:
        # if nightly build tag is in body, use that
        release_tag_name = release["body"].split()[-1]
    else:
        release_tag_name = release["tag_name"]

    expected_assets = [
        f"pestpp-{release_tag_name}-linux",
        f"pestpp-{release_tag_name}-mac",
        f"pestpp-{release_tag_name}-win",
    ]
    actual_assets = [asset["name"].replace("tar.gz", "").replace(".zip", "") for asset in assets]

    for ostag in expected_assets:
        assert any(
            ostag in a for a in actual_assets
        ), f"dist not found for {ostag}"


@pytest.mark.parametrize("bindir", bindir_options.keys())
def test_select_bindir(bindir, function_tmpdir):
    expected_path = bindir_options[bindir]
    if not os.access(expected_path, os.W_OK):
        pytest.skip(f"{expected_path} is not writable")
    selected = select_bindir(f":{bindir}")

    if system() != "Darwin":
        assert selected == expected_path
    else:
        # for some reason sys.prefix can return different python
        # installs when invoked here and get_modflow.py on macOS
        #   https://github.com/modflowpy/flopy/actions/runs/3331965840/jobs/5512345032#step:8:1835
        #
        # work around by just comparing the end of the bin path
        # should be .../Python.framework/Versions/<version>/bin
        assert selected.parts[-4:] == expected_path.parts[-4:]


def test_script_help():
    assert get_pestpp_script.exists()
    stdout, stderr, returncode = run_get_pestpp_script("-h")
    assert "usage" in stdout
    assert len(stderr) == 0
    assert returncode == 0


@flaky
@requires_github
def test_script_invalid_options(function_tmpdir, downloads_dir):
    # try with bindir that doesn't exist
    bindir = function_tmpdir / "bin1"
    assert not bindir.exists()
    stdout, stderr, returncode = run_get_pestpp_script(bindir)
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert "does not exist" in stderr
    assert returncode == 1

    # attempt to fetch a non-existing release-id
    bindir.mkdir()
    assert bindir.exists()
    stdout, stderr, returncode = run_get_pestpp_script(
        bindir, "--release-id", "1.9", "--downloads-dir", downloads_dir
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert "Release 1.9 not found" in stderr
    assert returncode == 1

    # try to select an invalid --subset
    bindir = function_tmpdir / "bin2"
    bindir.mkdir()
    stdout, stderr, returncode = run_get_pestpp_script(
        bindir, "--subset", "pestpp-opt,mpx", "--downloads-dir", downloads_dir
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert "subset item not found: mpx" in stderr
    assert returncode == 1


@flaky
@requires_github
def test_script_valid_options(function_tmpdir, downloads_dir):
    # fetch latest
    bindir = function_tmpdir / "bin1"
    bindir.mkdir()
    stdout, stderr, returncode = run_get_pestpp_script(
        bindir, "--downloads-dir", downloads_dir
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert len(stderr) == returncode == 0
    files = [item.name for item in bindir.iterdir() if item.is_file()]
    assert len(files) == 8

    # valid subset
    bindir = function_tmpdir / "bin2"
    bindir.mkdir()
    stdout, stderr, returncode = run_get_pestpp_script(
        bindir,
        "--subset",
        "pestpp-da,pestpp-swp,pestpp-ies",
        "--downloads-dir",
        downloads_dir,
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert len(stderr) == returncode == 0
    files = [item.stem for item in bindir.iterdir() if item.is_file()]
    assert sorted(files) == ["pestpp-da", "pestpp-ies", "pestpp-swp"]

    # similar as before, but also specify a ostag
    bindir = function_tmpdir / "bin3"
    bindir.mkdir()
    stdout, stderr, returncode = run_get_pestpp_script(
        bindir,
        "--subset",
        "pestpp-ies",
        "--release-id",
        "5.2.6",
        "--ostag",
        "win",
        "--downloads-dir",
        downloads_dir,
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert len(stderr) == returncode == 0
    files = [item.name for item in bindir.iterdir() if item.is_file()]
    assert sorted(files) == ["pestpp-ies.exe"]


@flaky
@requires_github
@pytest.mark.parametrize("owner", owner_options)
@pytest.mark.parametrize("repo", repo_options.keys())
def test_script(request, function_tmpdir, owner, repo, downloads_dir):
    if ((repo == "pestpp-nightly-builds" and owner != "pestpp") or
            (owner == "pestpp" and repo != "pestpp-nightly-builds")):
        request.applymarker(pytest.mark.xfail)
    bindir = str(function_tmpdir)
    stdout, stderr, returncode = run_get_pestpp_script(
        bindir,
        "--owner",
        owner,
        "--repo",
        repo,
        "--downloads-dir",
        downloads_dir,
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    elif returncode != 0:
        raise RuntimeError(stderr)
    paths = list(function_tmpdir.glob("*"))
    names = [p.name for p in paths]
    expected_names = [append_ext(p) for p in repo_options[repo]]
    assert set(names) >= set(expected_names),'{0} vs {1}'.format(str(names),set(expected_names))


@flaky
@requires_github
@pytest.mark.parametrize("owner", owner_options)
@pytest.mark.parametrize("repo", repo_options.keys())
def test_python_api(request, function_tmpdir, owner, repo, downloads_dir):
    if ((repo == "pestpp-nightly-builds" and owner != "pestpp") or
            (owner == "pestpp" and repo != "pestpp-nightly-builds")):
        request.applymarker(pytest.mark.xfail)
    bindir = str(function_tmpdir)
    try:
        get_pestpp(bindir, owner=owner, repo=repo, downloads_dir=downloads_dir)
    except (HTTPError, IOError) as err:
        if '403' in str(err):
            pytest.skip(f"GitHub {rate_limit_msg}")
        else:
            raise err

    paths = list(function_tmpdir.glob("*"))
    names = [p.name for p in paths]
    expected_names = [append_ext(p) for p in repo_options[repo]]
    assert set(names) >= set(expected_names),'{0} vs {1}'.format(str(names),set(expected_names))
