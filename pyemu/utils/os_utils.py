"""Operating system utilities in the PEST(++) realm
"""
import os
import sys
import platform
import shutil
import subprocess as sp
import multiprocessing as mp
import warnings
import socket
import time
from datetime import datetime
from ..pyemu_warnings import PyemuWarning

ext = ""
bin_path = os.path.join("..", "bin")
if "linux" in platform.platform().lower():
    bin_path = os.path.join(bin_path, "linux")
elif "darwin" in platform.platform().lower():
    bin_path = os.path.join(bin_path, "mac")
else:
    bin_path = os.path.join(bin_path, "win")
    ext = ".exe"

bin_path = os.path.abspath(bin_path)
os.environ["PATH"] += os.pathsep + bin_path


def _istextfile(filename, blocksize=512):
    """
    Function found from:
    https://eli.thegreenplace.net/2011/10/19/perls-guess-if-file-is-text-or-binary-implemented-in-python
    Returns True if file is most likely a text file
    Returns False if file is most likely a binary file
    Uses heuristics to guess whether the given file is text or binary,
    by reading a single block of bytes from the file.
    If more than 30% of the chars in the block are non-text, or there
    are NUL ('\x00') bytes in the block, assume this is a binary file.
    """

    import sys

    PY3 = sys.version_info[0] == 3

    # A function that takes an integer in the 8-bit range and returns
    # a single-character byte object in py3 / a single-character string
    # in py2.
    #
    int2byte = (lambda x: bytes((x,))) if PY3 else chr

    _text_characters = b"".join(int2byte(i) for i in range(32, 127)) + b"\n\r\t\f\b"
    block = open(filename, "rb").read(blocksize)
    if b"\x00" in block:
        # Files with null bytes are binary
        return False
    elif not block:
        # An empty file is considered a valid text file
        return True

    # Use translate's 'deletechars' argument to efficiently remove all
    # occurrences of _text_characters from the block
    nontext = block.translate(None, _text_characters)
    return float(len(nontext)) / len(block) <= 0.30


def _remove_readonly(func, path, excinfo):
    """remove readonly dirs, apparently only a windows issue
    add to all rmtree calls: shutil.rmtree(**,onerror=remove_readonly), wk"""
    os.chmod(path, 128)  # stat.S_IWRITE==128==normal
    func(path)


def run(cmd_str, cwd=".", verbose=False):
    """an OS agnostic function to execute a command line

    Args:
        cmd_str (`str`): the str to execute with `os.system()`

        cwd (`str`, optional): the directory to execute the command in.
            Default is ".".
        verbose (`bool`, optional): flag to echo to stdout the  `cmd_str`.
            Default is `False`.

    Notes:
        uses `platform` to detect OS and adds .exe suffix or ./ prefix as appropriate
        if `os.system` returns non-zero, an exception is raised

    Example::

        pyemu.os_utils.run("pestpp-ies my.pst",cwd="template")

    """
    bwd = os.getcwd()
    os.chdir(cwd)
    try:
        exe_name = cmd_str.split()[0]
        if "window" in platform.platform().lower():
            if not exe_name.lower().endswith("exe"):
                raw = cmd_str.split()
                raw[0] = exe_name + ".exe"
                cmd_str = " ".join(raw)
        else:
            if exe_name.lower().endswith("exe"):
                raw = cmd_str.split()
                exe_name = exe_name.replace(".exe", "")
                raw[0] = exe_name
                cmd_str = "{0} {1} ".format(*raw)
            if os.path.exists(exe_name) and not exe_name.startswith("./"):
                cmd_str = "./" + cmd_str

    except Exception as e:
        os.chdir(bwd)
        raise Exception("run() error preprocessing command line :{0}".format(str(e)))
    if verbose:
        print("run():{0}".format(cmd_str))

    try:
        ret_val = os.system(cmd_str)
    except Exception as e:
        os.chdir(bwd)
        raise Exception("run() raised :{0}".format(str(e)))
    os.chdir(bwd)

    if "window" in platform.platform().lower():
        if ret_val != 0:
            raise Exception("run() returned non-zero: {0}".format(ret_val))
    else:
        estat = os.WEXITSTATUS(ret_val)
        if estat != 0:
            raise Exception("run() returned non-zero: {0}".format(estat))


def _try_remove_existing(d, forgive=False):
    try:
        shutil.rmtree(d, onerror=_remove_readonly)  # , onerror=del_rw)
        return True
    except Exception as e:
        if not forgive:
            raise Exception(
                f"unable to remove existing dir: {d}\n{e}"
            )
        else:
            warnings.warn(
                f"unable to remove worker dir: {d}\n{e}",
                PyemuWarning,
            )
        return False


def _try_copy_dir(o_d, n_d):
    try:
        shutil.copytree(o_d, n_d)
    except PermissionError:
        time.sleep(3) # pause for windows locking issues
        try:
            shutil.copytree(o_d, n_d)
        except Exception as e:
            raise Exception(
                f"unable to copy files from base dir: "
                f"{o_d}, to new dir: {n_d}\n{e}"
            )


def start_workers(
    worker_dir,
    exe_rel_path,
    pst_rel_path,
    num_workers=None,
    worker_root="..",
    port=4004,
    rel_path=None,
    local=True,
    cleanup=True,
    master_dir=None,
    verbose=False,
    silent_master=False,
    reuse_master=False,
):
    """start a group of pest(++) workers on the local machine

    Args:
        worker_dir (`str`): the path to a complete set of input files need by PEST(++).
            This directory will be copied to make worker (and optionally the master)
            directories
        exe_rel_path (`str`): the relative path to and name of the pest(++) executable from within
            the `worker_dir`.  For example, if the executable is up one directory from
            `worker_dir`, the `exe_rel_path` would be `os.path.join("..","pestpp-ies")`
        pst_rel_path (`str`): the relative path to and name of the pest control file from within
            `worker_dir`.
        num_workers (`int`, optional): number of workers to start. defaults to number of cores
        worker_root (`str`, optional):  the root directory to make the new worker directories in.
            Default is ".."  (up one directory from where python is running).
        rel_path (`str`, optional): the relative path to where pest(++) should be run
            from within the worker_dir, defaults to the uppermost level of the worker dir.
            This option is usually not needed unless you are one of those crazy people who
            spreads files across countless subdirectories.
        local (`bool`, optional): flag for using "localhost" instead of actual hostname/IP address on
            worker command line. Default is True.  `local` can also be passed as an `str`, in which
            case `local` is used as the hostname (for example `local="192.168.10.1"`)
        cleanup (`bool`, optional):  flag to remove worker directories once processes exit. Default is
            True.  Set to False for debugging issues
        master_dir (`str`): name of directory for master instance.  If `master_dir`
            exists, then it will be REMOVED!!!  If `master_dir`, is None,
            no master instance will be started.  If not None, a copy of `worker_dir` will be
            made into `master_dir` and the PEST(++) executable will be started in master mode
            in this directory. Default is None
        verbose (`bool`, optional): flag to echo useful information to stdout.  Default is False
        silent_master (`bool`, optional): flag to pipe master output to devnull and instead print
            a simple message to stdout every few seconds.  This is only for
            pestpp Travis testing so that log file sizes dont explode. Default is False
        reuse_master (`bool`): flag to use an existing `master_dir` as is - this is an advanced user
            option for cases where you want to construct your own `master_dir` then have an async
            process started in it by this function.

    Notes:
        If all workers (and optionally master) exit gracefully, then the worker
        dirs will be removed unless `cleanup` is False

    Example::

        # start 10 workers using the directory "template" as the base case and
        # also start a master instance in a directory "master".
        pyemu.helpers.start_workers("template","pestpp-ies","pest.pst",10,master_dir="master",
                                    worker_root=".")

    """

    if not os.path.isdir(worker_dir):
        raise Exception("worker dir '{0}' not found".format(worker_dir))
    if not os.path.isdir(worker_root):
        raise Exception("worker root dir not found")
    if num_workers is None:
        num_workers = mp.cpu_count()
    else:
        num_workers = int(num_workers)
    # assert os.path.exists(os.path.join(worker_dir,rel_path,exe_rel_path))
    exe_verf = True

    if rel_path:
        if not os.path.exists(os.path.join(worker_dir, rel_path, exe_rel_path)):
            # print("warning: exe_rel_path not verified...hopefully exe is in the PATH var")
            exe_verf = False
    else:
        if not os.path.exists(os.path.join(worker_dir, exe_rel_path)):
            # print("warning: exe_rel_path not verified...hopefully exe is in the PATH var")
            exe_verf = False
    if rel_path is not None:
        if not os.path.exists(os.path.join(worker_dir, rel_path, pst_rel_path)):
            raise Exception("pst_rel_path not found from worker_dir using rel_path")
    else:
        if not os.path.exists(os.path.join(worker_dir, pst_rel_path)):
            raise Exception("pst_rel_path not found from worker_dir")
    if isinstance(local, str):
        hostname = local
    elif local:
        hostname = "localhost"
    else:
        hostname = socket.gethostname()

    base_dir = os.getcwd()
    port = int(port)

    if os.path.exists(os.path.join(worker_dir, exe_rel_path)):
        if "window" in platform.platform().lower():
            if not exe_rel_path.lower().endswith("exe"):
                exe_rel_path = exe_rel_path + ".exe"
        else:
            if not exe_rel_path.startswith("./"):
                exe_rel_path = "./" + exe_rel_path

    if master_dir is not None:
        if master_dir != "." and os.path.exists(master_dir) and not reuse_master:
            _try_remove_existing(master_dir)
        if master_dir != "." and not reuse_master:
            _try_copy_dir(worker_dir, master_dir)

        args = [exe_rel_path, pst_rel_path, "/h", ":{0}".format(port)]
        if rel_path is not None:
            cwd = os.path.join(master_dir, rel_path)
        else:
            cwd = master_dir
        if verbose:
            print("master:{0} in {1}".format(" ".join(args), cwd))
        stdout = None
        if silent_master:
            stdout = open(os.devnull, "w")
        try:
            os.chdir(cwd)
            master_p = sp.Popen(args, stdout=stdout)  # ,stdout=sp.PIPE,stderr=sp.PIPE)
            os.chdir(base_dir)
        except Exception as e:
            raise Exception("error starting master instance: {0}".format(str(e)))
        time.sleep(1.5)  # a few cycles to let the master get ready

    tcp_arg = "{0}:{1}".format(hostname, port)
    procs = []
    worker_dirs = []
    for i in range(num_workers):
        new_worker_dir = os.path.join(worker_root, "worker_{0}".format(i))
        if os.path.exists(new_worker_dir):
            _try_remove_existing(new_worker_dir)
        _try_copy_dir(worker_dir, new_worker_dir)
        try:
            if exe_verf:
                # if rel_path is not None:
                #     exe_path = os.path.join(rel_path,exe_rel_path)
                # else:
                exe_path = exe_rel_path
            else:
                exe_path = exe_rel_path
            args = [exe_path, pst_rel_path, "/h", tcp_arg]
            # print("starting worker in {0} with args: {1}".format(new_worker_dir,args))
            if rel_path is not None:
                cwd = os.path.join(new_worker_dir, rel_path)
            else:
                cwd = new_worker_dir

            os.chdir(cwd)
            if verbose:
                print("worker:{0} in {1}".format(" ".join(args), cwd))
            with open(os.devnull, "w") as f:
                p = sp.Popen(args, stdout=f, stderr=f)
            procs.append(p)
            os.chdir(base_dir)
        except Exception as e:
            raise Exception("error starting worker: {0}".format(str(e)))
        worker_dirs.append(new_worker_dir)

    if master_dir is not None:
        # while True:
        #     line = master_p.stdout.readline()
        #     if line != '':
        #         print(str(line.strip())+'\r',end='')
        #     if master_p.poll() is not None:
        #         print(master_p.stdout.readlines())
        #         break
        if silent_master:
            # this keeps travis from thinking something is wrong...
            while True:
                rv = master_p.poll()
                if master_p.poll() is not None:
                    break
                print(datetime.now(), "still running")
                time.sleep(5)
        else:
            master_p.wait()
            time.sleep(1.5)  # a few cycles to let the workers end gracefully

        # kill any remaining workers
        for p in procs:
            p.kill()
    # this waits for sweep to finish, but pre/post/model (sub)subprocs may take longer
    for p in procs:
        p.wait()
    if cleanup:
        cleanit = 0
        removed = set()
        while len(removed) < len(worker_dirs):  # arbitrary 100000 limit
            cleanit = cleanit + 1
            for d in worker_dirs:
                if os.path.exists(d):
                    success = _try_remove_existing(d, forgive=True)
                    if success:
                        removed.update(d)
                else:
                    removed.update(d)
            if cleanit > 100:
                break


    if master_dir is not None:
        ret_val = master_p.returncode
        if ret_val != 0:
            raise Exception("start_workers() master returned non-zero: {0}".format(ret_val))
