"""Operating system utilities in the PEST(++) realm
"""
import os
import sys
import platform
import time
import struct
import shutil
import threading
import subprocess as sp
import multiprocessing as mp
import warnings
import socket
import time
from datetime import datetime
import random
import logging
import tempfile
from contextlib import contextmanager
import json
import uuid
import concurrent.futures

import numpy as np
import pandas as pd

from ..pyemu_warnings import PyemuWarning
from ..pst import pst_handler
from ..logger import Logger

ext = ""
bin_path = os.path.join("..", "bin")
if "linux" in platform.system().lower():
    bin_path = os.path.join(bin_path, "linux")
elif "darwin" in platform.system().lower():
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

def run(cmd_str, cwd=".", verbose=False, use_sp = False, **kwargs):
    """main run function so both run_sp and run_ossystem can coexist

    Args:
        cmd_str (`str`): the str to execute with `os.system()`

        cwd (`str`, optional): the directory to execute the command in.
            Default is ".".
        verbose (`bool`, optional): flag to echo to stdout the  `cmd_str`.
            Default is `False`.
    Notes:
        by default calls run_ossystem which is the OG function from pyemu that uses `os.system()`
        if use_sp is True, then run_sp is called which uses `subprocess.Popen` instead of `os.system`

    Example::
        pyemu.os_utils.run("pestpp-ies my.pst",cwd="template")
    """

    if use_sp:
        run_sp(cmd_str, cwd, verbose, **kwargs)
    else:       
        run_ossystem(cmd_str, cwd, verbose)

def run_ossystem(cmd_str, cwd=".", verbose=False):
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
            if (os.path.exists(exe_name)
                    and not exe_name.startswith("./")
                    and not exe_name.startswith("/")):
                cmd_str = "./" + cmd_str

    except Exception as e:
        os.chdir(bwd)
        raise Exception("run() error preprocessing command line :{0}".format(str(e)))
    if verbose:
        print("run():{0}".format(cmd_str))

    try:
        print(cmd_str)
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
        if estat != 0 or ret_val != 0:
            raise Exception("run() returned non-zero: {0},{1}".format(estat,ret_val))
        
def run_sp(cmd_str, cwd=".", verbose=True, logfile=False, **kwargs):
    """an OS agnostic function to execute a command line with subprocess

    Args:
        cmd_str (`str`): the str to execute with `sp.Popen()`

        cwd (`str`, optional): the directory to execute the command in.
            Default is ".".
        verbose (`bool`, optional): flag to echo to stdout the  `cmd_str`.
            Default is `False`.
        shell (`bool`, optional): flag to use shell=True in the `subprocess.Popen` call. Not recommended

    Notes:
        uses sp Popen to execute the command line. By default does not run in shell mode (ie. does not look for the exe in env variables)

    """
    # update shell from  kwargs
    shell = kwargs.get("shell", False)
    # detached = kwargs.get("detached", False)

    # print warning if shell is True
    if shell:
        warnings.warn("shell=True is not recommended and may cause issues, but hey! YOLO", PyemuWarning)


    bwd = os.getcwd()
    os.chdir(cwd)
    exe_name = cmd_str.split()[0]
    if (platform.system() != "Windows"
            and not shutil.which(exe_name)
            and not exe_name.startswith("./")
            and not exe_name.startswith("/")):
        cmd_str = "./" + cmd_str

    try:
        cmd_ins = [i for i in cmd_str.split()]
        log_stream = open(os.path.join('pyemu.log'), 'w+', newline='') if logfile else None
        with sp.Popen(cmd_ins, stdout=sp.PIPE, 
                      stderr=sp.STDOUT, text=True,
                      shell=shell, bufsize=1) as process:
            for line in process.stdout:
                if verbose:
                    print(line, flush=True, end='')
                if logfile:
                    log_stream.write(line.strip('\n'))
                    log_stream.flush()
            process.wait() # wait for the process to finish
            retval = process.returncode

    except Exception as e:
        os.chdir(bwd)
        raise Exception("run() raised :{0}".format(str(e)))

    finally:
        if logfile:
            log_stream.close()
    os.chdir(bwd)

    if "window" in platform.platform().lower():
        if retval != 0:
            raise Exception("run() returned non-zero: {0}".format(retval))
    else:
        estat = os.WEXITSTATUS(retval)
        if estat != 0 or retval != 0:
            raise Exception("run() returned non-zero: {0},{1}".format(estat, retval))        
    return retval


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
        shutil.copytree(o_d, n_d, symlinks=True)
    except PermissionError:
        time.sleep(3) # pause for windows locking issues
        try:
            shutil.copytree(o_d, n_d, symlinks=True)
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
    port=None,
    rel_path=None,
    local=True,
    cleanup=True,
    master_dir=None,
    verbose=False,
    silent_master=False,
    reuse_master=False,
    restart=False,
    ppw_function=None,
    ppw_kwargs={}
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
        restart (`bool`): flag to add a restart flag to the master start. If `True`, this will include
            `/r` in the master call string.
        ppw_function (function): a function pointer that uses PyPestWorker to execute model runs.  
            This is to help speed up pure-python and really fast running forward models and the
            way its implemented in `start_workers` assumes each `PyPestWorker` instance does not need
            a separate set of model+files, that is, these workers are assumed to execute entirely in 
            memory.  The first three arguments to this function must be `pst_rel_path`, `host`, and `port`.
            Default is None.
        ppw_kwargs (dict): keyword arguments to pass to `ppw_function`.
            Default is empty dict.

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

    if port is None:
        if master_dir is None:
            port = 4004 # the 'ole standard
        else:
            port = PortManager().get_available_port()

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
            if (not exe_rel_path.startswith("./")
                    and not exe_rel_path.startswith("/")):
                exe_rel_path = "./" + exe_rel_path

    if master_dir is not None:
        if master_dir != "." and os.path.exists(master_dir) and not reuse_master:
            _try_remove_existing(master_dir)
        if master_dir != "." and not reuse_master:
            _try_copy_dir(worker_dir, master_dir)
        
        args = [exe_rel_path, pst_rel_path, "/h", ":{0}".format(port)]
        if restart is True:
            # add restart if requested
            args = [exe_rel_path, pst_rel_path, "/h", "/r", ":{0}".format(port)]
        
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
        time.sleep(0.5)  # a few cycles to let the master get ready


    procs = []
    worker_dirs = []
    if ppw_function is not None:
        args = (os.path.join(worker_dir, pst_rel_path), hostname, port)
        
        # Create processes in batches using ThreadPoolExecutor for faster deployment
        def create_and_start_worker(worker_id):
            p = mp.Process(target=ppw_function, args=args, kwargs=ppw_kwargs)
            p.daemon = True
            p.start()
            if verbose:
                print("Started worker {0} (PID: {1})".format(worker_id, p.pid))
            return p
        
        # Use ThreadPoolExecutor to create processes in parallel
        max_concurrent_starts = num_workers#min(num_workers, 300)  # Limit concurrent starts
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_starts) as executor:
            # Submit all worker creation tasks
            future_to_id = {
                executor.submit(create_and_start_worker, i): i 
                for i in range(num_workers)
            }
            
            # Collect started processes
            for future in concurrent.futures.as_completed(future_to_id):
                worker_id = future_to_id[future]
                try:
                    p = future.result()
                    procs.append(p)
                except Exception as e:
                    print("Error starting worker {0}: {1}".format(worker_id, e))
                    raise

    else:
        tcp_arg = "{0}:{1}".format(hostname, port)
        
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
            time.sleep(0.5)  # a few cycles to let the workers end gracefully

        # More efficient graceful termination for ppw_function workers
        if ppw_function is not None:
            termination_timeout = 10  # seconds
            
            if verbose:
                print("Initiating graceful shutdown of {0} workers...".format(len(procs)))
            
            # First, try graceful termination
            for p in procs:
                if p.is_alive():
                    try:
                        p.terminate()  # Send SIGTERM
                    except:
                        pass
            
            # Wait for processes to terminate gracefully
            start_time = time.time()
            remaining_procs = procs[:]  # Create a copy
            
            while remaining_procs and (time.time() - start_time) < termination_timeout:
                still_alive = []
                for p in remaining_procs:
                    if p.is_alive():
                        still_alive.append(p)
                    else:
                        try:
                            p.join(timeout=0.1)  # Clean up zombie processes
                        except:
                            pass
                
                remaining_procs = still_alive
                if remaining_procs:
                    time.sleep(0.1)
            
            # Force kill any remaining processes
            if remaining_procs:
                if verbose:
                    print("Force killing {0} remaining workers...".format(len(remaining_procs)))
                for p in remaining_procs:
                    try:
                        p.kill()  # Send SIGKILL
                        p.join(timeout=1)
                    except:
                        pass
        else:
            # kill any remaining workers (non-ppw_function case)
            for p in procs:
                p.kill()
    # Wait for all processes to finish (this waits for sweep to finish, but pre/post/model subprocs may take longer)
    for p in procs:
        if ppw_function is not None:
            # For ppw_function processes, we already handled termination above
            # Just ensure they're cleaned up
            if p.is_alive():
                try:
                    p.join(timeout=1)
                except:
                    pass
        else:
            # For regular worker processes
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
                        removed.add(d)  # Fixed: use add() instead of update()
                else:
                    removed.add(d)  # Fixed: use add() instead of update()
            if cleanit > 100:
                break


    if master_dir is not None:
        ret_val = master_p.returncode
        if ret_val != 0:
            raise Exception("start_workers() master returned non-zero: {0}".format(ret_val))



class NetPack(object):
    netpack_type = {
        0: "UNKN", 1: "OK", 2: "CONFIRM_OK", 3: "READY",
        4: "REQ_RUNDIR", 5: "RUNDIR", 6: "REQ_LINPACK",
        7: "LINPACK", 8: "PAR_NAMES", 9: "OBS_NAMES",
        10: "START_RUN", 11: "RUN_FINISHED", 12: "RUN_FAILED",
        13: "RUN_KILLED", 14: "TERMINATE", 15: "PING",
        16: "REQ_KILL", 17: "IO_ERROR", 18: "CORRUPT_MESG",
        19: "DEBUG_LOOP", 20: "DEBUG_FAIL_FREEZE",
        21: "START_FILE_WRKR2MSTR", 22: "CONT_FILE_WRKR2MSTR",
        23: "FINISH_FILE_WRKR2MSTR"}
    sec_message = [1, 3, 5, 7, 9]

    def __init__(self,timeout=0.1,verbose=True):
        self.header_size = 8 + 4 + 8 + 8 + 1001
        self.buf_size = None
        self.buf_idx = (0, 8)
        self.type_idx = (8, 12)
        self.group_idx = (12, 20)
        self.runid_idx = (20, 28)
        self.desc_idx = (28, 1028)
        self.data_idx = (1028,None)

        self.buf_size = None
        self.mtype = None
        self.group = None
        self.runid = None
        self.desc = None
        self.data_pak = None

        self.timeout = float(timeout)
        self.verbose = bool(verbose)

        self.sec_message_buf = bytearray()
        for sm in NetPack.sec_message:
            self.sec_message_buf += sm.to_bytes(1,byteorder="little")


    # def recv(self,num_bytes):
    #     data = bytes()
    #     total = 0
    #     bytes_left = num_bytes
    #     while total < num_bytes:
    #          data += s.recv()

    def recv_all(self,s,msg_len):
        # Helper function to recv n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < msg_len:
            packet = s.recv(msg_len - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def nonblocking_recv(self,s,msg_len):
        try:
            msg = self.recv_all(s,msg_len)
        except socket.timeout as e:
            emess = e.args[0]
            if emess == 'timed out':
                return None
            else:
                raise Exception(e)
        except socket.error as e:
            # Something else happened, handle error, exit, etc.
            raise Exception(e)
        else:
            if msg is None:
                return None
            if len(msg) == 0:
                return None
            else:
                return msg
        # got a message do something :)

    def recv(self,s,dtype=None):
        recv_sec_message = None

        #data = s.recv(len(self.sec_message_buf))
        data = self.nonblocking_recv(s,len(self.sec_message_buf))

        #if len(data) == 0:
        #    return 0
        if data is None:
            return 0

        recv_sec_message = [int(d) for d in data]
        self._check_sec_message(recv_sec_message)
        #data = s.recv(self.header_size)
        #if len(data) == 0:
        #    return -1
        data = self.nonblocking_recv(s,self.header_size)
        if data is None:
            raise Exception("didn't recv header after security message")
        self.buf_size = int.from_bytes(data[self.buf_idx[0]:self.buf_idx[1]], "little")
        self.mtype = int.from_bytes(data[self.type_idx[0]:self.type_idx[1]], "little")
        self.group = int.from_bytes(data[self.group_idx[0]:self.group_idx[1]], "little")
        self.runid = int.from_bytes(data[self.runid_idx[0]:self.runid_idx[1]], "little")
        self.desc = data[self.desc_idx[0]:self.desc_idx[1]].decode().replace("\x00","")
        data_len = self.buf_size - self.data_idx[0] - 1
        self.data_pak = None
        if data_len > 0:
            raw_data = self.nonblocking_recv(s,data_len)
            if raw_data is None:
                raise Exception("didn't recv data pack after header of {0} bytes".format(data_len))
            if dtype is None and self.mtype == 10:
                dtype = float
            self.data_pak = self.deserialize_data(raw_data,dtype)
        return len(data) + data_len


    def deserialize_data(self,data,dtype=None):
        if dtype is not None and dtype in [int,float]:
            return np.frombuffer(data,dtype=dtype)
        ddata = np.array(data.decode().lower().replace('\x00',' ').split())
        if dtype is not None:
            ddata = ddata.astype(dtype)
        return ddata

    def serialize_data(self,data):

        if isinstance(data,str):
            return data.encode()
        elif isinstance(data,list):
            return np.array(data).tobytes()
        elif isinstance(data,np.ndarray):
            return data.tobytes()
        elif isinstance(data,int):
            if self.verbose:
                print("warning: casting int to float to serialize")
            return struct.pack('d',float(data))
        elif isinstance(data,float):
            return struct.pack('d', data)

        else:
            raise Exception("can't serialize unknown 'data' type {0}".format(data))

    def send(self,s,mtype,group,runid,desc,data):
        buf = bytearray()
        buf += self.sec_message_buf
        sdata = self.serialize_data(data) + "\x00".encode()
        buf_size = self.header_size + len(sdata)
        buf += buf_size.to_bytes(length=8,byteorder="little")
        buf += mtype.to_bytes(length=4,byteorder="little")
        buf += group.to_bytes(length=8, byteorder="little")
        buf += runid.to_bytes(length=8, byteorder="little")
        fill_desc = "\x00" * (1001 - len(desc))
        full_desc = desc + fill_desc
        buf += full_desc.encode()
        buf += sdata
        s.sendall(buf)


    def _check_sec_message(self,recv_sec_message):
        if recv_sec_message != self.sec_message:
            raise Exception("recv'd security message {0} invalid, should be {1}".\
                            format(recv_sec_message,self.sec_message))

class PyPestWorker(object):
    """a pure python worker for pest++.  the pest++ master doesnt even know...

    Args:
        pst (str or pyemu.Pst): something about a control file
        host (str): master hostname or IPv4 address
        port (int): port number that the master is listening on
        timeout (float): number of seconds to sleep at different points in the process.  
            if you have lots of pars and/obs, a longer sleep can be helpful, but if you make this smaller,
            the worker responds faster...'it depends'
        verbose (bool): flag to echo what's going on to stdout
        socket_timeout (float): number of seconds that the socket should wait before giving up. 
            generally, this can be a big number...
    """

    def __init__(self, pst, host, port, timeout=0.25,verbose=True, socket_timeout=None):
        self.host = host
        self.port = port
        self._pst_arg = pst
        self._pst = None
        self.s = None
        self.timeout = float(timeout)
        self.net_pack = NetPack(timeout=timeout,verbose=verbose)
        self.verbose = bool(verbose)
        self.par_names = None
        self.obs_names = None
        if socket_timeout is None:
            socket_timeout = timeout * 100
        self.socket_timeout = socket_timeout
        self.par_values = None
        self.max_reconnect_attempts = 10
        self.logger_filename = "pypestworker_{0}.txt".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
        self.logger = Logger(self.logger_filename,echo=verbose)
        self.message("PyPestWorker starting with timeout:{0} and socket_timeout:{1}".format(self.timeout, self.socket_timeout))
        self._process_pst()
        self.connect()
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._listen_thread = threading.Thread(target=self.listen,args=(self._lock,self._send_lock))
        self._listen_thread.start()


    def _process_pst(self):
        self.message("processing control file")
        if isinstance(self._pst_arg,str):
            self._pst = pst_handler.Pst(self._pst_arg)
        elif isinstance(self._pst_arg,pst_handler.Pst):
            self._pst = self._pst_arg
        else:
            raise Exception("unrecognized 'pst' arg:{0}".\
                            format(type(self._pst_arg)))


    def connect(self,is_reconnect=False):
        self.message("trying to connect to {0}:{1}...".format(self.host,self.port))
        self.s = None
        c = 0
        while True:
            try:
                time.sleep(self.timeout)
                c += 1
                if is_reconnect and c > self.max_reconnect_attempts:
                    self.message("max reconnect attempts reached...",True)
                    return False
                self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.s.connect((self.host, self.port))
                self.message("connected to {0}:{1}".format(self.host,self.port))
                break

            except ConnectionRefusedError:
                continue
            except Exception as e:
                continue
            
        self.net_pack = NetPack(timeout=self.timeout,verbose=self.verbose)
        return True


    def message(self,msg,echo=False):
        self.logger.statement(msg)
        if self.verbose or echo:
            print(str(datetime.now())+" : "+msg)


    def recv(self,dtype=None):
        n = self.net_pack.recv(self.s,dtype=dtype)
        if n > 0:
            self.message("recv'd message type:{0}, group:{1}, run_id:{2}, desc:{3}"\
                .format(NetPack.netpack_type[self.net_pack.mtype], 
                    self.net_pack.group, self.net_pack.runid,self.net_pack.desc))
        return n


    def send(self,mtype,group,runid,desc="",data=0):
        try:
            self.net_pack.send(self.s,mtype,group,runid,desc,data)
        except Exception as e:
            self.message("WARNING: error sending message:{0}".format(str(e)), True)
            return False
        self.message("sent message type:{0}, group: {1}, run_id:{2}, desc:{3}".\
            format(NetPack.netpack_type[mtype], group, runid, desc))
        return True

    def listen(self,lock=None,send_lock=None):
        self.s.settimeout(self.socket_timeout)
        failed_reconnect = False
        while True:
            time.sleep(self.timeout)
            try:
                n = self.recv()
            except Exception as e:
                self.message("WARNING: recv exception:"+str(e)+"...trying to reconnect...", True)
                success = self.connect(is_reconnect=True)
                if not success:
                    self.message("...exiting")
                    time.sleep(self.timeout)
                    # set the teminate flag so that the get_pars() look will exit
                    self._lock.acquire()
                    self.net_pack.mtype = 14
                    self._lock.release()
                    return
                else:
                    self.message("...reconnected successfully...", True)
                    continue

            if n > 0:
                # need to sync here
                if self.net_pack.mtype == 10:
                    if lock is not None:
                        lock.acquire()
                    self.par_values =  self.net_pack.data_pak.copy()
                    if lock is not None:
                        lock.release()

                # request cwd
                elif self.net_pack.mtype == 4:
                    if self._send_lock is not None:
                        self._send_lock.acquire()
                    success = self.send(mtype=5, group=self.net_pack.group,
                              runid=self.net_pack.runid,
                              desc="sending cwd", data=os.getcwd())
                    if self._send_lock is not None:
                        self._send_lock.release()
                    if not success:
                        self.message("failed cwd send...trying to reconnect...", True)
                        success = self.connect(is_reconnect=True)
                        if not success:
                            self.message("...exiting", True)
                            time.sleep(self.timeout)
                            return
                        else:
                            self.message("reconnect successfully...", True)
                            continue

                elif self.net_pack.mtype == 8:
                    self.par_names = self.net_pack.data_pak
                    diff = set(self.par_names).symmetric_difference(set(self._pst.par_names))
                    if len(diff) > 0:
                        self.message("WARNING: the following par names are not common\n"+\
                                    " between the control file and the master:{0}".format(','.join(diff)), True)
                elif self.net_pack.mtype == 9:
                    self.obs_names = self.net_pack.data_pak
                    diff = set(self.obs_names).symmetric_difference(set(self._pst.obs_names))
                    if len(diff) > 0:
                        self.message("WARNING: the following obs names are not common\n"+\
                                    " between the control file and the master:{0}".format(','.join(diff)), True)

                elif self.net_pack.mtype == 6:
                    if self._send_lock is not None:
                        self._send_lock.acquire()
                    success = self.send(7, self.net_pack.group,
                              self.net_pack.runid,
                              "fake linpack result", data=1)
                    if self._send_lock is not None:
                        self._send_lock.release()
                    if not success:
                        self.message("failed linpack send...trying to reconnect...", True)
                        success = self.connect(is_reconnect=True)
                        if not success:
                            self.message("...exiting",True)
                            time.sleep(self.timeout)
                            return
                        else:
                            self.message("reconnect successfully...", True)
                            continue

                elif self.net_pack.mtype == 15:
                    if self._send_lock is not None:
                        self._send_lock.acquire()
                    sucess = self.send(15, self.net_pack.group,
                              self.net_pack.runid,
                              "ping back")
                    if self._send_lock is not None:
                        self._send_lock.release()
                    if not success:
                        self.message("failed ping back...trying to reconnect...", True)
                        success = self.connect(is_reconnect=True)
                        if not success:
                            self.message("...exiting",True)
                            time.sleep(self.timeout)
                            return
                        else:
                            self.message("reconnect successfully...", True)
                            continue
                elif self.net_pack.mtype == 14:
                    self.message("recv'd terminate signal", True)
                    return
                elif self.net_pack.mtype == 16:
                    self.message("master is requesting run kill...", True)

                else:
                    self.message("WARNING: unsupported request received: {0}".format(NetPack.netpack_type[self.net_pack.mtype]), True)


    def get_parameters(self):
        pars = None
        while True:
            self._lock.acquire()
            if self.net_pack.mtype == 14:
                self._lock.release()
                return None
            if self.par_values is not None:
                pars = self.par_values
                self._lock.release()
                break
            self._lock.release()
            time.sleep(self.timeout)
        if len(pars) != len(self.par_names):
            raise Exception("len(par vals) {0} != len(par names)".format(len(pars),len(self.par_names)))
        return pd.Series(data=pars,index=self.par_names)


    def send_observations(self,obsvals,parvals=None,request_more_pars=True):
        if len(obsvals) != len(self.obs_names):
            raise Exception("len(obs vals) {0} != len(obs names)".format(len(obsvals), len(self.obs_names)))
        if isinstance(obsvals,np.ndarray):
            _obsvals = obsvals
        elif isinstance(obsvals,pd.Series):
            _obsvals = obsvals.loc[self.obs_names].values
        elif isinstance(obsvals,list):
            _obsvals = np.array(obsvals)
        else:
            raise Exception("unknown obsvals type {0}".format(type(obsvals)))
        if np.any(np.isnan(_obsvals)):
            raise Exception("nans in obsvals")

        if parvals is None:
            _parvals = self.par_values
        elif isinstance(parvals,np.ndarray):
            _parvals = parvals
        elif isinstance(parvals,pd.Series):
            _parvals = parvals.loc[self.par_names].values
        elif isinstance(parvals,list):
            _parvals = np.array(parvals)
        else:
            raise Exception("unknown parvals type {0}".format(type(parvals)))
        if np.any(np.isnan(_parvals)):
            raise Exception("nans in parvals")

        #first reset par_values to None
        self._lock.acquire()
        self.par_values = None
        self._lock.release()

        data = np.concatenate((_parvals,_obsvals))

        self._send_lock.acquire()
        # send the stack obsvals and par vals
        self.send(11, self.net_pack.group, self.net_pack.runid, self.net_pack.desc, data=data)

        # now send the ready message
        if request_more_pars:
            self.send(3,0,0,"ready for next run",data=0)
        self._send_lock.release()


    def request_more_pars(self):
        self._send_lock.acquire()
        self.send(3, 0, 0, "ready for next run", data=0.0)
        self._send_lock.release()


    def send_failed_run(self,group=None,runid=None,desc="failed"):
        if group is None:
            group = self.net_pack.group
        if runid is None:
            runid = self.net_pack.runid
        self._send_lock.acquire()
        self.send(12, int(group), int(runid), desc, data=0.0)
        self._send_lock.release()


    def send_killed_run(self,group=None,runid=None,desc="killed"):
        if group is None:
            group = self.net_pack.group
        if runid is None:
            runid = self.net_pack.runid
        self._send_lock.acquire()
        self.send(13, int(group), int(runid), desc, data=0.0)
        self._send_lock.release()




class PortManager(object):
    """Cross-platform port manager for parallel processes."""
    def __init__(self,
                 port_range=(4004, 4999),
                 lock_dir=None,
                 max_retries=50,
                 lock_timeout=5,
                 log_level=logging.INFO):
        """
        Initialize the port manager.
        Args:
            port_range: Tuple of (min_port, max_port) to search within
            lock_dir: Directory to store lock files (default: system temp dir)
            max_retries: Maximum attempts to find an available port
            lock_timeout: Time in seconds after which a lock is considered stale
        """
        # Set up instance-specific logger
        self.logger = logging.getLogger(f"{__name__}.PortManager.{id(self)}")
        self.logger.setLevel(log_level)
        # Add a handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.min_port, self.max_port = port_range
        self.lock_dir = lock_dir or os.path.join(tempfile.gettempdir(), "port_locks")
        self.max_retries = max_retries
        self.lock_timeout = lock_timeout
        # Ensure lock directory exists
        os.makedirs(self.lock_dir, exist_ok=True)
        # Generate a unique ID for this process instance
        self.instance_id = str(uuid.uuid4())
    
    def _is_port_available(self, port):
        """Check if a port is available by attempting to bind to it."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Set socket to reuse address to handle TIME_WAIT state
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', port))
                return True
            except (socket.error, OSError):
                return False
    
    def _get_lock_file(self, port):
        """Get the path to the lock file for a specific port."""
        return os.path.join(self.lock_dir, f"port_{port}.lock")
    
    def _clean_stale_locks(self):
        """Remove stale lock files based on timeout."""
        now = time.time()
        try:
            for filename in os.listdir(self.lock_dir):
                if filename.startswith("port_") and filename.endswith(".lock"):
                    lock_path = os.path.join(self.lock_dir, filename)
                    if os.path.exists(lock_path):
                        # Check if lock is stale
                        if now - os.path.getmtime(lock_path) > self.lock_timeout:
                            try:
                                os.remove(lock_path)
                                self.logger.debug(f"Removed stale lock file: {lock_path}")
                            except OSError:
                                # Another process might have removed it already
                                pass
        except Exception as e:
            self.logger.warning(f"Error cleaning stale locks: {e}")
    
    @contextmanager
    def _try_lock_port(self, port):
        """
        Try to create a lock file for a port using a cross-platform approach.
        Uses atomic file creation to implement locking.
        """
        lock_file = self._get_lock_file(port)
        lock_acquired = False
        try:
            # Try to create the lock file - will only succeed if it doesn't exist
            lock_data = {
                "pid": os.getpid(),
                "instance_id": self.instance_id,
                "timestamp": time.time()
            }
            try:
                # Try exclusive creation of the file (atomic operation)
                with open(lock_file, 'x') as f:
                    json.dump(lock_data, f)
                lock_acquired = True
                yield True
            except FileExistsError:
                # Lock file already exists
                try:
                    # Check if lock file is stale
                    if os.path.exists(lock_file):
                        if time.time() - os.path.getmtime(lock_file) > self.lock_timeout:
                            # Lock is stale, try to replace it
                            try:
                                os.remove(lock_file)
                                with open(lock_file, 'x') as f:
                                    json.dump(lock_data, f)
                                lock_acquired = True
                                yield True
                                return
                            except (FileExistsError, OSError):
                                # Failed to acquire lock
                                pass
                except OSError:
                    pass
                yield False
        finally:
            # Clean up the lock file if we created it
            if lock_acquired:
                try:
                    if os.path.exists(lock_file):
                        os.remove(lock_file)
                except OSError as e:
                    self.logger.warning(f"Error removing lock file for port {port}: {e}")
    
    def get_available_port(self):
        """
        Find and reserve an available port.
        Returns:
            An available port number.
        Raises:
            RuntimeError: If no available port can be found after max_retries.
        """
        # Clean up stale locks first
        self._clean_stale_locks()
        # Shuffle port range to distribute port selection
        port_list = list(range(self.min_port, self.max_port + 1))
        random.shuffle(port_list)
        attempts = 0
        while attempts < self.max_retries:
            # Pick a random port from our shuffled list
            if not port_list:
                raise RuntimeError("Exhausted all ports in range")
            port = port_list.pop(0)
            attempts += 1
            # First check if port is available
            if not self._is_port_available(port):
                continue
            # Try to acquire a lock
            with self._try_lock_port(port) as locked:
                if not locked:
                    # Another process got this port
                    continue
                # Double-check port is still available after locking
                if self._is_port_available(port):
                    self.logger.info(f"Reserved port {port} for process {os.getpid()}")
                    return port
        raise RuntimeError(f"Could not find available port after {self.max_retries} attempts")
    
    @contextmanager
    def reserved_port(self):
        """Context manager that reserves a port and releases it after use."""
        port = self.get_available_port()
        lock_file = self._get_lock_file(port)
        try:
            yield port
        finally:
            # Release the port by removing the lock file
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                    self.logger.info(f"Released port {port}")
                except OSError as e:
                    self.logger.warning(f"Error releasing port {port}: {e}")




if __name__ == "__main__":
    host = "localhost"
    port = PortManager().get_available_port()
    ppw = PyPestWorker(None,host,port)
    #ppw.initialize()