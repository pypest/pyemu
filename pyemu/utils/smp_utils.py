"""PEST-style site sample (smp) file support utilities
"""
import os
import sys
import platform
import shutil
import subprocess as sp
import warnings
import socket
import time
from datetime import datetime
import numpy as np
import pandas as pd
from ..pyemu_warnings import PyemuWarning


def smp_to_ins(
    smp_filename,
    ins_filename=None,
    use_generic_names=False,
    gwutils_compliant=False,
    datetime_format=None,
    prefix="",
):
    """create an instruction file for an smp file

    Args:
        smp_filename (`str`):path and name of an existing smp file
        ins_filename (`str`, optional): the name of the instruction
            file to create.  If None, `smp_filename` +".ins" is used.
            Default is None.
        use_generic_names (`bool`): flag to force observations names
            to use a generic `int` counter instead of trying to use a
            datetime string.  Default is False
        gwutils_compliant (`bool`): flag to use instruction set that
            is compliant with the PEST gw utils (fixed format instructions).
            If false, use free format (with whitespace) instruction set.
            Default is False
        datetime_format (`str`): string to pass to datetime.strptime in
            the `smp_utils.smp_to_dataframe()` function.  If None, not
            used. Default is None.
        prefix (`str`): a prefix to add to the front of the derived
            observation names.  Default is ''


    Returns:
        `pandas.DataFrame`: a dataframe of the smp file
        information with the observation names and
        instruction lines as additional columns.

    Example::

        df = pyemu.smp_utils.smp_to_ins("my.smp")

    """
    if ins_filename is None:
        ins_filename = smp_filename + ".ins"
    df = smp_to_dataframe(smp_filename, datetime_format=datetime_format)

    # Read raw lines for accurate instruction generation.
    # The PEST 'w' instruction uses a two-step algorithm that treats
    # leading whitespace differently: it consumes one 'w' to skip past
    # leading blanks without advancing past a word. This means files
    # with leading whitespace need an extra 'w' marker. For fixed format,
    # the value column position depends on the actual file layout.
    with open(smp_filename) as f:
        raw_lines = [line for line in f.readlines() if line.strip()]

    df.loc[:, "ins_strings"] = None
    df.loc[:, "observation_names"] = None
    name_groups = df.groupby("name").groups
    for name, idxs in name_groups.items():
        if not use_generic_names and len(name) <= 11:
            onames = (
                df.loc[idxs, "datetime"]
                .apply(lambda x: prefix + name + "_" + x.strftime("%d%m%Y"))
                .values
            )
        else:
            onames = [prefix + name + "_{0:d}".format(i) for i in range(len(idxs))]
        if False in (map(lambda x: len(x) <= 20, onames)):
            long_names = [oname for oname in onames if len(oname) > 20]
            raise Exception(
                "observation names longer than 20 chars:\n{0}".format(str(long_names))
            )
        if gwutils_compliant:
            ins_strs = []
            for i, on in zip(idxs, onames):
                raw_line = raw_lines[i].rstrip()
                # Find the value column range (last whitespace-delimited token)
                j = len(raw_line) - 1
                while j >= 0 and raw_line[j] != ' ':
                    j -= 1
                val_start = j + 2  # 1-indexed for PEST
                val_end = len(raw_line)  # 1-indexed, inclusive
                ins_strs.append(
                    "l1  ({0:s}){1:d}:{2:d}".format(on, val_start, val_end)
                )
        else:
            ins_strs = []
            for i, on in zip(idxs, onames):
                raw_line = raw_lines[i]
                if raw_line[0] == ' ':
                    ins_strs.append("l1 w w w w !{0:s}!".format(on))
                else:
                    ins_strs.append("l1 w w w !{0:s}!".format(on))
        df.loc[idxs, "observation_names"] = onames
        df.loc[idxs, "ins_strings"] = ins_strs

    counts = df.observation_names.value_counts()
    dup_sites = [name for name in counts.index if counts[name] > 1]
    if len(dup_sites) > 0:
        raise Exception(
            "duplicate observation names found:{0}".format(",".join(dup_sites))
        )

    with open(ins_filename, "w") as f:
        f.write("pif ~\n")
        [f.write(ins_str + "\n") for ins_str in df.loc[:, "ins_strings"]]
    return df


def dataframe_to_smp(
    dataframe,
    smp_filename,
    name_col="name",
    datetime_col="datetime",
    value_col="value",
    datetime_format="dd/mm/yyyy",
    value_format="{0:15.6E}",
    max_name_len=12,
):
    """write a dataframe as an smp file

    Args:
        dataframe (`pandas.DataFrame`): the dataframe to write to an SMP
            file.  This dataframe should be in "long" form - columns for
            site name, datetime, and value.
        smp_filename (`str`): smp file to write
        name_col (`str`,optional): the name of the dataframe column
            that contains the site name.  Default is "name"
        datetime_col (`str`): the column in the dataframe that the
            datetime values.  Default is "datetime".
        value_col (`str`): the column in the dataframe that is the values
        datetime_format (`str`, optional): The format to write the datetimes in the
            smp file.  Can be either 'dd/mm/yyyy' or 'mm/dd/yyy'.  Default
            is 'dd/mm/yyyy'.
        value_format (`str`, optional):  a python float-compatible format.
            Default is "{0:15.6E}".

    Example::

        pyemu.smp_utils.dataframe_to_smp(df,"my.smp")

    """
    formatters = {
        "name": lambda x: "{0:<20s}".format(str(x)[:max_name_len]),
        "value": lambda x: value_format.format(x),
    }
    if datetime_format.lower().startswith("d"):
        dt_fmt = "%d/%m/%Y    %H:%M:%S"
    elif datetime_format.lower().startswith("m"):
        dt_fmt = "%m/%d/%Y    %H:%M:%S"
    else:
        raise Exception(
            "unrecognized datetime_format: " + "{0}".format(str(datetime_format))
        )

    for col in [name_col, datetime_col, value_col]:
        assert col in dataframe.columns

    dataframe.loc[:, "datetime_str"] = dataframe.loc[:, "datetime"].apply(
        lambda x: x.strftime(dt_fmt)
    )
    if isinstance(smp_filename, str):
        smp_filename = open(smp_filename, "w")
        # need this to remove the leading space that pandas puts in front
        s = dataframe.loc[:, [name_col, "datetime_str", value_col]].to_string(
            col_space=0, formatters=formatters, justify=None, header=False, index=False
        )
        for ss in s.split("\n"):
            smp_filename.write("{0:<s}\n".format(ss.strip()))
    dataframe.pop("datetime_str")


def _date_parser(items):
    """datetime parser to help load smp files"""
    try:
        dt = datetime.strptime(items, "%d/%m/%Y %H:%M:%S")
    except Exception as e:
        try:
            dt = datetime.strptime(items, "%m/%d/%Y %H:%M:%S")
        except Exception as ee:
            raise Exception(
                "error parsing datetime string"
                + " {0}: \n{1}\n{2}".format(str(items), str(e), str(ee))
            )
    return dt


def smp_to_dataframe(smp_filename, datetime_format=None):
    """load an smp file into a pandas dataframe

    Args:
        smp_filename (`str`): path and name of existing smp filename to load
        datetime_format (`str`, optional): The format of the datetime strings
            in the smp file. Can be either "%m/%d/%Y %H:%M:%S" or "%d/%m/%Y %H:%M:%S"
            If None, then we will try to deduce the format for you, which
            always dangerous.

    Returns:
        `pandas.DataFrame`: a dataframe with index of datetime and columns of
        site names.  Missing values are set to NaN.

    Example::

        df = smp_to_dataframe("my.smp")

    """

    if datetime_format is not None:
        date_func = lambda x: datetime.strptime(x, datetime_format)
    else:
        date_func = _date_parser
    df = pd.read_csv(
        smp_filename,
        sep=r"\s+",
        header=None,
        names=["name", "date", "time", "value"],
        dtype={"name": object, "value": np.float64},
        na_values=["dry"],
    )
    df['datetime'] = (df.date + " " + df.time).apply(date_func)
    return df
