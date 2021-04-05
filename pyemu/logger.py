"""module for logging pyemu progress
"""
from __future__ import print_function, division
from datetime import datetime
import warnings
from .pyemu_warnings import PyemuWarning
import copy


class Logger(object):
    """a basic class for logging events during the linear analysis calculations
        if filename is passed, then a file handle is opened.

    Args:
        filename (`str`): Filename to write logged events to. If False, no file will be created,
            and logged events will be displayed on standard out.
        echo (`bool`):  Flag to cause logged events to be echoed to the screen.

    """

    def __init__(self, filename, echo=False):
        self.items = {}
        self.echo = bool(echo)
        if filename == True:
            self.echo = True
            self.filename = None
        elif filename:
            self.filename = filename
            self.f = open(filename, "w")
            self.t = datetime.now()
            self.log("opening " + str(filename) + " for logging")
        else:
            self.filename = None

    def statement(self, phrase):
        """log a one-time statement

        Arg:
            phrase (`str`): statement to log

        """
        t = datetime.now()
        s = str(t) + " " + str(phrase) + "\n"
        if self.echo:
            print(s, end="")
        if self.filename:
            self.f.write(s)
            self.f.flush()

    def log(self, phrase):
        """log something that happened.

        Arg:
            phrase (`str`): statement to log

        Notes:
            The first time phrase is passed the start time is saved.
                The second time the phrase is logged, the elapsed time is written
        """
        pass
        t = datetime.now()
        if phrase in self.items.keys():
            s = (
                str(t)
                + " finished: "
                + str(phrase)
                + " took: "
                + str(t - self.items[phrase])
                + "\n"
            )
            if self.echo:
                print(s, end="")
            if self.filename:
                self.f.write(s)
                self.f.flush()
            self.items.pop(phrase)
        else:
            s = str(t) + " starting: " + str(phrase) + "\n"
            if self.echo:
                print(s, end="")
            if self.filename:
                self.f.write(s)
                self.f.flush()
            self.items[phrase] = copy.deepcopy(t)

    def warn(self, message):
        """write a warning to the log file.

        Arg:
            phrase (`str`): warning statement to log


        """
        s = str(datetime.now()) + " WARNING: " + message + "\n"
        if self.echo:
            print(s, end="")
        if self.filename:
            self.f.write(s)
            self.f.flush
        warnings.warn(s, PyemuWarning)

    def lraise(self, message):
        """log an exception, close the log file, then raise the exception

        Arg:
            phrase (`str`): exception statement to log and raise

        """
        s = str(datetime.now()) + " ERROR: " + message + "\n"
        print(s, end="")
        if self.filename:
            self.f.write(s)
            self.f.flush
            self.f.close()
        raise Exception(message)
