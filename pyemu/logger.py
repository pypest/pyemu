from __future__ import print_function, division
from datetime import datetime
import copy

class Logger(object):
    """ a basic class for logging events during the linear analysis calculations
        if filename is passed, then an file handle is opened
    Parameters:
    ----------
        filename (bool or string): if string, it is the log file to write
            if a bool, then log is written to the screen
        echo (bool): a flag to force screen output
    Attributes:
    ----------
        items (dict) : tracks when something is started.  If a log entry is
            not in items, then it is treated as a new entry with the string
            being the key and the datetime as the value.  If a log entry is
            in items, then the end time and delta time are written and
            the item is popped from the keys

    """
    def __init__(self,filename, echo=False):
        self.items = {}
        self.echo = bool(echo)
        if filename == True:
            self.echo = True
            self.filename = None
        elif filename:
            self.filename = filename
            self.f = open(filename, 'w')
            self.t = datetime.now()
            self.log("opening " + str(filename) + " for logging")
        else:
            self.filename = None


    def statement(self,phrase):
        t = datetime.now()
        s = str(t) + ' ' + str(phrase) + '\n'
        if self.echo:
            print(s,end='')
        if self.filename:
            self.f.write(s)
            self.f.flush()


    def log(self,phrase):
        """log something that happened
        Parameters:
        ----------
            phrase (str) : the thing that happened
        Returns:
        -------
            None
        """
        pass
        t = datetime.now()
        if phrase in self.items.keys():
            s = str(t) + ' finished: ' + str(phrase) + " took: " + \
                str(t - self.items[phrase]) + '\n'
            if self.echo:
                print(s,end='')
            if self.filename:
                self.f.write(s)
                self.f.flush()
            self.items.pop(phrase)
        else:
            s = str(t) + ' starting: ' + str(phrase) + '\n'
            if self.echo:
                print(s,end='')
            if self.filename:
                self.f.write(s)
                self.f.flush()
            self.items[phrase] = copy.deepcopy(t)

    def warn(self,message):
        """write a warning to the log file
        Parameters:
        ----------
            message (str) : the warning text
        Returns:
        -------
            None
        """
        s = str(datetime.now()) + " WARNING: " + message + '\n'
        if self.echo:
            print(s,end='')
        if self.filename:
            self.f.write(s)
            self.f.flush

    def lraise(self,message):
        s = str(datetime.now()) + " ERROR: " + message + '\n'
        print(s,end='')
        if self.filename:
            self.f.write(s)
            self.f.flush
            self.f.close()
        raise Exception(message)


