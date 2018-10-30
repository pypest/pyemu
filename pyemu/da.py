import os
import sys
import pyemu
import multiprocessing as mp
from pyemu_warnings import PyemuWarning
import copy


class Assimilator():
    def __init__(self, type='Smoother', iterate=False, pst=None, mode='stochastic', options={}):
        """
        A clase to implement one update cycle. For the Ensemble smoother, the update cycle includes all available
        observations; for ensemble filter, it will update parameter given new observations only; and finally for Kalman Smoother,
        the parameters will be updated given all observations avialble before a certain time
        """
        self.mode_options = ['Stochastic', 'Deterministic']
        self.type_options = ['Smoother', 'Kalman Filter', 'Kalman Smoother']

        if not isinstance(iterate, bool):
            raise ValueError("Error, 'iterate' must be boolian")  # I think warning would better
        self.iterate = iterate  # If true Chen-Oliver scheme will be used, otherwise classical data assimilation will used.

        if type not in self.type_options:
            raise ValueError("Assimilation type [{}] specified is not supported".format(type))
        self.type = type

        if mode not in self.mode_options:
            raise ValueError("Assimilation mode [{}] specified is not supported".format(mode))
        self.mode = mode

        # Obtain problem setting from the pst object.
        if isinstance(pst, pyemu):
            self.pst = pst
        else:  # TODO: it is better if we check if this is a file, throw exception otherwise
            self.pst = pyemu.Pst(pst)

        # Time Cycle info. example
        # {[ ['file_1.tpl', 'file_1.dat'],
        #    ['file_2.tpl', 'file_2.dat'],
        #    ['out_file.ins', 'out_file.out']}

        self.cycle_update_files = {}

    def forcast(self):
        """
        This function implements (1) generation of random ensemble for parameters, (2) forward model runs.

        :return:

        """
        self.generate_priors()
        self.model_evalutions()

    def generate_priors(self):
        """
        Use parameters dataframe from Pst to generate prior realizations
        :return:
        """
        # ToDo: Any plans to deal with nonGaussian distributions?

        pass

    def model_evalutions(self):
        pass

    def update(self, Pst):

        """
        Solve the anlaysis step
        Xa = Xf + C'H(H'CH + R) (do - df)
        """
        if self.mode == 'stochastic':
            pass
        else:
            # deterministic
            # Xa' = Xf' + C'H(H'CH + R) (do' - df')
            pass



    def run(self):
        """

        :return:
        """
        # TODO: add exceptions and warnings
        if self.type == "Smoother":
            self.smoother()
        elif self.type == "Kalman_filter":
            self.enkf()
        elif self.type == "Kalman_smoother":
            self.enks()
        else:
            print("We should'nt be here....")  # TODO: ???

    def smoother(self, pst):
        if self.iterate:
            # Chen_oliver algorithim
            pass
        else:
            self.forcast(pst)
            self.update(pst)

    def enkf(self):
        """
        Loop over time windows and apply da
        :return:
        """

        for cycle_index, time_point in enumerate(self.timeline):
            if cycle_index >= len(self.timeline) - 1:
                # Logging : Last Update cycle has finished
                break

            print("Print information about this assimilation Cycle ???")  # should be handeled in Logger

            # each cycle should have a dictionary of template files and instruction files to update the model inout
            # files
            # get current cycle update information
            current_cycle_files = self.cycle_update_files[cycle_index]

            #  (1)  update model input files for this cycle
            self.model_temporal_evolotion(cycle_index, current_cycle_files)

            # (2) generate new Pst object for the current time cycle
            current_pst = copy.deepcopy(self.pst)
            # update observation dataframe
            # update parameter dataframe
            # update in/out files if needed

            # At this stage the problem is equivalent to smoother problem
            self.smoother(current_pst)

            # how to save results for this cycle???

    def enks(self, pst):
        """ Similiar to EnkF ---  wait???"""

        pass


    def model_temporal_evolotion(self, time_index, cycle_files):
        """
         - The function prepares the model for this time cycle
         - Any time-dependant forcing should be handled here. This includes temporal stresses, boundary conditions, and
         initial conditions.
         - Two options are available (1) template files to update input parameters
                                     (2) use results from previous cycle to update input files using instruction
                                     files.
                                     (3) the user must prepare a pair of files : .tpl(or .ins) and corresponding file to change
        :param time_index:
        :return:
        """
        for file in cycle_files:
            # generate log here about files being updated
            self.update_inputfile(file[0], file[1])  # Jeremy: do we have something like this in python?


if __name__ == "__main__":
    sm = Assimilator(type='Smoother', iterate=False, mode='stochastic', pst='pst.control')
    sm.run()

    # Ensemble Kalman Filter
    kf = Assimilator(type='Kalman_filter', pst='pst.control')
    kf.run()

    # Kalman Smoother
    ks = Assimilator(type='Kalman_smoother', pst='pst.control')
    ks.run()
