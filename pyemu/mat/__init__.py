"""This module contains classes for handling matrices in a linear algebra setting.
The primary objects are the `Matrix` and `Cov`.  These objects overload most numerical
operators to autoalign the elements based on row and column names."""

from .mat_handler import Matrix, Cov, Jco, concat, save_coo
