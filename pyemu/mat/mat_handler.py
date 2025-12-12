from __future__ import print_function, division
import os
import copy
import struct
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# import scipy.linalg as la


from pyemu.pst.pst_handler import Pst
from ..pyemu_warnings import PyemuWarning


def save_coo(x, row_names, col_names, filename, chunk=None):
    """write a PEST-compatible binary file.  The data format is
    [int,int,float] for i,j,value.  It is autodetected during
    the read with Matrix.from_binary().

    Args:
        x (`numpy.sparse`): coo sparse matrix
        row_names ([`str`]): list of row names
        col_names (['str]): list of col_names
        filename (`str`):  filename
        droptol (`float`): absolute value tolerance to make values
            smaller than `droptol` zero.  Default is None (no dropping)
        chunk (`int`): number of elements to write in a single pass.
            Default is None

    """

    f = open(filename, "wb")
    # print("counting nnz")
    # write the header
    header = np.array((x.shape[1], x.shape[0], x.nnz), dtype=Matrix.binary_header_dt)
    header.tofile(f)

    data = np.rec.fromarrays([x.row, x.col, x.data], dtype=Matrix.coo_rec_dt)
    data.tofile(f)

    for name in col_names:
        if len(name) > Matrix.new_par_length:
            name = name[: Matrix.new_par_length - 1]
        elif len(name) < Matrix.new_par_length:
            for _ in range(len(name), Matrix.new_par_length):
                name = name + " "
        f.write(name.encode())
    for name in row_names:
        if len(name) > Matrix.new_obs_length:
            name = name[: Matrix.new_obs_length - 1]
        elif len(name) < Matrix.new_obs_length:
            for i in range(len(name), Matrix.new_obs_length):
                name = name + " "
        f.write(name.encode())
    f.close()


def concat(mats):
    """Concatenate Matrix objects.  Tries either axis.

    Args:
        mats ([`Matrix`]): list of Matrix objects

    Returns:
        `pyemu.Matrix`: a concatenated `Matrix` instance
    """
    for mat in mats:
        if mat.isdiagonal:
            raise NotImplementedError("concat not supported for diagonal mats")

    row_match = True
    col_match = True
    for mat in mats[1:]:
        if sorted(mats[0].row_names) != sorted(mat.row_names):
            row_match = False
        if sorted(mats[0].col_names) != sorted(mat.col_names):
            col_match = False
    if not row_match and not col_match:
        raise Exception(
            "mat_handler.concat(): all Matrix objects"
            + "must share either rows or cols"
        )

    if row_match and col_match:
        raise Exception(
            "mat_handler.concat(): all Matrix objects" + "share both rows and cols"
        )

    if row_match:
        row_names = copy.deepcopy(mats[0].row_names)
        col_names = []
        for mat in mats:
            col_names.extend(copy.deepcopy(mat.col_names))
        x = mats[0].newx.copy()
        for mat in mats[1:]:
            mat.align(mats[0].row_names, axis=0)
            other_x = mat.newx
            x = np.append(x, other_x, axis=1)

    else:
        col_names = copy.deepcopy(mats[0].col_names)
        row_names = []
        for mat in mats:
            row_names.extend(copy.deepcopy(mat.row_names))
        x = mats[0].newx.copy()
        for mat in mats[1:]:
            mat.align(mats[0].col_names, axis=1)
            other_x = mat.newx
            x = np.append(x, other_x, axis=0)
    return Matrix(x=x, row_names=row_names, col_names=col_names)


def get_common_elements(list1, list2):
    """find the common elements in two lists.  used to support auto align
        might be faster with sets

    Args:
        list1 ([`str`]): a list of strings (could be either row or col
            names, depending on calling function)
        list2 ([`str`]): a list of strings (could be either row or col
            names, depending on calling function)

    Returns:
        [`str`]:  list of common strings shared by list1 and list2

    Note:
        `result` is not ordered WRT `list1` or `list2`
    """
    set2 = set(list2)
    result = [item for item in list1 if item in set2]
    return result


class Matrix(object):
    """Easy linear algebra in the PEST(++) realm

    Args:
        x (`numpy.ndarray`): numeric values
        row_names ([`str`]): list of row names
        col_names (['str']): list of column names
        isdigonal (`bool`): flag if the Matrix is diagonal
        autoalign (`bool`): flag to control the autoalignment of Matrix
            during linear algebra operations

    Example::

        data = np.random.random((10,10))
        row_names = ["row_{0}".format(i) for i in range(10)]
        col_names = ["col_{0}".format(j) for j in range(10)]
        mat = pyemu.Matrix(x=data,row_names=row_names,col_names=col_names)
        mat.to_binary("mat.jco")

        # load an existing jacobian matrix
        jco = pyemu.Jco.from_binary("pest.jco")
        # form an observation noise covariance matrix from weights
        obscov = pyemu.Cov.from_observation_data(pst)
        # form the normal matrix, aligning rows and cols on-the-fly
        xtqx = jco * obscov.inv * jco.T


    Note:
        this class makes heavy use of property decorators to encapsulate
        private attributes

    """

    integer = np.int32
    double = np.float64
    char = np.uint8

    binary_header_dt = np.dtype(
        [("itemp1", integer), ("itemp2", integer), ("icount", integer)]
    )
    binary_rec_dt = np.dtype([("j", integer), ("dtemp", double)])
    coo_rec_dt = np.dtype([("i", integer), ("j", integer), ("dtemp", double)])

    par_length = 12
    obs_length = 20
    new_par_length = 200
    new_obs_length = 200

    def __init__(
        self, x=None, row_names=[], col_names=[], isdiagonal=False, autoalign=True
    ):

        self.col_names, self.row_names = [], []
        _ = [self.col_names.append(str(c).lower()) for c in col_names]
        _ = [self.row_names.append(str(r).lower()) for r in row_names]
        self.__x = None
        self.__u = None
        self.__s = None
        self.__v = None
        if x is not None:
            if x.ndim != 2:
                raise Exception("ndim != 2")
            # x = np.atleast_2d(x)
            if isdiagonal and len(row_names) > 0:
                # assert 1 in x.shape,"Matrix error: diagonal matrix must have " +\
                #                    "one dimension == 1,shape is {0}".format(x.shape)
                mx_dim = max(x.shape)
                if len(row_names) != mx_dim:
                    raise Exception(
                        "Matrix.__init__(): diagonal shape[1] != len(row_names) "
                        + str(x.shape)
                        + " "
                        + str(len(row_names))
                    )
                if mx_dim != x.shape[0]:
                    x = x.transpose()
                # x = x.transpose()
            else:
                if len(row_names) > 0:
                    if len(row_names) != x.shape[0]:
                        raise Exception(
                            "Matrix.__init__(): shape[0] != len(row_names) "
                            + str(x.shape)
                            + " "
                            + str(len(row_names))
                        )

                if len(col_names) > 0:
                    # if this a row vector
                    if len(row_names) == 0 and x.shape[1] == 1:
                        x.transpose()
                    if len(col_names) != x.shape[1]:
                        raise Exception(
                            "Matrix.__init__(): shape[1] != len(col_names) "
                            + str(x.shape)
                            + " "
                            + str(len(col_names))
                        )
            self.__x = x

        self.isdiagonal = bool(isdiagonal)
        self.autoalign = bool(autoalign)

    def reset_x(self, x, copy=True):
        """reset self.__x private attribute

        Args:
            x (`numpy.ndarray`): the new numeric data
            copy (`bool`): flag to make a copy of 'x'. Default is True

        Returns:
            None

        Note:
            operates in place

        """
        if x.shape != self.shape:
            raise Exception("shape mismatch")
        if copy:
            self.__x = x.copy()
        else:
            self.__x = x

    def __str__(self):
        """overload of object.__str__()

        Returns:
            `str`: string representation

        """
        s = (
            "shape:{0}:{1}".format(*self.shape)
            + " row names: "
            + str(self.row_names)
            + "\n"
            + "col names: "
            + str(self.col_names)
            + "\n"
            + str(self.__x)
        )
        return s

    def __getitem__(self, item):
        """a very crude overload of object.__getitem__().

        Args:
            item (`object`): something that can be used as an index

        Returns:
            `Matrix`: an object that is a sub-matrix of `Matrix`

        """
        if self.isdiagonal and isinstance(item, tuple):
            submat = np.atleast_2d((self.__x[item[0]]))
        else:
            submat = np.atleast_2d(self.__x[item])
        # transpose a row vector to a column vector
        if submat.shape[0] == 1:
            submat = submat.transpose()
        row_names = self.row_names[: submat.shape[0]]
        if self.isdiagonal:
            col_names = row_names
        else:
            col_names = self.col_names[: submat.shape[1]]
        return type(self)(
            x=submat,
            isdiagonal=self.isdiagonal,
            row_names=row_names,
            col_names=col_names,
            autoalign=self.autoalign,
        )

    def __pow__(self, power):
        """overload of numpy.ndarray.__pow__() operator

        Args:
            power (`float`): interpreted as follows: -1 = inverse of self,
                -0.5 = sqrt of inverse of self,
                0.5 = sqrt of self. All other positive
                ints = elementwise self raised to power

        Returns:
            `Matrix`: a new Matrix object raised to the power `power`

        Example::

            cov = pyemu.Cov.from_uncfile("my.unc")
            sqcov = cov**2

        """
        if power < 0:
            if power == -1:
                return self.inv
            elif power == -0.5:
                return (self.inv).sqrt
            else:
                raise NotImplementedError(
                    "Matrix.__pow__() not implemented "
                    + "for negative powers except for -1"
                )

        elif int(power) != float(power):
            if power == 0.5:
                return self.sqrt
            else:
                raise NotImplementedError(
                    "Matrix.__pow__() not implemented "
                    + "for fractional powers except 0.5"
                )
        else:
            return type(self)(
                self.__x ** power,
                row_names=self.row_names,
                col_names=self.col_names,
                isdiagonal=self.isdiagonal,
            )

    def __sub__(self, other):
        """numpy.ndarray.__sub__() overload.  Tries to speedup by
         checking for scalars of diagonal matrices on either side of operator

        Args:
            other : (`int`,`float`,`numpy.ndarray`,`Matrix`): the thing to subtract

        Returns:
            `Matrix`: the result of subtraction

        Note:
            if `Matrix` and other (if applicable) have `autoalign` set to `True`,
            both `Matrix` and `other` are aligned based on row and column names.
            If names are not common between the two, this may result in a smaller
            returned `Matrix`.  If no names are shared, an exception is raised

        Example::

            jco1 = pyemu.Jco.from_binary("pest.1.jcb")
            jco2 = pyemu.Jco.from_binary("pest.2.jcb")
            diff = jco1 - jco2


        """

        if np.isscalar(other):
            return Matrix(
                x=self.x - other,
                row_names=self.row_names,
                col_names=self.col_names,
                isdiagonal=self.isdiagonal,
            )
        else:
            if isinstance(other, pd.DataFrame):
                other = Matrix.from_dataframe(other)

            if isinstance(other, np.ndarray):
                if self.shape != other.shape:
                    raise Exception(
                        "Matrix.__sub__() shape"
                        + "mismatch: "
                        + str(self.shape)
                        + " "
                        + str(other.shape)
                    )
                if self.isdiagonal:
                    elem_sub = -1.0 * other
                    for j in range(self.shape[0]):
                        elem_sub[j, j] += self.x[j]
                    return type(self)(
                        x=elem_sub, row_names=self.row_names, col_names=self.col_names
                    )
                else:
                    return type(self)(
                        x=self.x - other,
                        row_names=self.row_names,
                        col_names=self.col_names,
                    )
            elif isinstance(other, Matrix):
                if (
                    self.autoalign
                    and other.autoalign
                    and not self.element_isaligned(other)
                ):
                    common_rows = get_common_elements(self.row_names, other.row_names)
                    common_cols = get_common_elements(self.col_names, other.col_names)

                    if len(common_rows) == 0:
                        raise Exception("Matrix.__sub__ error: no common rows")

                    if len(common_cols) == 0:
                        raise Exception("Matrix.__sub__ error: no common cols")
                    first = self.get(row_names=common_rows, col_names=common_cols)
                    second = other.get(row_names=common_rows, col_names=common_cols)
                else:
                    assert self.shape == other.shape, (
                        "Matrix.__sub__():shape mismatch: "
                        + str(self.shape)
                        + " "
                        + str(other.shape)
                    )
                    first = self
                    second = other

                if first.isdiagonal and second.isdiagonal:
                    return type(self)(
                        x=first.x - second.x,
                        isdiagonal=True,
                        row_names=first.row_names,
                        col_names=first.col_names,
                    )
                elif first.isdiagonal:
                    elem_sub = -1.0 * second.newx
                    for j in range(first.shape[0]):
                        elem_sub[j, j] += first.x[j, 0]
                    return type(self)(
                        x=elem_sub, row_names=first.row_names, col_names=first.col_names
                    )
                elif second.isdiagonal:
                    elem_sub = first.newx
                    for j in range(second.shape[0]):
                        elem_sub[j, j] -= second.x[j, 0]
                    return type(self)(
                        x=elem_sub, row_names=first.row_names, col_names=first.col_names
                    )
                else:
                    return type(self)(
                        x=first.x - second.x,
                        row_names=first.row_names,
                        col_names=first.col_names,
                    )

    def __add__(self, other):
        """Overload of numpy.ndarray.__add__() - elementwise addition.  Tries to speedup by checking for
            scalars of diagonal matrices on either side of operator

        Args:
            other : (`int`,`float`,`numpy.ndarray`,`Matrix`): the thing to add

        Returns:
            `Matrix`: the result of addition

        Note:
            if `Matrix` and other (if applicable) have `autoalign` set to `True`,
            both `Matrix` and `other` are aligned based on row and column names.
            If names are not common between the two, this may result in a smaller
            returned `Matrix`.  If no names are shared, an exception is raised.
            The addition of a scalar does not expand non-zero elements.

        Example::

            m1 = pyemu.Cov.from_parameter_data(pst)
            m1_plus_on1 = m1 + 1.0 #add 1.0 to all non-zero elements
            m2 = m1.copy()
            m2_plus_m1 = m1 + m2


        """
        if np.isscalar(other):
            return type(self)(
                x=self.x + other,
                row_names=self.row_names,
                col_names=self.col_names,
                isdiagonal=self.isdiagonal,
            )

        if isinstance(other, pd.DataFrame):
            other = Matrix.from_dataframe(other)

        if isinstance(other, np.ndarray):
            assert self.shape == other.shape, (
                "Matrix.__add__(): shape mismatch: "
                + str(self.shape)
                + " "
                + str(other.shape)
            )
            if self.isdiagonal:
                raise NotImplementedError(
                    "Matrix.__add__ not supported for" + "diagonal self"
                )
            else:
                return type(self)(
                    x=self.x + other, row_names=self.row_names, col_names=self.col_names
                )

        elif isinstance(other, Matrix):
            if self.autoalign and other.autoalign and not self.element_isaligned(other):
                common_rows = get_common_elements(self.row_names, other.row_names)
                common_cols = get_common_elements(self.col_names, other.col_names)
                if len(common_rows) == 0:
                    raise Exception("Matrix.__add__ error: no common rows")

                if len(common_cols) == 0:
                    raise Exception("Matrix.__add__ error: no common cols")

                first = self.get(row_names=common_rows, col_names=common_cols)
                second = other.get(row_names=common_rows, col_names=common_cols)
            else:
                assert self.shape == other.shape, (
                    "Matrix.__add__(): shape mismatch: "
                    + str(self.shape)
                    + " "
                    + str(other.shape)
                )
                first = self
                second = other
            if first.isdiagonal and second.isdiagonal:
                return type(self)(
                    x=first.x + second.x,
                    isdiagonal=True,
                    row_names=first.row_names,
                    col_names=first.col_names,
                )
            elif first.isdiagonal:
                ox = second.newx
                for j in range(first.shape[0]):
                    ox[j, j] += first.__x[j]
                return type(self)(
                    x=ox, row_names=first.row_names, col_names=first.col_names
                )
            elif second.isdiagonal:
                x = first.x
                js = range(second.shape[0])
                x[js, js] += second.x.ravel()
                # for j in range(second.shape[0]):
                #     x[j, j] += second._Matrix__x[j,0]
                return type(self)(
                    x=x, row_names=first.row_names, col_names=first.col_names
                )
            else:
                return type(self)(
                    x=first.x + second.x,
                    row_names=first.row_names,
                    col_names=first.col_names,
                )
        else:
            raise Exception(
                "Matrix.__add__(): unrecognized type for "
                + "other in __add__: "
                + str(type(other))
            )

    def hadamard_product(self, other):
        """Overload of numpy.ndarray.__mult__(): element-wise multiplication.
        Tries to speedup by checking for scalars of diagonal matrices on
        either side of operator

        Args:
            other : (`int`,`float`,`numpy.ndarray`,`Matrix`): the thing to multiply

        Returns:
            `Matrix`: the result of multiplication

        Note:
            if `Matrix` and other (if applicable) have `autoalign` set to `True`,
            both `Matrix` and `other` are aligned based on row and column names.
            If names are not common between the two, this may result in a smaller
            returned `Matrix`.  If not common elements are shared, an exception is raised


        Example::

            cov = pyemu.Cov.from_parameter_data(pst)
            cov2 = cov * 10
            cov3 = cov * cov2

        """
        if np.isscalar(other):
            return type(self)(x=self.x * other)

        if isinstance(other, pd.DataFrame):
            other = Matrix.from_dataframe(other)

        if isinstance(other, np.ndarray):
            if other.shape != self.shape:
                raise Exception(
                    "Matrix.hadamard_product(): shape mismatch: "
                    + str(self.shape)
                    + " "
                    + str(other.shape)
                )
            if self.isdiagonal:
                raise NotImplementedError(
                    "Matrix.hadamard_product() not supported for" + "diagonal self"
                )
            else:
                return type(self)(
                    x=self.x * other, row_names=self.row_names, col_names=self.col_names
                )
        elif isinstance(other, Matrix):
            if self.autoalign and other.autoalign and not self.element_isaligned(other):
                common_rows = get_common_elements(self.row_names, other.row_names)
                common_cols = get_common_elements(self.col_names, other.col_names)
                if len(common_rows) == 0:
                    raise Exception("Matrix.hadamard_product error: no common rows")

                if len(common_cols) == 0:
                    raise Exception("Matrix.hadamard_product error: no common cols")

                first = self.get(row_names=common_rows, col_names=common_cols)
                second = other.get(row_names=common_rows, col_names=common_cols)
            else:
                if other.shape != self.shape:
                    raise Exception(
                        "Matrix.hadamard_product(): shape mismatch: "
                        + str(self.shape)
                        + " "
                        + str(other.shape)
                    )
                first = self
                second = other

            if first.isdiagonal and second.isdiagonal:
                return type(self)(
                    x=first.x * second.x,
                    isdiagonal=True,
                    row_names=first.row_names,
                    col_names=first.col_names,
                )
            # elif first.isdiagonal:
            #     #ox = second.as_2d
            #     #for j in range(first.shape[0]):
            #     #    ox[j, j] *= first.__x[j]
            #     return type(self)(x=first.as_2d * second.as_2d, row_names=first.row_names,
            #                       col_names=first.col_names)
            # elif second.isdiagonal:
            #     #x = first.as_2d
            #     #for j in range(second.shape[0]):
            #     #    x[j, j] *= second.x[j]
            #     return type(self)(x=first.x * second.as_2d, row_names=first.row_names,
            #                       col_names=first.col_names)
            else:
                return type(self)(
                    x=first.as_2d * second.as_2d,
                    row_names=first.row_names,
                    col_names=first.col_names,
                )
        else:
            raise Exception(
                "Matrix.hadamard_product(): unrecognized type for "
                + "other: "
                + str(type(other))
            )

    def __mul__(self, other):
        """Dot product multiplication overload.  Tries to speedup by
        checking for scalars or diagonal matrices on either side of operator

        Args:
            other : (`int`,`float`,`numpy.ndarray`,`Matrix`): the thing to dot product

        Returns:
            `Matrix`: the result of dot product

        Note:
            if `Matrix` and other (if applicable) have `autoalign` set to `True`,
            both `Matrix` and `other` are aligned based on row and column names.
            If names are not common between the two, this may result in a smaller
            returned `Matrix`.  If not common elements are found, an exception is raised

        Example::

            jco = pyemu.Jco.from_binary("pest.jcb")
            cov = pyemu.Cov.from_parmaeter_data(pst)
            # get the forecast prior covariance matrix
            forecast_cov = jco.get(pst.forecast_names).T * cov * jco.get(pst.forecast_names)


        """

        if isinstance(other, pd.DataFrame):
            other = Matrix.from_dataframe(other)

        if np.isscalar(other):
            return type(self)(
                x=self.x.copy() * other,
                row_names=self.row_names,
                col_names=self.col_names,
                isdiagonal=self.isdiagonal,
            )

        elif isinstance(other, np.ndarray):

            if self.shape[1] != other.shape[0]:
                raise Exception(
                    "Matrix.__mul__(): matrices are not aligned: "
                    + str(self.shape)
                    + " "
                    + str(other.shape)
                )
            if self.isdiagonal:
                return type(self)(
                    x=np.dot(np.diag(self.__x.flatten()).transpose(), other)
                )
            else:
                return type(self)(x=np.atleast_2d(np.dot(self.__x, other)))
        elif isinstance(other, Matrix):
            if self.autoalign and other.autoalign and not self.mult_isaligned(other):
                common = get_common_elements(self.col_names, other.row_names)
                if len(common) == 0:
                    raise Exception(
                        "Matrix.__mult__():self.col_names "
                        + "and other.row_names"
                        + "don't share any common elements.  first 10: "
                        + ",".join(self.col_names[:9])
                        + "...and.."
                        + ",".join(other.row_names[:9])
                    )
                # these should be aligned
                if isinstance(self, Cov):
                    first = self.get(row_names=common, col_names=common)
                else:
                    first = self.get(row_names=self.row_names, col_names=common)
                if isinstance(other, Cov):
                    second = other.get(row_names=common, col_names=common)
                else:
                    second = other.get(row_names=common, col_names=other.col_names)

            else:
                if self.shape[1] != other.shape[0]:
                    raise Exception(
                        "Matrix.__mul__(): matrices are not aligned: "
                        + str(self.shape)
                        + " "
                        + str(other.shape)
                    )
                first = self
                second = other
            if first.isdiagonal and second.isdiagonal:
                elem_prod = type(self)(
                    x=first.x.transpose() * second.x,
                    row_names=first.row_names,
                    col_names=second.col_names,
                )
                elem_prod.isdiagonal = True
                return elem_prod
            elif first.isdiagonal:
                ox = second.newx
                for j in range(first.shape[0]):
                    ox[j, :] *= first.x[j]
                return type(self)(
                    x=ox, row_names=first.row_names, col_names=second.col_names
                )
            elif second.isdiagonal:
                x = first.newx
                ox = second.x
                for j in range(first.shape[1]):
                    x[:, j] *= ox[j]
                return type(self)(
                    x=x, row_names=first.row_names, col_names=second.col_names
                )
            else:
                return type(self)(
                    np.dot(first.x, second.x),
                    row_names=first.row_names,
                    col_names=second.col_names,
                )
        else:
            raise Exception(
                "Matrix.__mul__(): unrecognized "
                + "other arg type in __mul__: "
                + str(type(other))
            )

    def __rmul__(self, other):
        """Reverse order Dot product multiplication overload.

        Args:
            other : (`int`,`float`,`numpy.ndarray`,`Matrix`): the thing to dot product

        Returns:
            `Matrix`: the result of dot product

        Note:
            if `Matrix` and other (if applicable) have `autoalign` set to `True`,
            both `Matrix` and `other` are aligned based on row and column names.
            If names are not common between the two, this may result in a smaller
            returned `Matrix`.  If not common elements are found, an exception is raised

        Example::

            # multiply by a scalar
            jco = pyemu.Jco.from_binary("pest.jcb")
            jco_times_10 = 10 * jco

        """

        # if isinstance(other,pd.DataFrame):
        #     other = Matrix.from_dataframe(other)

        if np.isscalar(other):
            return type(self)(
                x=self.x.copy() * other,
                row_names=self.row_names,
                col_names=self.col_names,
                isdiagonal=self.isdiagonal,
            )

        elif isinstance(other, np.ndarray):
            if self.shape[0] != other.shape[1]:
                raise Exception(
                    "Matrix.__rmul__(): matrices are not aligned: "
                    + str(other.shape)
                    + " "
                    + str(self.shape)
                )
            if self.isdiagonal:
                return type(self)(
                    x=np.dot(other, np.diag(self.__x.flatten()).transpose())
                )
            else:
                return type(self)(x=np.dot(other, self.__x))
        elif isinstance(other, Matrix):
            if self.autoalign and other.autoalign and not self.mult_isaligned(other):
                common = get_common_elements(self.row_names, other.col_names)
                if len(common) == 0:
                    raise Exception(
                        "Matrix.__rmul__():self.col_names "
                        + "and other.row_names"
                        + "don't share any common elements"
                    )
                # these should be aligned
                if isinstance(self, Cov):
                    first = self.get(row_names=common, col_names=common)
                else:
                    first = self.get(col_names=self.row_names, row_names=common)
                if isinstance(other, Cov):
                    second = other.get(row_names=common, col_names=common)
                else:
                    second = other.get(col_names=common, row_names=other.col_names)

            else:
                if self.shape[0] != other.shape[1]:
                    raise Exception(
                        "Matrix.__rmul__(): matrices are not aligned: "
                        + str(other.shape)
                        + " "
                        + str(self.shape)
                    )
                first = other
                second = self
            if first.isdiagonal and second.isdiagonal:
                elem_prod = type(self)(
                    x=first.x.transpose() * second.x,
                    row_names=first.row_names,
                    col_names=second.col_names,
                )
                elem_prod.isdiagonal = True
                return elem_prod
            elif first.isdiagonal:
                ox = second.newx
                for j in range(first.shape[0]):
                    ox[j, :] *= first.x[j]
                return type(self)(
                    x=ox, row_names=first.row_names, col_names=second.col_names
                )
            elif second.isdiagonal:
                x = first.newx
                ox = second.x
                for j in range(first.shape[1]):
                    x[:, j] *= ox[j]
                return type(self)(
                    x=x, row_names=first.row_names, col_names=second.col_names
                )
            else:
                return type(self)(
                    np.dot(first.x, second.x),
                    row_names=first.row_names,
                    col_names=second.col_names,
                )
        else:
            raise Exception(
                "Matrix.__rmul__(): unrecognized "
                + "other arg type in __mul__: "
                + str(type(other))
            )

    def __set_svd(self):
        """private method to set SVD components.

        Note: this should not be called directly

        """
        if self.isdiagonal:
            x = np.diag(self.x.flatten())
        else:
            # just a pointer to x
            x = self.x
        try:

            u, s, v = np.linalg.svd(x, full_matrices=True)
            v = v.transpose()
        except Exception as e:
            print("standard SVD failed: {0}".format(str(e)))
            try:
                v, s, u = np.linalg.svd(x.transpose(), full_matrices=True)
                u = u.transpose()
            except Exception as e:
                np.savetxt("failed_svd.dat", x, fmt="%15.6E")
                raise Exception(
                    "Matrix.__set_svd(): "
                    + "unable to compute SVD of self.x, "
                    + "saved matrix to 'failed_svd.dat' -- {0}".format(str(e))
                )

        col_names = ["left_sing_vec_" + str(i + 1) for i in range(u.shape[1])]
        self.__u = Matrix(
            x=u, row_names=self.row_names, col_names=col_names, autoalign=False
        )

        sing_names = ["sing_val_" + str(i + 1) for i in range(s.shape[0])]
        self.__s = Matrix(
            x=np.atleast_2d(s).transpose(),
            row_names=sing_names,
            col_names=sing_names,
            isdiagonal=True,
            autoalign=False,
        )

        col_names = ["right_sing_vec_" + str(i + 1) for i in range(v.shape[0])]
        self.__v = Matrix(
            v, row_names=self.col_names, col_names=col_names, autoalign=False
        )

    def mult_isaligned(self, other):
        """check if matrices are aligned for dot product multiplication

        Args:
            other (`Matrix`): the other matrix to check for alignment with

        Returns:
            `bool`: True if aligned, False if not aligned

        """
        assert isinstance(
            other, Matrix
        ), "Matrix.isaligned(): other argumnent must be type Matrix, not: " + str(
            type(other)
        )
        if self.col_names == other.row_names:
            return True
        else:
            return False

    def element_isaligned(self, other):
        """check if matrices are aligned for element-wise operations

        Args:
            other (`Matrix`): the other matrix to check for alignment with

        Returns:
            `bool`: True if aligned, False if not aligned

        """
        if not isinstance(other, Matrix):
            raise Exception(
                "Matrix.isaligned(): other argument must be type Matrix, not: "
                + str(type(other))
            )
        if self.row_names == other.row_names and self.col_names == other.col_names:
            return True
        else:
            return False

    @property
    def newx(self):
        """return a copy of `Matrix.x` attribute

        Returns:
            `numpy.ndarray`: a copy `Matrix.x`

        """
        return self.__x.copy()

    @property
    def x(self):
        """return a reference to `Matrix.x`

        Returns:
            `numpy.ndarray`: reference to `Matrix.x`

        """
        return self.__x

    @property
    def as_2d(self):
        """get a 2D numeric representation of `Matrix.x`.  If not `isdiagonal`, simply
        return reference to `Matrix.x`, otherwise, constructs and returns
        a 2D, diagonal ndarray

        Returns:
            `numpy.ndarray` : numpy.ndarray

        Example::

            # A diagonal cov
            cov = pyemu.Cov.from_parameter_data
            x2d = cov.as_2d # a numpy 2d array
            print(cov.shape,cov.x.shape,x2d.shape)

        """
        if not self.isdiagonal:
            return self.x
        return np.diag(self.x.flatten())

    def to_2d(self):
        """get a 2D `Matrix` representation of `Matrix`.  If not `Matrix.isdiagonal`, simply
                return a copy of `Matrix`, otherwise, constructs and returns a new `Matrix`
                instance that is stored as diagonal

        Returns:
            `Martrix`: non-diagonal form of `Matrix`

        Example::

            # A diagonal cov
            cov = pyemu.Cov.from_parameter_data
            cov2d = cov.as_2d # a numpy 2d array
            print(cov.shape,cov.x.shape,cov2d.shape,cov2d.x.shape)

        """
        if not self.isdiagonal:
            return self.copy()
        return type(self)(
            x=np.diag(self.x.flatten()),
            row_names=self.row_names,
            col_names=self.col_names,
            isdiagonal=False,
        )

    @property
    def shape(self):
        """get the implied, 2D shape of `Matrix`

        Returns:
            `int`: length of 2 tuple

        Example::

            jco = pyemu.Jco.from_binary("pest.jcb")
            shape = jco.shape
            nrow,ncol = shape #unpack to ints

        """
        if self.__x is not None:
            if self.isdiagonal:
                return (max(self.__x.shape), max(self.__x.shape))
            if len(self.__x.shape) == 1:
                raise Exception("Matrix.shape: Matrix objects must be 2D")
            return self.__x.shape
        return None

    @property
    def ncol(self):
        """length of second dimension

        Returns:
            `int`: number of columns

        """
        return self.shape[1]

    @property
    def nrow(self):
        """length of first dimension

        Returns:
            `int`: number of rows

        """
        return self.shape[0]

    @property
    def T(self):
        """wrapper function for `Matrix.transpose()` method

        Returns:
            `Matrix`: transpose of `Matrix`

        Note:
            returns a copy of self

            A syntatic-sugar overload of Matrix.transpose()

        Example::

            jcb = pyemu.Jco.from_binary("pest.jcb")
            jcbt = jcb.T


        """
        return self.transpose

    @property
    def transpose(self):
        """transpose operation of self

        Returns:
            `Matrix`: transpose of `Matrix`

        Example::

            jcb = pyemu.Jco.from_binary("pest.jcb")
            jcbt = jcb.T

        """
        if not self.isdiagonal:
            return type(self)(
                x=self.__x.copy().transpose(),
                row_names=self.col_names,
                col_names=self.row_names,
                autoalign=self.autoalign,
            )
        else:
            return type(self)(
                x=self.__x.copy(),
                row_names=self.row_names,
                col_names=self.col_names,
                isdiagonal=True,
                autoalign=self.autoalign,
            )

    @property
    def inv(self):
        """inversion operation of `Matrix`

        Returns:
            `Matrix`: inverse of `Matrix`

        Note:
            uses `numpy.linalg.inv` for the inversion

        Example::

            mat = pyemu.Matrix.from_binary("my.jco")
            mat_inv = mat.inv
            mat_inv.to_binary("my_inv.jco")

        """

        if self.isdiagonal:
            inv = 1.0 / self.__x
            if np.any(~np.isfinite(inv)):
                idx = np.isfinite(inv)
                np.savetxt("testboo.dat", idx)
                invalid = [
                    self.row_names[i] for i in range(idx.shape[0]) if idx[i] == 0.0
                ]
                raise Exception(
                    "Matrix.inv has produced invalid floating points "
                    + " for the following elements:"
                    + ",".join(invalid)
                )
            return type(self)(
                x=inv,
                isdiagonal=True,
                row_names=self.row_names,
                col_names=self.col_names,
                autoalign=self.autoalign,
            )
        else:
            return type(self)(
                x=np.linalg.inv(self.__x),
                row_names=self.row_names,
                col_names=self.col_names,
                autoalign=self.autoalign,
            )

    @staticmethod
    def get_maxsing_from_s(s, eigthresh=1.0e-5):
        """static method to work out the maxsing for a
        given singular spectrum

        Args:
            s (`numpy.ndarray`): 1-D array of singular values. This
                array should come from calling either `numpy.linalg.svd`
                or from the `pyemu.Matrix.s.x` attribute
            eigthresh (`float`): the ratio of smallest to largest
                singular value to retain.  Since it is assumed that
                `s` is sorted from largest to smallest, once a singular value
                is reached that yields a ratio with the first (largest)
                singular value, the index of this singular is returned.

        Returns:
            `int`: the index of the singular value who's ratio with the
            first singular value is less than or equal to `eigthresh`


        Example::

            jco = pyemu.Jco.from_binary("pest.jco")
            max_sing = pyemu.Matrix.get_maxsing_from_s(jco.s,eigthresh=pst.svd_data.eigthresh)

        """
        sthresh = s.flatten() / s[0]
        ising = 0
        for st in sthresh:
            if st > eigthresh:
                ising += 1
            else:
                break
        return max(1, ising)

    def get_maxsing(self, eigthresh=1.0e-5):
        """Get the number of singular components with a singular
        value ratio greater than or equal to eigthresh

         Args:
            eigthresh (`float`): the ratio of smallest to largest
                singular value to retain.  Since it is assumed that
                `s` is sorted from largest to smallest, once a singular value
                is reached that yields a ratio with the first (largest)
                singular value, the index of this singular is returned.

        Returns:
            `int`: the index of the singular value who's ratio with the
            first singular value is less than or equal to `eigthresh`

        Note:
            this method calls the static method `Matrix.get_maxsing_from_s()`
            with `Matrix.s.x`

        Example::

            jco = pyemu.Jco.from_binary("pest.jco")
            max_sing = jco.get_maxsing(eigthresh=pst.svd_data.eigthresh)

        """

        return Matrix.get_maxsing_from_s(self.s.x, eigthresh=eigthresh)

    def pseudo_inv_components(self, maxsing=None, eigthresh=1.0e-5, truncate=True):
        """Get the (optionally) truncated SVD components

        Args:
            maxsing (`int`, optional): the number of singular components to use.  If None,
                `maxsing` is calculated using `Matrix.get_maxsing()` and `eigthresh`
            `eigthresh` : (`float`, optional): the ratio of largest to smallest singular
                components to use for truncation.  Ignored if maxsing is not None.  Default is
                1.0e-5
            truncate (`bool`): flag to truncate components. If False, U, s, and V will be
                zeroed out at locations greater than `maxsing` instead of truncated. Default is True

        Returns:
            tuple containing

            - **Matrix**: (optionally truncated) left singular vectors
            - **Matrix**: (optionally truncated) singular value matrix
            - **Matrix**: (optionally truncated) right singular vectors

        Example::

            mat = pyemu.Matrix.from_binary("my.jco")
            u1,s1,v1 = mat.pseudo_inv_components(maxsing=10)
            resolution_matrix = v1 * v1.T
            resolution_matrix.to_ascii("resol.mat")

        """

        if maxsing is None:
            maxsing = self.get_maxsing(eigthresh=eigthresh)
        else:
            maxsing = min(self.get_maxsing(eigthresh=eigthresh), maxsing)

        s = self.full_s.copy()
        v = self.v.copy()
        u = self.u.copy()
        if truncate:

            s = s[:maxsing, :maxsing]
            v = v[:, :maxsing]
            u = u[:, :maxsing]
        else:
            new_s = self.full_s.copy()
            s = new_s
            s.x[maxsing:, maxsing:] = 0.0
            v.x[:, maxsing:] = 0.0
            u.x[:, maxsing:] = 0.0

        return u, s, v

    def pseudo_inv(self, maxsing=None, eigthresh=1.0e-5):
        """The pseudo inverse of self.  Formed using truncated singular
        value decomposition and `Matrix.pseudo_inv_components`

        Args:
            maxsing (`int`, optional): the number of singular components to use.  If None,
                `maxsing` is calculated using `Matrix.get_maxsing()` and `eigthresh`
            `eigthresh` : (`float`, optional): the ratio of largest to smallest singular
                components to use for truncation.  Ignored if maxsing is not None.  Default is
                1.0e-5

        Returns:
              `Matrix`: the truncated-SVD pseudo inverse of `Matrix` (V_1 * s_1^-1 * U^T)

        Example::

            jco = pyemu.Jco.from_binary("pest.jcb")
            jco_psinv = jco.pseudo_inv(pst.svd_data.maxsing,pst.svd_data.eigthresh)
            jco_psinv.to_binary("pseudo_inv.jcb")
        """
        if maxsing is None:
            maxsing = self.get_maxsing(eigthresh=eigthresh)
        full_s = self.full_s.T
        for i in range(self.s.shape[0]):
            if i <= maxsing:
                full_s.x[i, i] = 1.0 / full_s.x[i, i]
            else:
                full_s.x[i, i] = 0.0
        return self.v * full_s * self.u.T

    @property
    def sqrt(self):
        """element-wise square root operation

        Returns:
            `Matrix`: element-wise square root of `Matrix`

        Note:
            uses `numpy.sqrt`

        Example::

            cov = pyemu.Cov.from_uncfile("my.unc")
            sqcov = cov.sqrt
            sqcov.to_uncfile("sqrt_cov.unc")


        """
        if self.isdiagonal:
            return type(self)(
                x=np.sqrt(self.__x),
                isdiagonal=True,
                row_names=self.row_names,
                col_names=self.col_names,
                autoalign=self.autoalign,
            )
        elif self.shape[1] == 1:  # a vector
            return type(self)(
                x=np.sqrt(self.__x),
                isdiagonal=False,
                row_names=self.row_names,
                col_names=self.col_names,
                autoalign=self.autoalign,
            )
        else:
            return type(self)(
                x=np.sqrt(self.__x),
                row_names=self.row_names,
                col_names=self.col_names,
                autoalign=self.autoalign,
            )

    @property
    def full_s(self):
        """Get the full singular value matrix

        Returns:
            `Matrix`: full singular value matrix.  Shape is `(max(Matrix.shape),max(Matrix.shape))`
            with zeros along the diagonal from `min(Matrix.shape)` to `max(Matrix.shape)`

        Example::

            jco = pyemu.Jco.from_binary("pest.jcb")
            jco.full_s.to_ascii("full_sing_val_matrix.mat")

        """
        x = np.zeros((self.shape), dtype=np.float32)

        x[: self.s.shape[0], : self.s.shape[0]] = self.s.as_2d
        s = Matrix(
            x=x,
            row_names=self.row_names,
            col_names=self.col_names,
            isdiagonal=False,
            autoalign=False,
        )
        return s

    @property
    def s(self):
        """the singular value (diagonal) Matrix

        Returns:
            `Matrix`: singular value matrix.  shape is `(min(Matrix.shape),min(Matrix.shape))`

        Example::

            # plot the singular spectrum of the jacobian
            import matplotlib.pyplot as plt
            jco = pyemu.Jco.from_binary("pest.jcb")
            plt.plot(jco.s.x)
            plt.show()

        """
        if self.__s is None:
            self.__set_svd()
        return self.__s

    @property
    def u(self):
        """the left singular vector Matrix

        Returns:
            `Matrix`: left singular vectors.  Shape is `(Matrix.shape[0], Matrix.shape[0])`

        Example::

            jco = pyemu.Jco.from_binary("pest.jcb")
            jco.u.to_binary("u.jcb")


        """
        if self.__u is None:
            self.__set_svd()
        return self.__u

    @property
    def v(self):
        """the right singular vector Matrix

        Returns:
            `Matrix`: right singular vectors.  Shape is `(Matrix.shape[1], Matrix.shape[1])`

        Example::

            jco = pyemu.Jco.from_binary("pest.jcb")
            jco.v.to_binary("v.jcb")

        """
        if self.__v is None:
            self.__set_svd()
        return self.__v

    @property
    def zero2d(self):
        """get an 2D instance of self with all zeros

        Returns:
            `Matrix`: `Matrix of zeros`

        Example::

            jco = pyemu.Jco.from_binary("pest.jcb")
            zero_jco = jco.zero2d


        """
        return type(self)(
            x=np.atleast_2d(np.zeros((self.shape[0], self.shape[1]))),
            row_names=self.row_names,
            col_names=self.col_names,
            isdiagonal=False,
        )

    @staticmethod
    def find_rowcol_indices(names, row_names, col_names, axis=None):
        """fast(er) look of row and column names indices

        Args:
            names ([`str`]): list of names to look for in `row_names` and/or `col_names` names
            row_names ([`str`]): list of row names
            col_names([`str`]): list of column names
            axis (`int`, optional): axis to search along.  If None, search both.
                Default is `None`

        Returns:
            `numpy.ndarray`: array of (integer) index locations.  If `axis` is
            `None`, a 2 `numpy.ndarrays` of both row and column name indices is returned

        """

        self_row_idxs = {row_names[i]: i for i in range(len(row_names))}
        self_col_idxs = {col_names[i]: i for i in range(len(col_names))}

        scol = set(col_names)
        srow = set(row_names)
        row_idxs = []
        col_idxs = []
        for name in names:
            name = name.lower()
            if name not in scol and name not in srow:
                raise Exception("Matrix.indices(): name not found: " + name)
            if name in scol:
                col_idxs.append(self_col_idxs[name])
            if name.lower() in srow:
                row_idxs.append(self_row_idxs[name])
        if axis is None:
            return (
                np.array(row_idxs, dtype=np.int32),
                np.array(col_idxs, dtype=np.int32),
            )
        elif axis == 0:
            if len(row_idxs) != len(names):
                raise Exception(
                    "Matrix.indices(): " + "not all names found in row_names"
                )
            return np.array(row_idxs, dtype=np.int32)
        elif axis == 1:
            if len(col_idxs) != len(names):
                raise Exception(
                    "Matrix.indices(): " + "not all names found in col_names"
                )
            return np.array(col_idxs, dtype=np.int32)
        else:
            raise Exception(
                "Matrix.indices(): " + "axis argument must 0 or 1, not:" + str(axis)
            )

    def indices(self, names, axis=None):
        """get the row and col indices of names. If axis is None, two ndarrays
                are returned, corresponding the indices of names for each axis

        Args:
            names ([`str`]): list of names to look for in `row_names` and/or `col_names` names
            row_names ([`str`]): list of row names
            col_names([`str`]): list of column names
            axis (`int`, optional): axis to search along.  If None, search both.
                Default is `None`

        Returns:
            `numpy.ndarray`: array of (integer) index locations.  If `axis` is
            `None`, a 2 `numpy.ndarrays` of both row and column name indices is returned

        Note:
            thin wrapper around `Matrix.find_rowcol_indices` static method

        """
        return Matrix.find_rowcol_indices(
            names, self.row_names, self.col_names, axis=axis
        )

    def align(self, names, axis=None):
        """reorder `Matrix` by names in place.  If axis is None, reorder both indices

        Args:
            names (['str']): names in `Matrix.row_names` and or `Matrix.col_names`
            axis (`int`, optional): the axis to reorder. if None, reorder both axes

        Note:
              Works in place.
              Is called programmatically during linear algebra operations

        Example::

            # load a jco that has more obs (rows) than non-zero weighted obs
            # in the control file
            jco = pyemu.Jco.from_binary("pest.jcb")
            # get an obs noise cov matrix
            obscov = pyemu.Cov.from_observation_data(pst)
            jco.align(obscov.row_names,axis=0)

        """
        if not isinstance(names, list):
            names = [names]
        row_idxs, col_idxs = self.indices(names)
        if self.isdiagonal or isinstance(self, Cov):
            assert row_idxs.shape[0] == self.shape[0]
            if row_idxs.shape != col_idxs.shape:
                raise Exception("shape mismatch")
            if row_idxs.shape[0] != self.shape[0]:
                raise Exception("shape mismatch")

            if self.isdiagonal:
                self.__x = self.__x[row_idxs]
            else:
                self.__x = self.__x[row_idxs, :]
                self.__x = self.__x[:, col_idxs]
            row_names = []
            _ = [row_names.append(self.row_names[i]) for i in row_idxs]
            self.row_names, self.col_names = row_names, row_names

        else:
            if axis is None:
                raise Exception(
                    "Matrix.align(): must specify axis in "
                    + "align call for non-diagonal instances"
                )
            if axis == 0:
                if row_idxs.shape[0] != self.shape[0]:
                    raise Exception(
                        "Matrix.align(): not all names found in self.row_names"
                    )
                self.__x = self.__x[row_idxs, :]
                row_names = []
                _ = [row_names.append(self.row_names[i]) for i in row_idxs]
                self.row_names = row_names
            elif axis == 1:
                if col_idxs.shape[0] != self.shape[1]:
                    raise Exception(
                        "Matrix.align(): not all names found in self.col_names"
                    )
                self.__x = self.__x[:, col_idxs]
                col_names = []
                _ = [col_names.append(self.col_names[i]) for i in row_idxs]
                self.col_names = col_names
            else:
                raise Exception(
                    "Matrix.align(): axis argument to align()"
                    + " must be either 0 or 1"
                )

    def get(self, row_names=None, col_names=None, drop=False):
        """get a new `Matrix` instance ordered on row_names or col_names

        Args:
            row_names (['str'], optional): row_names for new Matrix.  If `None`,
                all row_names are used.
            col_names (['str'], optional): col_names for new Matrix. If `None`,
                all col_names are used.
            drop (`bool`): flag to remove row_names and/or col_names from this `Matrix`

        Returns:
            `Matrix`: a new `Matrix`

        Example::
            # load a jco that has more obs (rows) than non-zero weighted obs
            # in the control file
            jco = pyemu.Jco.from_binary("pest.jcb")
            # get an obs noise cov matrix
            obscov = pyemu.Cov.from_observation_data(pst)
            nnz_jco = jco.get(row_names = obscov.row_names)


        """
        if row_names is None and col_names is None:
            raise Exception(
                "Matrix.get(): must pass at least" + " row_names or col_names"
            )

        if row_names is not None and not isinstance(row_names, list):
            row_names = [row_names]
        if col_names is not None and not isinstance(col_names, list):
            col_names = [col_names]

        if isinstance(self, Cov) and (row_names is None or col_names is None):
            if row_names is not None:
                idxs = self.indices(row_names, axis=0)
                names = row_names
            else:
                idxs = self.indices(col_names, axis=1)
                names = col_names

            if self.isdiagonal:
                extract = self.__x[idxs].copy()
            else:
                extract = self.__x[idxs, :].copy()
                extract = extract[:, idxs]
            if drop:
                self.drop(names, 0)
            return Cov(x=extract, names=names, isdiagonal=self.isdiagonal)
        if self.isdiagonal:
            extract = np.diag(self.__x[:, 0])
        else:
            extract = self.__x.copy()
        if row_names is not None:
            row_idxs = self.indices(row_names, axis=0)
            extract = np.atleast_2d(extract[row_idxs, :].copy())
            if drop:
                self.drop(row_names, axis=0)
        else:
            row_names = self.row_names
        if col_names is not None:
            col_idxs = self.indices(col_names, axis=1)
            extract = np.atleast_2d(extract[:, col_idxs].copy())
            if drop:
                self.drop(col_names, axis=1)
        else:
            col_names = copy.deepcopy(self.col_names)

        return type(self)(x=extract, row_names=row_names, col_names=col_names)

    def copy(self):
        """get a copy of `Matrix`

        Returns:
            `Matrix`: copy of this `Matrix`

        """
        return type(self)(
            x=self.newx,
            row_names=self.row_names,
            col_names=self.col_names,
            isdiagonal=self.isdiagonal,
            autoalign=self.autoalign,
        )

    def drop(self, names, axis):
        """drop elements from `Matrix` in place

        Args:
            names (['str']): list of names to drop
            axis (`int`): the axis to drop from. must be in [0,1]

        """
        if axis is None:
            raise Exception("Matrix.drop(): axis arg is required")
        if not isinstance(names, list):
            names = [names]
        if axis == 1:
            if len(names) >= self.shape[1]:
                raise Exception("can't drop all names along axis 1")
        else:
            if len(names) >= self.shape[0]:
                raise Exception("can't drop all names along axis 0")

        idxs = self.indices(names, axis=axis)

        if self.isdiagonal:
            self.__x = np.delete(self.__x, idxs, 0)
            keep_names = [name for name in self.row_names if name not in names]
            if len(keep_names) != self.__x.shape[0]:
                raise Exception(
                    "shape-name mismatch:"
                    + "{0}:{0}".format(len(keep_names), self.__x.shape)
                )
            self.row_names = keep_names
            self.col_names = copy.deepcopy(keep_names)
            # idxs = np.sort(idxs)
            # for idx in idxs[::-1]:
            #     del self.row_names[idx]
            #     del self.col_names[idx]
        elif isinstance(self, Cov):
            self.__x = np.delete(self.__x, idxs, 0)
            self.__x = np.delete(self.__x, idxs, 1)
            keep_names = [name for name in self.row_names if name not in names]

            if len(keep_names) != self.__x.shape[0]:
                raise Exception(
                    "shape-name mismatch:"
                    + "{0}:{0}".format(len(keep_names), self.__x.shape)
                )
            self.row_names = keep_names
            self.col_names = copy.deepcopy(keep_names)
            # idxs = np.sort(idxs)
            # for idx in idxs[::-1]:
            #     del self.row_names[idx]
            #     del self.col_names[idx]
        elif axis == 0:
            if idxs.shape[0] == self.shape[0]:
                raise Exception("Matrix.drop(): can't drop all rows")
            elif idxs.shape == 0:
                raise Exception("Matrix.drop(): nothing to drop on axis 0")
            self.__x = np.delete(self.__x, idxs, 0)
            keep_names = [name for name in self.row_names if name not in names]
            if len(keep_names) != self.__x.shape[0]:
                raise Exception(
                    "shape-name mismatch:"
                    + "{0}:{1}".format(len(keep_names), self.__x.shape)
                )
            self.row_names = keep_names
            # idxs = np.sort(idxs)
            # for idx in idxs[::-1]:
            #     del self.row_names[idx]
        elif axis == 1:
            if idxs.shape[0] == self.shape[1]:
                raise Exception("Matrix.drop(): can't drop all cols")
            if idxs.shape == 0:
                raise Exception("Matrix.drop(): nothing to drop on axis 1")
            self.__x = np.delete(self.__x, idxs, 1)
            keep_names = [name for name in self.col_names if name not in names]
            if len(keep_names) != self.__x.shape[1]:
                raise Exception(
                    "shape-name mismatch:"
                    + "{0}:{1}".format(len(keep_names), self.__x.shape)
                )
            self.col_names = keep_names
            # idxs = np.sort(idxs)
            # for idx in idxs[::-1]:
            #     del self.col_names[idx]
        else:
            raise Exception("Matrix.drop(): axis argument must be 0 or 1")

    def extract(self, row_names=None, col_names=None):
        """wrapper method that `Matrix.gets()` then `Matrix.drops()` elements.
        one of row_names or col_names must be not None.

        Args:
            row_names (['str'], optional): row_names to extract.  If `None`,
                all row_names are retained.
            col_names (['str'], optional): col_names to extract. If `None`,
                all col_names are retained.

        Returns:
            `Matrix`: the extract sub-matrix defined by `row_names` and/or `col_names`

        Example::

            cov = pyemu.Cov.from_parameter_data(pst)
            hk_cov = cov.extract(row_names=["hk1","hk2","hk3"])

        """
        if row_names is None and col_names is None:
            raise Exception("Matrix.extract() " + "row_names and col_names both None")
        extract = self.get(row_names, col_names, drop=True)
        return extract

    def get_diagonal_vector(self, col_name="diag"):
        """Get a new Matrix instance that is the diagonal of self.  The
        shape of the new matrix is (self.shape[0],1).  Self must be square

        Args:
            col_name (`str`): the name of the single column in the new Matrix

        Returns:
            `Matrix`: vector-shaped `Matrix` instance of the diagonal of this `Matrix`

        Example::

            cov = pyemu.Cov.from_unc_file("my.unc")
            cov_diag = cov.get_diagonal_vector()
            print(cov_diag.col_names)

        """
        if self.shape[0] != self.shape[1]:
            raise Exception("not diagonal")
        if self.isdiagonal:
            raise Exception("already diagonal")
        if not isinstance(col_name, str):
            raise Exception("col_name must be type str")
        return type(self)(
            x=np.atleast_2d(np.diag(self.x)).transpose(),
            row_names=self.row_names,
            col_names=[col_name],
            isdiagonal=False,
        )

    def to_coo(self, filename, droptol=None, chunk=None):
        """write an extended PEST-format binary file.  The data format is
        [int,int,float] for i,j,value.  It is autodetected during
        the read with `Matrix.from_binary()`.

        Args:
            filename (`str` or `Path`): filename to save binary file
            droptol (`float`): absolute value tolerance to make values
                smaller `droptol` than zero.  Default is None (no dropping)
            chunk (`int`): number of elements to write in a single pass.
                Default is `None`, which writes the entire numeric part of the
                `Matrix` at once. This is faster but requires more memory.

        Note:
            This method is needed when the number of dimensions times 2 is larger
            than the max value for a 32-bit integer.  happens!
            This method is used by pyemu.Ensemble.to_binary()

        """
        if self.isdiagonal:
            # raise NotImplementedError()
            self.__x = self.as_2d
            self.isdiagonal = False
        if droptol is not None:
            self.x[np.abs(self.x) < droptol] = 0.0
        f = open(filename, "wb")
        # print("counting nnz")
        nnz = np.count_nonzero(self.x)  # number of non-zero entries
        # write the header
        header = np.array(
            (self.shape[1], self.shape[0], nnz), dtype=self.binary_header_dt
        )
        header.tofile(f)
        # get the indices of non-zero entries
        # print("getting nnz idxs")
        row_idxs, col_idxs = np.nonzero(self.x)

        if chunk is None:
            flat = self.x[row_idxs, col_idxs].flatten()
            data = np.rec.fromarrays(
                [row_idxs, col_idxs, flat], dtype=self.coo_rec_dt
            )
            data.tofile(f)
        else:

            start, end = 0, min(chunk, row_idxs.shape[0])
            while True:
                # print(row_idxs[start],row_idxs[end])
                # print("chunk",start,end)
                flat = self.x[row_idxs[start:end], col_idxs[start:end]].flatten()
                data = np.rec.fromarrays(
                    [row_idxs[start:end], col_idxs[start:end], flat],
                    dtype=self.coo_rec_dt,
                )
                data.tofile(f)
                if end == row_idxs.shape[0]:
                    break
                start = end
                end = min(row_idxs.shape[0], start + chunk)

        for name in self.col_names:
            if len(name) > self.new_par_length:
                warnings.warn(
                    "par name '{0}' greater than {1} chars".format(
                        name, self.new_par_length
                    )
                )
                name = name[: self.new_par_length - 1]
            elif len(name) < self.new_par_length:
                for _ in range(len(name), self.new_par_length):
                    name = name + " "
            f.write(name.encode())
        for name in self.row_names:
            if len(name) > self.new_obs_length:
                warnings.warn(
                    "obs name '{0}' greater than {1} chars".format(
                        name, self.new_obs_length
                    )
                )
                name = name[: self.new_obs_length - 1]
            elif len(name) < self.new_obs_length:
                for _ in range(len(name), self.new_obs_length):
                    name = name + " "
            f.write(name.encode())
        f.close()

    def to_dense(self, filename, close=True):
        """experimental new dense matrix storage format to support faster I/O with ensembles

        Args:
            filename (`str`): the filename to save to
            close (`bool`): flag to close the filehandle after saving

        Returns:
            f (`file`): the file handle.  Only returned if `close` is False

        Note:
            calls Matrix.write_dense()


        """
        return Matrix.write_dense(
            filename,
            row_names=self.row_names,
            col_names=self.col_names,
            data=self.x,
            close=close,
        )

    @staticmethod
    def write_dense(filename, row_names, col_names, data, close=False):
        """experimental new dense matrix storage format to support faster I/O with ensembles

        Args:
            filename (`str` or `file`): the file to write to
            row_names ([`str`]): row names of the matrix
            col_names ([`str`]): col names of the matrix
            data (`np.ndarray`): matrix elements
            close (`bool`): flag to close the file after writing

        Returns:
            f (`file`): the file handle.  Only returned if `close` is False

        """
        row_names = [str(r) for r in row_names]
        col_names = [str(c) for c in col_names]
        if isinstance(filename, (str, Path)):
            f = open(filename, "wb")
            header = np.array(
                (0, -len(col_names), -len(col_names)), dtype=Matrix.binary_header_dt
            )
            header.tofile(f)
            slengths = np.array(
                [len(col_name) for col_name in col_names], dtype=Matrix.integer
            )
            slengths.tofile(f)
            for i, col_name in enumerate(col_names):
                # slengths[[i]].tofile(f)
                f.write(col_name.encode())
        else:
            f = filename
        slengths = np.array(
            [len(row_name) for row_name in row_names], dtype=Matrix.integer
        )
        for i in range(data.shape[0]):
            slengths[[i]].tofile(f)
            f.write(row_names[i].encode())
            data[i, :].astype(Matrix.double).tofile(f)
        if close:
            f.close()
        else:
            return f

    def to_binary(self, filename, droptol=None, chunk=None):
        """write a PEST-compatible binary file.  The format is the same
        as the format used to storage a PEST Jacobian matrix

        Args:
            filename (`str`): filename to save binary file
            droptol (`float`): absolute value tolerance to make values
                smaller `droptol` than zero.  Default is None (no dropping)
            chunk (`int`): number of elements to write in a single pass.
                Default is `None`, which writes the entire numeric part of the
                `Matrix` at once. This is faster but requires more memory.

        """
        # print(self.x)
        # print(type(self.x))

        if np.any(np.isnan(self.x)):
            raise Exception("Matrix.to_binary(): nans found")
        if self.isdiagonal:
            # raise NotImplementedError()
            self.__x = self.as_2d
            self.isdiagonal = False
        if droptol is not None:
            self.x[np.abs(self.x) < droptol] = 0.0
        f = open(filename, "wb")
        nnz = np.count_nonzero(self.x)  # number of non-zero entries
        # write the header
        header = np.array(
            (-self.shape[1], -self.shape[0], nnz), dtype=self.binary_header_dt
        )
        header.tofile(f)
        # get the indices of non-zero entries
        row_idxs, col_idxs = np.nonzero(self.x)
        icount = row_idxs + 1 + col_idxs * self.shape[0]
        # flatten the array
        # flat = self.x[row_idxs, col_idxs].flatten()
        # zip up the index position and value pairs
        # data = np.array(list(zip(icount, flat)), dtype=self.binary_rec_dt)

        if chunk is None:
            flat = self.x[row_idxs, col_idxs].flatten()
            data = np.rec.fromarrays([icount, flat], dtype=self.binary_rec_dt)
            # write
            data.tofile(f)
        else:
            start, end = 0, min(chunk, row_idxs.shape[0])
            while True:
                # print(row_idxs[start],row_idxs[end])
                flat = self.x[row_idxs[start:end], col_idxs[start:end]].flatten()
                data = np.rec.fromarrays(
                    [icount[start:end], flat], dtype=self.binary_rec_dt
                )
                data.tofile(f)
                if end == row_idxs.shape[0]:
                    break
                start = end
                end = min(row_idxs.shape[0], start + chunk)

        for name in self.col_names:
            if len(name) > self.par_length:
                warnings.warn(
                    "par name '{0}' greater than {1} chars".format(
                        name, self.par_length
                    )
                )
                name = name[: self.par_length]
            elif len(name) < self.par_length:
                for i in range(len(name), self.par_length):
                    name = name + " "
            f.write(name.encode())
        for name in self.row_names:
            if len(name) > self.obs_length:
                warnings.warn(
                    "obs name '{0}' greater than {1} chars".format(
                        name, self.obs_length
                    )
                )
                name = name[: self.obs_length]
            elif len(name) < self.obs_length:
                for i in range(len(name), self.obs_length):
                    name = name + " "
            f.write(name.encode())

        f.close()

    @staticmethod
    def read_dense(filename, forgive=False, close=True, only_rows=None):
        """read a dense-format binary file.

        Args:
            filename (`str`): the filename or the open filehandle
            forgive (`bool`): flag to forgive incomplete records.  If True and
                an incomplete record is encountered, only the previously read
                records are returned.  If False, an exception is raised for an
                incomplete record
            close (`bool`): flag to close the filehandle.  Default is True
            only_rows (`iterable`): rows to read.  If None, all rows are read

        Returns:
            tuple containing

            - **numpy.ndarray**: the numeric values in the file
            - **['str']**: list of row names
            - **[`str`]**: list of col_names


        """
        if not os.path.exists(filename):
            raise Exception(
                "Matrix.read_dense(): filename '{0}' not found".format(filename)
            )
        if isinstance(filename, str):
            f = open(filename, "rb")
        else:
            f = filename

        col_names = []
        row_names = []
        data_rows = []

        row_names,row_offsets,col_names,success = Matrix.get_dense_binary_info(filename)
        if not forgive and not success:
            raise Exception("Matrix.read_dense(): error reading dense binary info")
        if only_rows is not None:
            missing = list(set(only_rows)-set(row_names))
            if len(missing) > 0:
                raise Exception("the following only_rows are missing:{0}".format(",".join(missing)))
            only_offsets = [row_offsets[row_names.index(only_row)] for only_row in only_rows]
            row_names = only_rows
            row_offsets = only_offsets
        ncol = len(col_names)

        i = 0
        for row_name,offset in zip(row_names,row_offsets):
            f.seek(offset)
            try:
                slen = np.fromfile(f, Matrix.integer, 1)[0]
            except Exception as e:
                break
            try:
                name = (
                    struct.unpack(str(slen) + "s", f.read(slen))[0]
                    .strip()
                    .lower()
                    .decode()
                )

                data_row = np.fromfile(f, Matrix.double, ncol)
                if data_row.shape[0] != ncol:
                    raise Exception(
                        "incomplete data in row {0}: {1} vs {2}".format(
                            i, data_row.shape[0], ncol
                        )
                    )
            except Exception as e:
                if forgive:
                    print("error reading row {0}: {1}".format(i, str(e)))
                    break
                else:
                    raise Exception("error reading row {0}: {1}".format(i, str(e)))
            data_rows.append(data_row)
            i += 1

        data_rows = np.array(data_rows)
        if close:
            f.close()
        return data_rows, row_names, col_names



    @staticmethod
    def get_dense_binary_info(filename):
        """read the header and row and offsets for a dense binary file.

        Parameters
        ----------
            filename (`str`): dense binary filename


        Returns:
            tuple containing

            - **['str']**: list of row names
            - **['int']**: list of row offsets
            - **[`str`]**: list of col names
            - **bool**: flag indicating successful reading of all records found


        """
        if not os.path.exists(filename):
            raise Exception(
                "Matrix.read_dense(): filename '{0}' not found".format(filename)
            )
        if isinstance(filename, str):
            f = open(filename, "rb")
        else:
            f = filename
        # the header datatype
        itemp1, itemp2, icount = np.fromfile(f, Matrix.binary_header_dt, 1)[0]
        # print(itemp1,itemp2,icount)
        if itemp1 != 0:
            raise Exception("Matrix.read_dense() itemp1 != 0")
        if itemp2 != icount:
            raise Exception("Matrix.read_dense() itemp2 != icount")
        ncol = np.abs(itemp2)
        col_slens = np.fromfile(f, Matrix.integer, ncol)
        i = 0
        col_names = []
        for slen in col_slens:
            name = (
                struct.unpack(str(slen) + "s", f.read(slen))[0].strip().lower().decode()
            )
            col_names.append(name)
        row_names = []
        row_offsets = []
        data_len = np.array(1,dtype=Matrix.double).itemsize * ncol
        success = True
        while True:
            curr_pos = f.tell()
            try:
                slen = np.fromfile(f, Matrix.integer, 1)[0]
            except Exception as e:
                break
            try:
                name = (
                    struct.unpack(str(slen) + "s", f.read(slen))[0]
                    .strip()
                    .lower()
                    .decode()
                )
            except Exception as e:
                    print("error reading row name {0}: {1}".format(i, str(e)))
                    success = False
                    break
            try:
                data_row = np.fromfile(f, Matrix.double, ncol)
                if data_row.shape[0] != ncol:
                    raise Exception(
                        "incomplete data in row {0}: {1} vs {2}".format(
                            i, data_row.shape[0], ncol))
            except Exception as e:
                print("error reading row data record {0}: {1}".format(i, str(e)))
                success = False
                break

            row_offsets.append(curr_pos)
            row_names.append(name)

            i += 1
        f.close()
        return row_names,row_offsets,col_names,success


    @classmethod
    def from_binary(cls, filename, forgive=False):
        """class method load from PEST-compatible binary file into a
        Matrix instance

        Args:
            filename (`str`): filename to read
            forgive (`bool`): flag to forgive incomplete data records. Only
                applicable to dense binary format.  Default is `False`

        Returns:
            `Matrix`: `Matrix` loaded from binary file

        Example::

            mat = pyemu.Matrix.from_binary("my.jco")
            cov = pyemu.Cov.from_binary("large_cov.jcb")

        """
        x, row_names, col_names = Matrix.read_binary(filename, forgive=forgive)
        if np.any(np.isnan(x)):
            warnings.warn("Matrix.from_binary(): nans in matrix", PyemuWarning)
        return cls(x=x, row_names=row_names, col_names=col_names)

    @staticmethod
    def read_binary_header(filename):
        """read the first elements of a PEST(++)-style binary file to get
        format and dimensioning information.

        Args:
            filename (`str`): the filename of the binary file

        Returns:
            tuple containing

            - **int**: the itemp1 value
            - **int**: the itemp2 value
            - **int**: the icount value

        """
        if not os.path.exists(filename):
            raise Exception(
                "Matrix.read_binary_header(): filename '{0}' not found".format(filename)
            )
        f = open(filename, "rb")
        itemp1, itemp2, icount = np.fromfile(f, Matrix.binary_header_dt, 1)[0]
        f.close()
        return itemp1, itemp2, icount

    @staticmethod
    def read_binary(filename, forgive=False):
        """static method to read PEST-format binary files

        Args:
            filename (`str`): filename to read
            forgive (`bool`): flag to forgive incomplete data records. Only
                applicable to dense binary format.  Default is `False`


        Returns:
            tuple containing

            - **numpy.ndarray**: the numeric values in the file
            - **['str']**: list of row names
            - **[`str`]**: list of col_names

        """
        if not os.path.exists(filename):
            raise Exception(
                "Matrix.read_binary(): filename '{0}' not found".format(filename)
            )
        f = open(filename, "rb")
        itemp1, itemp2, icount = np.fromfile(f, Matrix.binary_header_dt, 1)[0]
        if itemp1 > 0 and itemp2 < 0 and icount < 0:
            print(
                " WARNING: it appears this file was \n"
                + " written with 'sequential` "
                + " binary fortran specification\n...calling "
                + " Matrix.from_fortranfile()"
            )
            f.close()
            return Matrix.from_fortranfile(filename)
        if itemp1 == 0 and itemp2 == icount:
            f.close()
            return Matrix.read_dense(filename, forgive=forgive)

        ncol, nrow = abs(itemp1), abs(itemp2)
        if itemp1 >= 0:
            # raise TypeError('Matrix.from_binary(): Jco produced by ' +
            #                 'deprecated version of PEST,' +
            #                 'Use JcoTRANS to convert to new format')
            # print("new binary format detected...")

            data = np.fromfile(f, Matrix.coo_rec_dt, icount)
            if data["i"].min() < 0:
                raise Exception("Matrix.from_binary(): 'i' index values less than 0")
            if data["j"].min() < 0:
                raise Exception("Matrix.from_binary(): 'j' index values less than 0")
            x = np.zeros((nrow, ncol))
            x[data["i"], data["j"]] = data["dtemp"]
            data = x
            # read obs and parameter names
            col_names = []
            row_names = []
            for j in range(ncol):
                name = (
                    struct.unpack(
                        str(Matrix.new_par_length) + "s", f.read(Matrix.new_par_length)
                    )[0]
                    .strip()
                    .lower()
                    .decode()
                )
                col_names.append(name)
            for i in range(nrow):
                name = (
                    struct.unpack(
                        str(Matrix.new_obs_length) + "s", f.read(Matrix.new_obs_length)
                    )[0]
                    .strip()
                    .lower()
                    .decode()
                )
                row_names.append(name)
            f.close()
        else:

            # read all data records
            # using this a memory hog, but really fast
            data = np.fromfile(f, Matrix.binary_rec_dt, icount)
            icols = ((data["j"] - 1) // nrow) + 1
            irows = data["j"] - ((icols - 1) * nrow)
            x = np.zeros((nrow, ncol))
            x[irows - 1, icols - 1] = data["dtemp"]
            data = x
            # read obs and parameter names
            col_names = []
            row_names = []
            for j in range(ncol):
                name = (
                    struct.unpack(
                        str(Matrix.par_length) + "s", f.read(Matrix.par_length)
                    )[0]
                    .strip()
                    .lower()
                    .decode()
                )
                col_names.append(name)
            for i in range(nrow):

                name = (
                    struct.unpack(
                        str(Matrix.obs_length) + "s", f.read(Matrix.obs_length)
                    )[0]
                    .strip()
                    .lower()
                    .decode()
                )
                row_names.append(name)
            f.close()
        if len(row_names) != data.shape[0]:
            raise Exception(
                "Matrix.read_binary() len(row_names) ("
                + str(len(row_names))
                + ") != x.shape[0] ("
                + str(data.shape[0])
                + ")"
            )
        if len(col_names) != data.shape[1]:
            raise Exception(
                "Matrix.read_binary() len(col_names) ("
                + str(len(col_names))
                + ") != self.shape[1] ("
                + str(data.shape[1])
                + ")"
            )
        return data, row_names, col_names

    @staticmethod
    def from_fortranfile(filename):
        """a binary load method to accommodate one of the many
            bizarre fortran binary writing formats

        Args:
            filename (`str`): name of the binary matrix file

        Returns:
            tuple containing

            - **numpy.ndarray**: the numeric values in the file
            - **['str']**: list of row names
            - **[`str`]**: list of col_names


        """
        try:
            from scipy.io import FortranFile
        except Exception as e:
            raise Exception("Matrix.from_fortranfile requires scipy")
        f = FortranFile(filename, mode="r")
        itemp1, itemp2 = f.read_ints()
        icount = int(f.read_ints())
        if itemp1 >= 0:
            raise TypeError(
                "Matrix.from_binary(): Jco produced by "
                + "deprecated version of PEST,"
                + "Use JcoTRANS to convert to new format"
            )
        ncol, nrow = abs(itemp1), abs(itemp2)
        data = []
        for _ in range(icount):
            d = f.read_record(Matrix.binary_rec_dt)[0]
            data.append(d)
        data = np.array(data, dtype=Matrix.binary_rec_dt)
        icols = ((data["j"] - 1) // nrow) + 1
        irows = data["j"] - ((icols - 1) * nrow)
        x = np.zeros((nrow, ncol))
        x[irows - 1, icols - 1] = data["dtemp"]
        row_names = []
        col_names = []
        for j in range(ncol):
            name = f.read_record("|S12")[0].strip().decode()
            col_names.append(name)
        # obs_rec = np.dtype((np.str_, self.obs_length))
        for i in range(nrow):
            name = f.read_record("|S20")[0].strip().decode()
            row_names.append(name)
        if len(row_names) != x.shape[0]:
            raise Exception(
                "Matrix.from_fortranfile() len(row_names) ("
                + str(len(row_names))
                + ") != self.shape[0] ("
                + str(x.shape[0])
                + ")"
            )
        if len(col_names) != x.shape[1]:
            raise Exception(
                "Matrix.from_fortranfile() len(col_names) ("
                + str(len(col_names))
                + ") != self.shape[1] ("
                + str(x.shape[1])
                + ")"
            )
        # return cls(x=x,row_names=row_names,col_names=col_names)
        return x, row_names, col_names

    def to_ascii(self, filename, icode=2):
        """write a PEST-compatible ASCII Matrix/vector file

        Args:
            filename (`str`): filename to write to
            icode (`int`, optional): PEST-style info code for matrix style.
                Default is 2.
        Note:
            if `icode` == -1, a 1-d  vector is written that represents a diagonal matrix.  An
            exception is raised if `icode` == -1 and `isdiagonal` is False



        """
        if icode == -1:
            if not self.isdiagonal:
                raise Exception(
                    "Matrix.to_ascii(): error: icode supplied as -1 for non-diagonal matrix"
                )
            if self.shape[0] != self.shape[1]:
                raise Exception(
                    "Matrix.to_ascii(): error: icode supplied as -1 for non-square matrix"
                )
        nrow, ncol = self.shape
        f_out = open(filename, "w")
        f_out.write(" {0:7.0f} {1:7.0f} {2:7.0f}\n".format(nrow, ncol, icode))
        f_out.close()
        f_out = open(filename, "ab")
        if self.isdiagonal and icode != -1:
            x = np.diag(self.__x[:, 0])
        else:
            x = self.__x
        np.savetxt(f_out, x, fmt="%15.7E", delimiter="")
        f_out.close()
        f_out = open(filename, "a")
        if icode == 1 or icode == -1:
            f_out.write("* row and column names\n")
            for r in self.row_names:
                f_out.write(r + "\n")
        else:
            f_out.write("* row names\n")
            for r in self.row_names:
                f_out.write(r + "\n")
            f_out.write("* column names\n")
            for c in self.col_names:
                f_out.write(c + "\n")
            f_out.close()

    @classmethod
    def from_ascii(cls, filename):
        """load a PEST-compatible ASCII matrix/vector file into a
        `Matrix` instance

        Args:
            filename (`str`): name of the file to read

        Returns:
            `Matrix`: `Matrix` loaded from ASCII file

        Example::

            mat = pyemu.Matrix.from_ascii("my.mat")
            cov = pyemi.Cov.from_ascii("my.cov")

        """
        x, row_names, col_names, isdiag = Matrix.read_ascii(filename)
        return cls(x=x, row_names=row_names, col_names=col_names, isdiagonal=isdiag)

    @staticmethod
    def read_ascii(filename):
        """read a PEST-compatible ASCII matrix/vector file

        Args:
            filename (`str`): file to read from

        Returns:
            tuple containing

            - **numpy.ndarray**: numeric values
            - **['str']**: list of row names
            - **[`str`]**: list of column names
            - **bool**: diagonal flag

        """

        f = open(filename, "r")
        raw = f.readline().strip().split()
        nrow, ncol = int(raw[0]), int(raw[1])
        icode = int(raw[2])
        # x = np.fromfile(f, dtype=self.double, count=nrow * ncol, sep=' ')
        # this painfully slow and ugly read is needed to catch the
        # fortran floating points that have 3-digit exponents,
        # which leave out the base (e.g. 'e') : "-1.23455+300"
        count = 0
        x = []
        while True:
            line = f.readline()
            if line == "":
                raise Exception("Matrix.from_ascii() error: EOF")
            raw = line.strip().split()
            for r in raw:
                try:
                    x.append(float(r))
                except Exception as e:
                    # overflow
                    if "+" in r:
                        x.append(1.0e30)
                    # underflow
                    elif "-" in r:
                        x.append(0.0)
                    else:
                        raise Exception(
                            "Matrix.from_ascii() error: "
                            + " can't cast "
                            + r
                            + " to float"
                        )
                count += 1
                if icode != -1 and count == (nrow * ncol):
                    break
                elif icode == -1 and count == nrow:
                    break
            if icode != -1 and count == (nrow * ncol):
                break
            elif icode == -1 and count == nrow:
                break

        x = np.array(x, dtype=Matrix.double)
        if icode != -1:
            x.resize(nrow, ncol)
        else:
            x = np.diag(x)
        line = f.readline().strip().lower()
        if not line.startswith("*"):
            raise Exception(
                'Matrix.from_ascii(): error loading ascii file," +\
                "line should start with * not '
                + line
            )
        if "row" in line and "column" in line:
            if nrow != ncol:
                raise Exception("nrow != ncol")
            names = []
            for i in range(nrow):
                line = f.readline().strip().lower()
                names.append(line)
            row_names = copy.deepcopy(names)
            col_names = names

        else:
            names = []
            for _ in range(nrow):
                line = f.readline().strip().lower()
                names.append(line)
            row_names = names
            line = f.readline().strip().lower()
            if "column" not in line:
                raise Exception(
                    "Matrix.from_ascii(): line should be * column names "
                    + "instead of: "
                    + line
                )
            names = []
            for _ in range(ncol):
                line = f.readline().strip().lower()
                names.append(line)
            col_names = names
        f.close()
        # test for diagonal
        isdiagonal = False
        if nrow == ncol:
            diag = np.diag(np.diag(x))
            diag_tol = 1.0e-6
            diag_delta = np.abs(diag.sum() - x.sum())
            if diag_delta < diag_tol:
                isdiagonal = True
                x = np.atleast_2d(np.diag(x)).transpose()
        return x, row_names, col_names, isdiagonal

    def df(self):
        """wrapper of Matrix.to_dataframe()"""
        return self.to_dataframe()

    @classmethod
    def from_dataframe(cls, df):
        """class method to create a new `Matrix` instance from a
         `pandas.DataFrame`

        Args:
            df (`pandas.DataFrame`): dataframe

        Returns:
            `Matrix`: `Matrix` instance derived from `df`.

        Example::

            df = pd.read_csv("my.csv")
            mat = pyemu.Matrix.from_dataframe(df)

        """
        if not isinstance(df, pd.DataFrame):
            raise Exception("df is not a DataFrame")
        row_names = copy.deepcopy(list(df.index))
        col_names = copy.deepcopy(list(df.columns))
        return cls(x=df.values, row_names=row_names, col_names=col_names)

    @classmethod
    def from_names(
        cls, row_names, col_names, isdiagonal=False, autoalign=True, random=False
    ):
        """class method to create a new Matrix instance from
        row names and column names, filled with trash

        Args:
            row_names (['str']): row names for the new `Matrix`
            col_names (['str']): col_names for the new matrix
            isdiagonal (`bool`, optional): flag for diagonal matrix. Default is False
            autoalign (`bool`, optional): flag for autoaligning new matrix
                during linear algebra calcs. Default is True
            random (`bool`): flag for contents of the trash matrix.
                If True, fill with random numbers, if False, fill with zeros
                Default is False

        Returns:
            `Matrix`: the new Matrix instance

        Example::

            row_names = ["row_1","row_2"]
            col_names = ["col_1,"col_2"]
            m = pyemu.Matrix.from_names(row_names,col_names)


        """
        if random:
            return cls(
                x=np.random.random((len(row_names), len(col_names))),
                row_names=row_names,
                col_names=col_names,
                isdiagonal=isdiagonal,
                autoalign=autoalign,
            )
        else:
            return cls(
                x=np.empty((len(row_names), len(col_names))),
                row_names=row_names,
                col_names=col_names,
                isdiagonal=isdiagonal,
                autoalign=autoalign,
            )

    def to_dataframe(self):
        """return a pandas.DataFrame representation of `Matrix`

        Returns:
            `pandas.DataFrame`: a dataframe derived from `Matrix`

        Note:
            if `self.isdiagonal` is True, the full matrix is used to fill
            the dataframe - lots of zeros.

        """
        if self.isdiagonal:
            x = np.diag(self.__x[:, 0])
        else:
            x = self.__x
        return pd.DataFrame(data=x, index=self.row_names, columns=self.col_names)

    def extend(self, other):
        """extend `Matrix` with the elements of other, returning a new matrix.

        Args:
        other (`Matrix`):  the Matrix to extend self by

        Returns:
            `Matrix`: new, extended `Matrix`

        Note:
            No row or column names can be shared between self and other

        Example::

            jco1 = pyemu.Jco.from_binary("pest_history.jco")
            jco2 = pyemu.Jco.from_binary("pest_forecast.jco")

            jco_ext = jco1.extend(jco2)


        """

        if len(set(self.row_names).intersection(set(other.row_names))) != 0:
            raise Exception("shared row names")
        if len(set(self.col_names).intersection(set(other.col_names))) != 0:
            raise Exception("shared col_names")
        if type(self) != type(other):
            raise Exception("type mismatch")
        new_row_names = copy.copy(self.row_names)
        new_row_names.extend(other.row_names)
        new_col_names = copy.copy(self.col_names)
        new_col_names.extend(other.col_names)

        new_x = np.zeros((len(new_row_names), len(new_col_names)))
        new_x[0 : self.shape[0], 0 : self.shape[1]] = self.as_2d
        new_x[
            self.shape[0] : self.shape[0] + other.shape[0],
            self.shape[1] : self.shape[1] + other.shape[1],
        ] = other.as_2d
        isdiagonal = True
        if not self.isdiagonal or not other.isdiagonal:
            isdiagonal = False

        return type(self)(
            x=new_x,
            row_names=new_row_names,
            col_names=new_col_names,
            isdiagonal=isdiagonal,
        )


class Jco(Matrix):
    """a thin wrapper class to get more intuitive attribute names.  Functions
    exactly like `Matrix`
    """

    def __init(self, **kwargs):
        """Jco constructor takes the same arguments as Matrix.

        Args:
            **kwargs (`dict`): constructor arguments for `Matrix`

        Example:

            jco = pyemu.Jco.from_binary("my.jco")


        """

        super(Jco, self).__init__(kwargs)

    @property
    def par_names(self):
        """thin wrapper around `Matrix.col_names`

        Returns:
            [`str`]: a list of parameter names

        """
        return self.col_names

    @property
    def obs_names(self):
        """thin wrapper around `Matrix.row_names`

        Returns:
            ['str']: a list of observation names

        """
        return self.row_names

    @property
    def npar(self):
        """number of parameters in the Jco

        Returns:
            `int`: number of parameters (columns)

        """
        return self.shape[1]

    @property
    def nobs(self):
        """number of observations in the Jco

        Returns:
            `int`: number of observations (rows)

        """
        return self.shape[0]

    @classmethod
    def from_pst(cls, pst, random=False):
        """construct a new empty Jco from a control file optionally filled
        with trash

        Args:
            pst (`pyemu.Pst`): a pest control file instance.  If type is 'str',
                `pst` is loaded from filename
            random (`bool`): flag for contents of the trash matrix.
                If True, fill with random numbers, if False, fill with zeros
                Default is False

        Returns:
            `Jco`: the new Jco instance

        """

        if isinstance(pst, str):
            pst = Pst(pst)

        return Jco.from_names(pst.obs_names, pst.adj_par_names, random=random)


class Cov(Matrix):
    """Diagonal and/or dense Covariance matrices

    Args:
        x (`numpy.ndarray`): numeric values
        names ([`str`]): list of row and column names
        isdigonal (`bool`): flag if the Matrix is diagonal
        autoalign (`bool`): flag to control the autoalignment of Matrix during
            linear algebra operations

    Example::

        data = np.random.random((10,10))
        names = ["par_{0}".format(i) for i in range(10)]
        mat = pyemu.Cov(x=data,names=names)
        mat.to_binary("mat.jco")

    Note:
        `row_names` and `col_names` args are supported in the constructor
        so support inheritance.  However, users should only pass `names`

    """

    def __init__(
        self,
        x=None,
        names=[],
        row_names=[],
        col_names=[],
        isdiagonal=False,
        autoalign=True,
    ):

        self.__identity = None
        self.__zero = None
        # if len(row_names) > 0 and len(col_names) > 0:
        #    assert row_names == col_names
        self.__identity = None
        self.__zero = None
        # if len(row_names) > 0 and len(col_names) > 0:
        #    assert row_names == col_names
        if len(names) != 0 and len(row_names) == 0:
            row_names = names
        if len(names) != 0 and len(col_names) == 0:
            col_names = names
        super(Cov, self).__init__(
            x=x,
            isdiagonal=isdiagonal,
            row_names=row_names,
            col_names=col_names,
            autoalign=autoalign,
        )
        super(Cov, self).__init__(
            x=x,
            isdiagonal=isdiagonal,
            row_names=row_names,
            col_names=col_names,
            autoalign=autoalign,
        )

    @property
    def identity(self):
        """get an identity `Cov` of the same shape

        Returns:
            `Cov`: new `Cov` instance with identity matrix

        Note:
            the returned identity matrix has the same row-col names as self

        """
        if self.__identity is None:
            self.__identity = Cov(
                x=np.atleast_2d(np.ones(self.shape[0])).transpose(),
                names=self.row_names,
                isdiagonal=True,
            )
        return self.__identity

    @property
    def zero(self):
        """get a diagonal instance of `Cov` with all zeros on the diagonal

        Returns:
            `Cov`: new `Cov` instance with zeros

        """
        if self.__zero is None:
            self.__zero = Cov(
                x=np.atleast_2d(np.zeros(self.shape[0])).transpose(),
                names=self.row_names,
                isdiagonal=True,
            )
        return self.__zero

    def condition_on(self, conditioning_elements):
        """get a new Covariance object that is conditional on knowing some
        elements.  uses Schur's complement for conditional Covariance
        propagation

        Args:
            conditioning_elements (['str']): list of names of elements to condition on

        Returns:
            `Cov`: new conditional `Cov` that assumes `conditioning_elements` have become known

        Example::

            prior_cov = pyemu.Cov.from_parameter_data(pst)
            now_known_pars = pst.adj_par_names[:5]
            post_cov = prior_cov.condition_on(now_known_pars)


        """
        if not isinstance(conditioning_elements, list):
            conditioning_elements = [conditioning_elements]
        for iname, name in enumerate(conditioning_elements):
            conditioning_elements[iname] = name.lower()
            if name.lower() not in self.col_names:
                raise Exception("Cov.condition_on() name not found: " + name)
        keep_names = []
        for name in self.col_names:
            if name not in conditioning_elements:
                keep_names.append(name)
        # C11
        new_Cov = self.get(keep_names)
        if self.isdiagonal:
            return new_Cov
        # C22^1
        cond_Cov = self.get(conditioning_elements).inv
        # C12
        upper_off_diag = self.get(keep_names, conditioning_elements)
        # print(new_Cov.shape,upper_off_diag.shape,cond_Cov.shape)
        return new_Cov - (upper_off_diag * cond_Cov * upper_off_diag.T)

    @property
    def names(self):
        """wrapper for getting row_names.  row_names == col_names for Cov

        Returns:
            [`str`]: list of names

        """
        return self.row_names

    def replace(self, other):
        """replace elements in the covariance matrix with elements from other.
        if other is not diagonal, then this `Cov` becomes non diagonal

        Args:
            `Cov`: the Cov to replace elements in this `Cov` with

        Note:
            operates in place.  Other must have the same row-col names as self

        """
        if not isinstance(other, Cov):
            raise Exception(
                "Cov.replace() other must be Cov, not {0}".format(type(other))
            )
        # make sure the names of other are in self
        missing = [n for n in other.names if n not in self.names]
        if len(missing) > 0:
            raise Exception(
                "Cov.replace(): the following other names are not"
                + " in self names: {0}".format(",".join(missing))
            )
        self_idxs = self.indices(other.names, 0)
        other_idxs = other.indices(other.names, 0)

        if self.isdiagonal and other.isdiagonal:
            self._Matrix__x[self_idxs] = other.x[other_idxs]
            return
        if self.isdiagonal:
            self._Matrix__x = self.as_2d
            self.isdiagonal = False

        # print("allocating other_x")
        other_x = other.as_2d
        # print("replacing")
        for i, ii in zip(self_idxs, other_idxs):
            self._Matrix__x[i, self_idxs] = other_x[ii, other_idxs].copy()
        # print("resetting")
        # self.reset_x(self_x)
        # self.isdiagonal = False

    def to_uncfile(
        self, unc_file, covmat_file="cov.mat", var_mult=1.0, include_path=False
    ):
        """write a PEST-compatible uncertainty file

        Args:
            unc_file (`str`): filename of the uncertainty file
            covmat_file (`str`): covariance matrix filename. Default is
                "Cov.mat".  If None, and Cov.isdiaonal, then a standard deviation
                form of the uncertainty file is written.  Exception raised if `covmat_file` is `None`
                and not `Cov.isdiagonal`
            var_mult (`float`): variance multiplier for the covmat_file entry
            include_path (`bool`): flag to include the path of `unc_file` in the name of `covmat_file`.
                Default is False - not sure why you would ever make this True...

        Example::

            cov = pyemu.Cov.from_parameter_data(pst)
            cov.to_uncfile("my.unc")

        """
        assert (
            len(self.row_names) == self.shape[0]
        ), "Cov.to_uncfile(): len(row_names) != x.shape[0] "
        if len(self.row_names) != self.shape[0]:
            raise Exception("Cov.to_uncfile(): len(row_names) != x.shape[0]")
        if covmat_file:
            f = open(unc_file, "w")
            f.write("START COVARIANCE_MATRIX\n")
            if include_path:
                f.write(" file " + covmat_file + "\n")
            else:
                f.write(" file " + os.path.split(covmat_file)[-1] + "\n")
            f.write(" variance_multiplier {0:15.6E}\n".format(var_mult))
            f.write("END COVARIANCE_MATRIX\n")
            f.close()
            if include_path:
                self.to_ascii(covmat_file, icode=1)
            else:
                self.to_ascii(
                    os.path.join(
                        os.path.dirname(unc_file), os.path.split(covmat_file)[-1]
                    ),
                    icode=1,
                )

        else:
            if self.isdiagonal:
                f = open(unc_file, "w")
                f.write("START STANDARD_DEVIATION\n")
                for iname, name in enumerate(self.row_names):
                    f.write(
                        "  {0:20s}  {1:15.6E}\n".format(name, np.sqrt(self.x[iname, 0]))
                    )
                f.write("END STANDARD_DEVIATION\n")
                f.close()
            else:
                raise Exception(
                    "Cov.to_uncfile(): can't write non-diagonal "
                    + "object as standard deviation block"
                )

    @classmethod
    def from_obsweights(cls, pst_file):
        """instantiates a `Cov` instance from observation weights in
        a PEST control file.

        Args:
            pst_file (`str`): pest control file name

        Returns:
            `Cov`: a diagonal observation noise covariance matrix derived from the
            weights in the pest control file.  Zero-weighted observations
            are included with a weight of 1.0e-30

        Note:
            Calls `Cov.from_observation_data()`

        Example::

            obscov = pyemu.Cov.from_obsweights("my.pst")


        """
        if not pst_file.endswith(".pst"):
            pst_file += ".pst"
        return Cov.from_observation_data(Pst(pst_file))

    @classmethod
    def from_observation_data(cls, pst):
        """instantiates a `Cov` from pyemu.Pst.observation_data

        Args:
            pst (`pyemu.Pst`): control file instance

        Returns:
            `Cov`: a diagonal observation noise covariance matrix derived from the
            weights in the pest control file.  Zero-weighted observations
            are included with a weight of 1.0e-30

        Example::

            obscov = pyemu.Cov.from_observation_data(pst)

        """
        nobs = pst.observation_data.shape[0]
        x = np.zeros((nobs, 1))
        onames = []
        ocount = 0
        std_dict = {}
        if "standard_deviation" in pst.observation_data.columns:
            std_dict = pst.observation_data.standard_deviation.to_dict()
            std_dict = {o:s**2 for o,s in std_dict.items() if pd.notna(s)}
        for weight, obsnme in zip(
            pst.observation_data.weight, pst.observation_data.obsnme
        ):
            w = float(weight)
            w = max(w, 1.0e-30)

            x[ocount] = std_dict.get(obsnme,(1.0 / w) ** 2)
            ocount += 1
            onames.append(obsnme.lower())
        return cls(x=x, names=onames, isdiagonal=True)

    @classmethod
    def from_parbounds(cls, pst_file, sigma_range=4.0, scale_offset=True):
        """Instantiates a `Cov` from a pest control file parameter data section using
        parameter bounds as a proxy for uncertainty.


        Args:
            pst_file (`str`): pest control file name
            sigma_range (`float`): defines range of upper bound - lower bound in terms of standard
                deviation (sigma). For example, if sigma_range = 4, the bounds
                represent 4 * sigma.  Default is 4.0, representing approximately
                95% confidence of implied normal distribution
            scale_offset (`bool`): flag to apply scale and offset to parameter upper and lower
                bounds before calculating variance. In some cases, not applying scale and
                offset can result in undefined (log) variance.  Default is True.

        Returns:
            `Cov`: diagonal parameter `Cov` matrix created from parameter bounds

        Note:
            Calls `Cov.from_parameter_data()`

        """
        if not pst_file.endswith(".pst"):
            pst_file += ".pst"
        new_pst = Pst(pst_file)
        return Cov.from_parameter_data(new_pst, sigma_range, scale_offset)

    @classmethod
    def from_parameter_data(cls, pst, sigma_range=4.0, scale_offset=True,
                            subset=None):
        """Instantiates a `Cov` from a pest control file parameter data section using
        parameter bounds as a proxy for uncertainty.


        Args:
            pst_file (`str`): pest control file name
            sigma_range (`float`): defines range of upper bound - lower bound in terms of standard
                deviation (sigma). For example, if sigma_range = 4, the bounds
                represent 4 * sigma.  Default is 4.0, representing approximately
                95% confidence of implied normal distribution
            scale_offset (`bool`): flag to apply scale and offset to parameter upper and lower
                bounds before calculating variance. In some cases, not applying scale and
                offset can result in undefined (log) variance.  Default is True.
            subset (`list`-like, optional): Subset of parameters to draw

        Returns:
            `Cov`: diagonal parameter `Cov` matrix created from parameter bounds

        Note:
            Calls `Cov.from_parameter_data()`

        """
        if subset is not None:
            missing = subset.difference(pst.par_names)
            if not missing.empty:
                warnings.warn(
                    f"{len(missing)} parameter names not present in Pst:\n"
                    f"{missing}", PyemuWarning
                )
                subset = subset.intersection(pst.par_names)
            par_dat = pst.parameter_data.loc[subset, :]
        else:
            par_dat = pst.parameter_data
        npar = (~par_dat.partrans.isin(["fixed", "tied"])).sum()
        x = np.zeros((npar, 1))
        names = []
        idx = 0
        for i, row in par_dat.iterrows():
            t = row["partrans"]
            if t in ["fixed", "tied"]:
                continue
            if "standard_deviation" in row.index and pd.notna(row["standard_deviation"]):
                if t == "log":
                    var = (np.log10(row["standard_deviation"])) ** 2
                else:
                    var = row["standard_deviation"] ** 2
            else:
                if scale_offset:
                    lb = row.parlbnd * row.scale + row.offset
                    ub = row.parubnd * row.scale + row.offset
                else:
                    lb = row.parlbnd
                    ub = row.parubnd

                if t == "log":
                    var = ((np.log10(np.abs(ub)) - np.log10(np.abs(lb))) / sigma_range) ** 2
                else:
                    var = ((ub - lb) / sigma_range) ** 2
            if np.isnan(var) or not np.isfinite(var):
                raise Exception(
                    "Cov.from_parameter_data() error: "
                    + "variance for parameter {0} is nan".format(row["parnme"])
                )
            if var == 0.0:
                s = (
                    "Cov.from_parameter_data() error: "
                    + "variance for parameter {0} is 0.0.".format(row["parnme"])
                )
                s += "  This might be from enforcement of scale/offset and log transform."
                s += "  Try changing 'scale_offset' arg"
                raise Exception(s)
            x[idx] = var
            names.append(row["parnme"].lower())
            idx += 1

        return cls(x=x, names=names, isdiagonal=True)

    @classmethod
    def from_uncfile(cls, filename, pst=None):
        """instaniates a `Cov` from a PEST-compatible uncertainty file

        Args:
            filename (`str`):  uncertainty file name
            pst ('pyemu.Pst`): a control file instance.  this is needed if
                "first_parameter" and "last_parameter" keywords.  Default is None

        Returns:
            `Cov`: `Cov` instance from uncertainty file

        Example::

            cov = pyemu.Cov.from_uncfile("my.unc")

        """

        if pst is not None:
            if isinstance(pst, str):
                pst = Pst(pst)

        nentries = Cov._get_uncfile_dimensions(filename)
        x = np.zeros((nentries, nentries))
        row_names = []
        col_names = []
        f = open(filename, "r")
        isdiagonal = True
        idx = 0
        while True:
            line = f.readline().lower()
            if len(line) == 0:
                break
            line = line.strip()
            if "start" in line:
                if "pest_control_file" in line:
                    raise Exception(
                        "Cov.from_uncfile() error: 'pest_control_file' block not supported"
                    )

                if "standard_deviation" in line:
                    std_mult = 1.0
                    while True:
                        line2 = f.readline().strip().lower()
                        if line2.strip().lower().startswith("end"):
                            break

                        raw = line2.strip().split()
                        name, val = raw[0], float(raw[1])
                        if name == "std_multiplier":
                            std_mult = val
                        else:
                            x[idx, idx] = (val * std_mult) ** 2
                            if name in row_names:
                                raise Exception(
                                    "Cov.from_uncfile():"
                                    + "duplicate name: "
                                    + str(name)
                                )
                            row_names.append(name)
                            col_names.append(name)
                            idx += 1

                elif "covariance_matrix" in line:
                    isdiagonal = False
                    var = 1.0
                    first_par = None
                    last_par = None
                    while True:
                        line2 = f.readline().strip().lower()
                        if line2.strip().lower().startswith("end"):
                            break
                        if line2.startswith("file"):
                            mat_filename = os.path.join(
                                os.path.dirname(filename),
                                line2.split()[1].replace("'", "").replace('"', ""),
                            )
                            cov = Matrix.from_ascii(mat_filename)

                        elif line2.startswith("variance_multiplier"):
                            var = float(line2.split()[1])
                        elif line2.startswith("first_parameter"):
                            if pst is None:
                                raise Exception(
                                    "Cov.from_uncfile(): 'first_parameter' usage requires the 'pst' arg to be passed"
                                )
                            first_par = line2.split()[1]
                        elif line2.startswith("last_parameter"):
                            if pst is None:
                                raise Exception(
                                    "Cov.from_uncfile(): 'last_parameter' usage requires the 'pst' arg to be passed"
                                )
                            last_par = line2.split()[1]

                        else:
                            raise Exception(
                                "Cov.from_uncfile(): "
                                + "unrecognized keyword in"
                                + "std block: "
                                + line2
                            )
                    if var != 1.0:
                        cov *= var
                    if first_par is not None:
                        if last_par is None:
                            raise Exception(
                                "'first_par' found but 'last_par' not found"
                            )
                        if first_par not in pst.par_names:
                            raise Exception(
                                "'first_par' {0} not found in pst.par_names".format(
                                    first_par
                                )
                            )
                        if last_par not in pst.par_names:
                            raise Exception(
                                "'last_par' {0} not found in pst.par_names".format(
                                    last_par
                                )
                            )
                        names = pst.parameter_data.loc[
                            first_par:last_par, "parnme"
                        ].tolist()
                        if len(names) != cov.shape[0]:
                            print(names)
                            print(len(names), cov.shape)
                            raise Exception(
                                "the number of par names between 'first_par' and "
                                "'last_par' != elements in the cov matrix {0}".format(
                                    mat_filename
                                )
                            )
                        cov.row_names = names
                        cov.col_names = names

                    for name in cov.row_names:
                        if name in row_names:
                            raise Exception(
                                "Cov.from_uncfile():" + " duplicate name: " + str(name)
                            )
                    row_names.extend(cov.row_names)
                    col_names.extend(cov.col_names)

                    for i in range(cov.shape[0]):
                        x[idx + i, idx : idx + cov.shape[0]] = cov.x[i, :].copy()
                    idx += cov.shape[0]
                else:
                    raise Exception(
                        "Cov.from_uncfile(): " + "unrecognized block:" + str(line)
                    )
        f.close()
        if isdiagonal:
            x = np.atleast_2d(np.diag(x)).transpose()
        return cls(x=x, names=row_names, isdiagonal=isdiagonal)

    @staticmethod
    def _get_uncfile_dimensions(filename):
        """quickly read an uncertainty file to find the dimensions"""
        f = open(filename, "r")
        nentries = 0
        while True:
            line = f.readline().lower()
            if len(line) == 0:
                break
            line = line.strip()
            if "start" in line:
                if "standard_deviation" in line:
                    while True:
                        line2 = f.readline().strip().lower()
                        if "std_multiplier" in line2:
                            continue
                        if line2.strip().lower().startswith("end"):
                            break
                        nentries += 1

                elif "covariance_matrix" in line:
                    while True:
                        line2 = f.readline().strip().lower()
                        if line2 == "":
                            raise Exception("EOF while looking for 'end' block")
                        if line2.strip().lower().startswith("end"):
                            break
                        if line2.startswith("file"):
                            mat_filename = os.path.join(
                                os.path.dirname(filename),
                                line2.split()[1].replace("'", "").replace('"', ""),
                            )
                            cov = Matrix.from_ascii(mat_filename)
                            nentries += len(cov.row_names)
                        elif line2.startswith("variance_multiplier"):
                            pass
                        elif line2.startswith("first_parameter"):
                            pass
                        elif line2.startswith("last_parameter"):
                            pass

                        else:
                            raise Exception(
                                "Cov.get_uncfile_dimensions(): "
                                + "unrecognized keyword in Covariance block: "
                                + line2
                            )
                else:
                    raise Exception(
                        "Cov.get_uncfile_dimensions():"
                        + "unrecognized block:"
                        + str(line)
                    )
        f.close()
        return nentries

    @classmethod
    def identity_like(cls, other):
        """Get an identity matrix Cov instance like other `Cov`

        Args:
            other (`Matrix`):  other matrix - must be square

        Returns:
            `Cov`: new identity matrix `Cov` with shape of `other`

        Note:
            the returned identity cov matrix is treated as non-diagonal

        """
        if other.shape[0] != other.shape[1]:
            raise Exception("not square")
        x = np.identity(other.shape[0])
        return cls(x=x, names=other.row_names, isdiagonal=False)

    def to_pearson(self):
        """Convert Cov instance to Pearson correlation coefficient
        matrix

        Returns:
            `Matrix`: A `Matrix` of correlation coefs.  Return type is `Matrix`
            on purpose so that it is clear the returned instance is not a Cov

        Example::

            # plot the posterior parameter correlation matrix
            import matplotlib.pyplot as plt
            cov = pyemu.Cov.from_ascii("pest.post.cov")
            cc = cov.to_pearson()
            cc.x[cc.x==1.0] = np.nan
            plt.imshow(cc)

        """
        std_dict = (
            self.get_diagonal_vector().to_dataframe()["diag"].apply(np.sqrt).to_dict()
        )
        pearson = self.identity.as_2d
        if self.isdiagonal:
            return Matrix(x=pearson, row_names=self.row_names, col_names=self.col_names)
        df = self.to_dataframe()
        # fill the lower triangle
        for i, iname in enumerate(self.row_names):
            for j, jname in enumerate(self.row_names[i + 1 :]):
                # cv = df.loc[iname,jname]
                # std1,std2 = std_dict[iname],std_dict[jname]
                # cc = cv / (std1*std2)
                # v1 = np.sqrt(df.loc[iname,iname])
                # v2 = np.sqrt(df.loc[jname,jname])
                pearson[i, j + i + 1] = df.loc[iname, jname] / (
                    std_dict[iname] * std_dict[jname]
                )

        # replicate across diagonal
        for i, iname in enumerate(self.row_names[:-1]):
            pearson[i + 1 :, i] = pearson[i, i + 1 :]
        return Matrix(x=pearson, row_names=self.row_names, col_names=self.col_names)
