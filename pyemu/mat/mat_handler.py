"""Matrix, Jco and Cov classes for easy linear algebra
"""
from __future__ import print_function, division
import copy
import struct
import warnings
from datetime import datetime
import numpy as np
import pandas
import scipy.linalg as la
from scipy.io import FortranFile
import scipy.sparse

from pyemu.pst.pst_handler import Pst


def save_coo(x, row_names, col_names,  filename, chunk=None):
    """write a PEST-compatible binary file.  The data format is
    [int,int,float] for i,j,value.  It is autodetected during
    the read with Matrix.from_binary().

    Parameters
    ----------
    x : numpy.sparse
        coo sparse matrix
    row_names : list
        list of row_names
    col_names : list
        list of col_names
    filename : str
        filename to save binary file
    droptol : float
        absolute value tolerance to make values smaller than zero.  Default is None
    chunk : int
        number of elements to write in a single pass.  Default is None

    """

    f = open(filename, 'wb')
    # print("counting nnz")
    # write the header
    header = np.array((x.shape[1], x.shape[0], x.nnz),
                      dtype=Matrix.binary_header_dt)
    header.tofile(f)

    data = np.core.records.fromarrays([x.row, x.col, x.data], dtype=Matrix.coo_rec_dt)
    data.tofile(f)

    for name in col_names:
        if len(name) > Matrix.par_length:
            name = name[:Matrix.par_length - 1]
        elif len(name) < Matrix.par_length:
            for i in range(len(name), Matrix.par_length):
                name = name + ' '
        f.write(name.encode())
    for name in row_names:
        if len(name) > Matrix.obs_length:
            name = name[:Matrix.obs_length - 1]
        elif len(name) < Matrix.obs_length:
            for i in range(len(name), Matrix.obs_length):
                name = name + ' '
        f.write(name.encode())
    f.close()


def concat(mats):
    """Concatenate Matrix objects.  Tries either axis.

    Parameters
    ----------
    mats: list
        list of Matrix objects

    Returns
    -------
    Matrix : Matrix
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
        raise Exception("mat_handler.concat(): all Matrix objects"+\
                        "must share either rows or cols")

    if row_match and col_match:
        raise Exception("mat_handler.concat(): all Matrix objects"+\
                        "share both rows and cols")

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

    Parameters
    ----------
    list1 : list
        a list of objects
    list2 : list
        a list of objects

    Returns
    -------
    list : list
        list of common objects shared by list1 and list2
        
    """
    #result = []
    #for item in list1:
    #    if item in list2:
    #        result.append(item)
    #Return list(set(list1).intersection(set(list2)))
    set2 = set(list2)
    result = [item for item in list1 if item in set2]
    return result


class Matrix(object):
    """a class for easy linear algebra

    Parameters
    ----------
    x : numpy.ndarray
        Matrix entries
    row_names : list
        list of row names
    col_names : list
        list of column names
    isdigonal : bool
        to determine if the Matrix is diagonal
    autoalign: bool
        used to control the autoalignment of Matrix objects
        during linear algebra operations

    Returns
    -------
        Matrix : Matrix

    Attributes
    ----------
        binary_header_dt : numpy.dtype
            the header info in the PEST binary file type
        binary_rec_dt : numpy.dtype
            the record info in the PEST binary file type

    Methods
    -------
    to_ascii : write a PEST-style ASCII matrix format file
    to_binary : write a PEST-stle compressed binary format file
    
    Note
    ----
    this class makes heavy use of property decorators to encapsulate
    private attributes

    """
    integer = np.int32
    double = np.float64
    char = np.uint8

    binary_header_dt = np.dtype([('itemp1', integer),
                                ('itemp2', integer),
                                ('icount', integer)])
    binary_rec_dt = np.dtype([('j', integer),
                            ('dtemp', double)])
    coo_rec_dt = np.dtype([('i', integer),('j', integer),
                          ('dtemp', double)])

    par_length = 12
    obs_length = 20

    def __init__(self, x=None, row_names=[], col_names=[], isdiagonal=False,
                 autoalign=True):


        self.col_names, self.row_names = [], []
        [self.col_names.append(str(c).lower()) for c in col_names]
        [self.row_names.append(str(r).lower()) for r in row_names]
        self.__x = None
        self.__u = None
        self.__s = None
        self.__v = None
        if x is not None:
            assert x.ndim == 2
            #x = np.atleast_2d(x)
            if isdiagonal and len(row_names) > 0:
                #assert 1 in x.shape,"Matrix error: diagonal matrix must have " +\
                #                    "one dimension == 1,shape is {0}".format(x.shape)
                mx_dim = max(x.shape)
                assert len(row_names) == mx_dim,\
                    'Matrix.__init__(): diagonal shape[1] != len(row_names) ' +\
                    str(x.shape) + ' ' + str(len(row_names))
                #x = x.transpose()
            else:
                if len(row_names) > 0:
                    assert len(row_names) == x.shape[0],\
                        'Matrix.__init__(): shape[0] != len(row_names) ' +\
                        str(x.shape) + ' ' + str(len(row_names))
                if len(col_names) > 0:
                    # if this a row vector
                    if len(row_names) == 0 and x.shape[1] == 1:
                        x.transpose()
                    assert len(col_names) == x.shape[1],\
                        'Matrix.__init__(): shape[1] != len(col_names) ' + \
                        str(x.shape) + ' ' + str(len(col_names))
            self.__x = x

        self.isdiagonal = bool(isdiagonal)
        self.autoalign = bool(autoalign)

    def reset_x(self,x,copy=True):
        """reset self.__x private attribute

        Parameters
        ----------
        x : numpy.ndarray
        copy : bool
            flag to make a copy of 'x'. Defaule is True
        
        Note
        ----
        makes a copy of 'x' argument
        
        """
        assert x.shape == self.shape
        if copy:
            self.__x = x.copy()
        else:
            self.__x = x

    def __str__(self):
        """overload of object.__str__()
        
        Returns
        -------
            str : str 

        """
        s = "shape:{0}:{1}".format(*self.shape)+" row names: " + str(self.row_names) + \
            '\n' + "col names: " + str(self.col_names) + '\n' + str(self.__x)
        return s

    def __getitem__(self, item):
        """a very crude overload of object.__getitem__().

        Parameters
        ----------
        item : iterable
         something that can be used as an index

        Returns
        -------
        Matrix : Matrix
            an object that is a sub-Matrix of self

        """
        if self.isdiagonal and isinstance(item, tuple):
            submat = np.atleast_2d((self.__x[item[0]]))
        else:
            submat = np.atleast_2d(self.__x[item])
        # transpose a row vector to a column vector
        if submat.shape[0] == 1:
            submat = submat.transpose()
        row_names = self.row_names[:submat.shape[0]]
        if self.isdiagonal:
            col_names = row_names
        else:
            col_names = self.col_names[:submat.shape[1]]
        return type(self)(x=submat, isdiagonal=self.isdiagonal,
                          row_names=row_names, col_names=col_names,
                          autoalign=self.autoalign)


    def __pow__(self, power):
        """overload of numpy.ndarray.__pow__() operator

        Parameters
        ----------
        power: (int or float)
            interpreted as follows: -1 = inverse of self,
            -0.5 = sqrt of inverse of self,
            0.5 = sqrt of self. All other positive
            ints = elementwise self raised to power

        Returns
        -------
        Matrix : Matrix
            a new Matrix object

        """
        if power < 0:
            if power == -1:
                return self.inv
            elif power == -0.5:
                return (self.inv).sqrt
            else:
                raise NotImplementedError("Matrix.__pow__() not implemented " +
                                          "for negative powers except for -1")

        elif int(power) != float(power):
            if power == 0.5:
                return self.sqrt
            else:
                raise NotImplementedError("Matrix.__pow__() not implemented " +
                                          "for fractional powers except 0.5")
        else:
            return type(self)(self.__x**power, row_names=self.row_names,
                              col_names=self.col_names,
                              isdiagonal=self.isdiagonal)


    def __sub__(self, other):
        """numpy.ndarray.__sub__() overload.  Tries to speedup by
         checking for scalars of diagonal matrices on either side of operator

        Parameters
        ----------
        other : scalar,numpy.ndarray,Matrix object
            the thing to difference

        Returns
        -------
        Matrix : Matrix

        """

        if np.isscalar(other):
            return Matrix(x=self.x - other, row_names=self.row_names,
                          col_names=self.col_names,
                          isdiagonal=self.isdiagonal)
        else:
            if isinstance(other, np.ndarray):
                assert self.shape == other.shape, "Matrix.__sub__() shape" +\
                                                  "mismatch: " +\
                                                  str(self.shape) + ' ' + \
                                                  str(other.shape)
                if self.isdiagonal:
                    elem_sub = -1.0 * other
                    for j in range(self.shape[0]):
                        elem_sub[j, j] += self.x[j]
                    return type(self)(x=elem_sub, row_names=self.row_names,
                                      col_names=self.col_names)
                else:
                    return type(self)(x=self.x - other,
                                      row_names=self.row_names,
                                      col_names=self.col_names)
            elif isinstance(other, Matrix):
                if self.autoalign and other.autoalign \
                        and not self.element_isaligned(other):
                    common_rows = get_common_elements(self.row_names,
                                                      other.row_names)
                    common_cols = get_common_elements(self.col_names,
                                                      other.col_names)

                    if len(common_rows) == 0:
                        raise Exception("Matrix.__sub__ error: no common rows")

                    if len(common_cols) == 0:
                        raise Exception("Matrix.__sub__ error: no common cols")
                    first = self.get(row_names=common_rows,
                                     col_names=common_cols)
                    second = other.get(row_names=common_rows,
                                       col_names=common_cols)
                else:
                    assert self.shape == other.shape, \
                        "Matrix.__sub__():shape mismatch: " +\
                        str(self.shape) + ' ' + str(other.shape)
                    first = self
                    second = other

                if first.isdiagonal and second.isdiagonal:
                    return type(self)(x=first.x - second.x, isdiagonal=True,
                                      row_names=first.row_names,
                                      col_names=first.col_names)
                elif first.isdiagonal:
                    elem_sub = -1.0 * second.newx
                    for j in range(first.shape[0]):
                        elem_sub[j, j] += first.x[j, 0]
                    return type(self)(x=elem_sub, row_names=first.row_names,
                                      col_names=first.col_names)
                elif second.isdiagonal:
                    elem_sub = first.newx
                    for j in range(second.shape[0]):
                        elem_sub[j, j] -= second.x[j, 0]
                    return type(self)(x=elem_sub, row_names=first.row_names,
                                      col_names=first.col_names)
                else:
                    return type(self)(x=first.x - second.x,
                                      row_names=first.row_names,
                                      col_names=first.col_names)


    def __add__(self, other):
        """Overload of numpy.ndarray.__add__().  Tries to speedup by checking for
            scalars of diagonal matrices on either side of operator

        Parameters
        ----------
        other : scalar,numpy.ndarray,Matrix object
            the thing to add

        Returns
        -------
        Matrix : Matrix

        """
        if np.isscalar(other):
            return type(self)(x=self.x + other,row_names=self.row_names,
                              col_names=self.col_names,isdiagonal=self.isdiagonal)
        if isinstance(other, np.ndarray):
            assert self.shape == other.shape, \
                "Matrix.__add__(): shape mismatch: " +\
                str(self.shape) + ' ' + str(other.shape)
            if self.isdiagonal:
                raise NotImplementedError("Matrix.__add__ not supported for" +
                                          "diagonal self")
            else:
                return type(self)(x=self.x + other, row_names=self.row_names,
                                  col_names=self.col_names)
        elif isinstance(other, Matrix):
            if self.autoalign and other.autoalign \
                    and not self.element_isaligned(other):
                common_rows = get_common_elements(self.row_names,
                                                  other.row_names)
                common_cols = get_common_elements(self.col_names,
                                                  other.col_names)
                if len(common_rows) == 0:
                    raise Exception("Matrix.__add__ error: no common rows")

                if len(common_cols) == 0:
                    raise Exception("Matrix.__add__ error: no common cols")

                first = self.get(row_names=common_rows, col_names=common_cols)
                second = other.get(row_names=common_rows, col_names=common_cols)
            else:
                assert self.shape == other.shape, \
                    "Matrix.__add__(): shape mismatch: " +\
                    str(self.shape) + ' ' + str(other.shape)
                first = self
                second = other
            if first.isdiagonal and second.isdiagonal:
                return type(self)(x=first.x + second.x, isdiagonal=True,
                                  row_names=first.row_names,
                                  col_names=first.col_names)
            elif first.isdiagonal:
                ox = second.newx
                for j in range(first.shape[0]):
                    ox[j, j] += first.__x[j]
                return type(self)(x=ox, row_names=first.row_names,
                                  col_names=first.col_names)
            elif second.isdiagonal:
                x = first.x
                for j in range(second.shape[0]):
                    x[j, j] += second.x[j]
                return type(self)(x=x, row_names=first.row_names,
                                  col_names=first.col_names)
            else:
                return type(self)(x=first.x + second.x,
                                  row_names=first.row_names,
                                  col_names=first.col_names)
        else:
            raise Exception("Matrix.__add__(): unrecognized type for " +
                            "other in __add__: " + str(type(other)))

    def hadamard_product(self, other):
        """Overload of numpy.ndarray.__mult__(): element-wise multiplication.
        Tries to speedup by checking for scalars of diagonal matrices on
        either side of operator

        Parameters
        ----------
        other : scalar,numpy.ndarray,Matrix object
            the thing for element-wise multiplication

        Returns
        -------
        Matrix : Matrix

        """
        if np.isscalar(other):
            return type(self)(x=self.x * other)
        if isinstance(other, np.ndarray):
            assert self.shape == other.shape, \
                "Matrix.hadamard_product(): shape mismatch: " + \
                str(self.shape) + ' ' + str(other.shape)
            if self.isdiagonal:
                raise NotImplementedError("Matrix.hadamard_product() not supported for" +
                                          "diagonal self")
            else:
                return type(self)(x=self.x * other, row_names=self.row_names,
                                  col_names=self.col_names)
        elif isinstance(other, Matrix):
            if self.autoalign and other.autoalign \
                    and not self.element_isaligned(other):
                common_rows = get_common_elements(self.row_names,
                                                  other.row_names)
                common_cols = get_common_elements(self.col_names,
                                                  other.col_names)
                if len(common_rows) == 0:
                    raise Exception("Matrix.hadamard_product error: no common rows")

                if len(common_cols) == 0:
                    raise Exception("Matrix.hadamard_product error: no common cols")

                first = self.get(row_names=common_rows, col_names=common_cols)
                second = other.get(row_names=common_rows, col_names=common_cols)
            else:
                assert self.shape == other.shape, \
                    "Matrix.hadamard_product(): shape mismatch: " + \
                    str(self.shape) + ' ' + str(other.shape)
                first = self
                second = other

            if first.isdiagonal and second.isdiagonal:
                return type(self)(x=first.x * second.x, isdiagonal=True,
                                  row_names=first.row_names,
                                  col_names=first.col_names)
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
                return type(self)(x=first.as_2d * second.as_2d,
                                  row_names=first.row_names,
                                  col_names=first.col_names)
        else:
            raise Exception("Matrix.hadamard_product(): unrecognized type for " +
                            "other: " + str(type(other)))


    def __mul__(self, other):
        """Dot product multiplication overload.  Tries to speedup by
        checking for scalars or diagonal matrices on either side of operator

        Parameters
        ----------
        other : scalar,numpy.ndarray,Matrix object
            the thing the dot product against

        Returns:
            Matrix : Matrix
        """
        if np.isscalar(other):
            return type(self)(x=self.x.copy() * other,
                              row_names=self.row_names,
                              col_names=self.col_names,
                              isdiagonal=self.isdiagonal)
        elif isinstance(other, np.ndarray):
            assert self.shape[1] == other.shape[0], \
                "Matrix.__mul__(): matrices are not aligned: " +\
                str(self.shape) + ' ' + str(other.shape)
            if self.isdiagonal:
                return type(self)(x=np.dot(np.diag(self.__x.flatten()).transpose(),
                                           other))
            else:
                return type(self)(x=np.dot(self.__x, other))
        elif isinstance(other, Matrix):
            if self.autoalign and other.autoalign\
               and not self.mult_isaligned(other):
                common = get_common_elements(self.col_names, other.row_names)
                assert len(common) > 0,"Matrix.__mult__():self.col_names " +\
                                       "and other.row_names" +\
                                       "don't share any common elements.  first 10: " +\
                                       ','.join(self.col_names[:9]) + '...and..' +\
                                       ','.join(other.row_names[:9])
                # these should be aligned
                if isinstance(self, Cov):
                    first = self.get(row_names=common, col_names=common)
                else:
                    first = self.get(row_names=self.row_names, col_names=common)
                if isinstance(other, Cov):
                    second = other.get(row_names=common, col_names=common)
                else:
                    second = other.get(row_names=common,
                                       col_names=other.col_names)

            else:
                assert self.shape[1] == other.shape[0], \
                    "Matrix.__mul__(): matrices are not aligned: " +\
                    str(self.shape) + ' ' + str(other.shape)
                first = self
                second = other
            if first.isdiagonal and second.isdiagonal:
                elem_prod = type(self)(x=first.x.transpose() * second.x,
                                   row_names=first.row_names,
                                   col_names=second.col_names)
                elem_prod.isdiagonal = True
                return elem_prod
            elif first.isdiagonal:
                ox = second.newx
                for j in range(first.shape[0]):
                    ox[j, :] *= first.x[j]
                return type(self)(x=ox, row_names=first.row_names,
                              col_names=second.col_names)
            elif second.isdiagonal:
                x = first.newx
                ox = second.x
                for j in range(first.shape[1]):
                    x[:, j] *= ox[j]
                return type(self)(x=x, row_names=first.row_names,
                              col_names=second.col_names)
            else:
                return type(self)(np.dot(first.x, second.x),
                              row_names=first.row_names,
                              col_names=second.col_names)
        else:
            raise Exception("Matrix.__mul__(): unrecognized " +
                            "other arg type in __mul__: " + str(type(other)))


    def __rmul__(self, other):
        """Reverse order Dot product multiplication overload.

        Parameters
        ----------
        other : scalar,numpy.ndarray,Matrix object
            the thing the dot product against

        Returns
        -------
        Matrix : Matrix

        """

        if np.isscalar(other):
            return type(self)(x=self.x.copy() * other,row_names=self.row_names,\
                              col_names=self.col_names,isdiagonal=self.isdiagonal)
        elif isinstance(other, np.ndarray):
            assert self.shape[0] == other.shape[1], \
                "Matrix.__rmul__(): matrices are not aligned: " +\
                str(other.shape) + ' ' + str(self.shape)
            if self.isdiagonal:
                return type(self)(x=np.dot(other,np.diag(self.__x.flatten()).\
                                           transpose()))
            else:
                return type(self)(x=np.dot(other,self.__x))
        elif isinstance(other, Matrix):
            if self.autoalign and other.autoalign \
                    and not self.mult_isaligned(other):
                common = get_common_elements(self.row_names, other.col_names)
                assert len(common) > 0,"Matrix.__rmul__():self.col_names " +\
                                       "and other.row_names" +\
                                       "don't share any common elements"
                # these should be aligned
                if isinstance(self, Cov):
                    first = self.get(row_names=common, col_names=common)
                else:
                    first = self.get(col_names=self.row_names, row_names=common)
                if isinstance(other, Cov):
                    second = other.get(row_names=common, col_names=common)
                else:
                    second = other.get(col_names=common,
                                       row_names=other.col_names)

            else:
                assert self.shape[0] == other.shape[1], \
                    "Matrix.__rmul__(): matrices are not aligned: " +\
                    str(other.shape) + ' ' + str(self.shape)
                first = other
                second = self
            if first.isdiagonal and second.isdiagonal:
                elem_prod = type(self)(x=first.x.transpose() * second.x,
                                   row_names=first.row_names,
                                   col_names=second.col_names)
                elem_prod.isdiagonal = True
                return elem_prod
            elif first.isdiagonal:
                ox = second.newx
                for j in range(first.shape[0]):
                    ox[j, :] *= first.x[j]
                return type(self)(x=ox, row_names=first.row_names,
                              col_names=second.col_names)
            elif second.isdiagonal:
                x = first.newx
                ox = second.x
                for j in range(first.shape[1]):
                    x[:, j] *= ox[j]
                return type(self)(x=x, row_names=first.row_names,
                              col_names=second.col_names)
            else:
                return type(self)(np.dot(first.x, second.x),
                              row_names=first.row_names,
                              col_names=second.col_names)
        else:
            raise Exception("Matrix.__rmul__(): unrecognized " +
                            "other arg type in __mul__: " + str(type(other)))


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

            u, s, v = la.svd(x, full_matrices=True)
            v = v.transpose()
        except Exception as e:
            print("standard SVD failed: {0}".format(str(e)))
            try:
                v, s, u = la.svd(x.transpose(), full_matrices=True)
                u = u.transpose()
            except Exception as e:
                np.savetxt("failed_svd.dat",x,fmt="%15.6E")
                raise Exception("Matrix.__set_svd(): " +
                                "unable to compute SVD of self.x, " +
                                "saved matrix to 'failed_svd.dat' -- {0}".\
                                format(str(e)))

        col_names = ["left_sing_vec_" + str(i + 1) for i in range(u.shape[1])]
        self.__u = Matrix(x=u, row_names=self.row_names,
                          col_names=col_names, autoalign=False)

        sing_names = ["sing_val_" + str(i + 1) for i in range(s.shape[0])]
        self.__s = Matrix(x=np.atleast_2d(s).transpose(), row_names=sing_names,
                          col_names=sing_names, isdiagonal=True,
                          autoalign=False)

        col_names = ["right_sing_vec_" + str(i + 1) for i in range(v.shape[0])]
        self.__v = Matrix(v, row_names=self.col_names, col_names=col_names,
                          autoalign=False)

    def mult_isaligned(self, other):
        """check if matrices are aligned for dot product multiplication

        Parameters
        ----------
        other : (Matrix)

        Returns
        -------
        bool : bool
            True if aligned, False if not aligned
        """
        assert isinstance(other, Matrix), \
            "Matrix.isaligned(): other argumnent must be type Matrix, not: " +\
            str(type(other))
        if self.col_names == other.row_names:
            return True
        else:
            return False


    def element_isaligned(self, other):
        """check if matrices are aligned for element-wise operations

        Parameters
        ----------
        other : Matrix

        Returns
        -------
        bool : bool
            True if aligned, False if not aligned
        """
        assert isinstance(other, Matrix), \
            "Matrix.isaligned(): other argument must be type Matrix, not: " +\
            str(type(other))
        if self.row_names == other.row_names \
                and self.col_names == other.col_names:
            return True
        else:
            return False


    @property
    def newx(self):
        """return a copy of x

        Returns
        -------
        numpy.ndarray : numpy.ndarray

        """
        return self.__x.copy()


    @property
    def x(self):
        """return a reference to x

        Returns
        -------
        numpy.ndarray : numpy.ndarray

        """
        return self.__x

    @property
    def as_2d(self):
        """ get a 2D representation of x.  If not self.isdiagonal, simply
        return reference to self.x, otherwise, constructs and returns
        a 2D, diagonal ndarray

        Returns
        -------
        numpy.ndarray : numpy.ndarray

        """
        if not self.isdiagonal:
            return self.x
        return np.diag(self.x.flatten())

    @property
    def shape(self):
        """get the implied, 2D shape of self

        Returns
        -------
        tuple : tuple
            length 2 tuple of ints

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
        """ length of second dimension

        Returns
        -------
        int : int
            number of columns

        """
        return self.shape[1]

    @property
    def nrow(self):
        """ length of first dimensions

        Returns
        -------
        int : int
            number of rows

        """
        return self.shape[0]

    @property
    def T(self):
        """wrapper function for Matrix.transpose() method

        """
        return self.transpose


    @property
    def transpose(self):
        """transpose operation of self

        Returns
        -------
        Matrix : Matrix
            transpose of self

        """
        if not self.isdiagonal:
            return type(self)(x=self.__x.copy().transpose(),
                              row_names=self.col_names,
                              col_names=self.row_names,
                              autoalign=self.autoalign)
        else:
            return type(self)(x=self.__x.copy(), row_names=self.row_names,
                              col_names=self.col_names,
                              isdiagonal=True, autoalign=self.autoalign)


    @property
    def inv(self):
        """inversion operation of self

        Returns
        -------
        Matrix : Matrix
            inverse of self

       """

        if self.isdiagonal:
            inv = 1.0 / self.__x
            if (np.any(~np.isfinite(inv))):
                idx = np.isfinite(inv)
                np.savetxt("testboo.dat",idx)
                invalid = [self.row_names[i] for i in range(idx.shape[0]) if idx[i] == 0.0]
                raise Exception("Matrix.inv has produced invalid floating points " +
                                " for the following elements:" + ','.join(invalid))
            return type(self)(x=inv, isdiagonal=True,
                              row_names=self.row_names,
                              col_names=self.col_names,
                              autoalign=self.autoalign)
        else:
            return type(self)(x=la.inv(self.__x), row_names=self.row_names,
                              col_names=self.col_names,
                              autoalign=self.autoalign)

    def get_maxsing(self,eigthresh=1.0e-5):
        """ Get the number of singular components with a singular
        value ratio greater than or equal to eigthresh

        Parameters
        ----------
        eigthresh : float
            the ratio of the largest to smallest singular value

        Returns
        -------
        int : int
            number of singular components

        """
        #sthresh =np.abs((self.s.x / self.s.x[0]) - eigthresh)
        sthresh = self.s.x.flatten()/self.s.x[0]
        ising = 0
        for i,st in enumerate(sthresh):
            if st > eigthresh:
                ising += 1
                #return max(1,i)
            else:
                break
        #return max(1,np.argmin(sthresh))
        return max(1,ising)

    def pseudo_inv_components(self,maxsing=None,eigthresh=1.0e-5):
        """ Get the truncated SVD components

        Parameters
        ----------
        maxsing : int
            the number of singular components to use.  If None,
            maxsing is calculated using Matrix.get_maxsing() and eigthresh
        eigthresh : float
            the ratio of largest to smallest singular components to use
            for truncation.  Ignored if maxsing is not None

        Returns
        -------
        u : Matrix
            truncated left singular vectors
        s : Matrix
            truncated singular value matrix
        v : Matrix
            truncated right singular vectors

        """

        if maxsing is None:
            maxsing = self.get_maxsing(eigthresh=eigthresh)

        s = self.s[:maxsing,:maxsing]
        v = self.v[:,:maxsing]
        u = self.u[:,:maxsing]
        return u,s,v

    def pseudo_inv(self,maxsing=None,eigthresh=1.0e-5):
        """ The pseudo inverse of self.  Formed using truncated singular
        value decomposition and Matrix.pseudo_inv_components

        Parameters
        ----------
        maxsing : int
            the number of singular components to use.  If None,
            maxsing is calculated using Matrix.get_maxsing() and eigthresh
        eigthresh : float
            the ratio of largest to smallest singular components to use
            for truncation.  Ignored if maxsing is not None

        Returns
        -------
        Matrix : Matrix
        """
        if maxsing is None:
            maxsing = self.get_maxsing(eigthresh=eigthresh)
        full_s = self.full_s.T
        for i in range(self.s.shape[0]):
            if i <= maxsing:
                full_s.x[i,i] = 1.0 / full_s.x[i,i]
            else:
                full_s.x[i,i] = 0.0
        return self.v * full_s * self.u.T

    @property
    def sqrt(self):
        """square root operation

        Returns
        -------
        Matrix : Matrix
            square root of self

        """
        if self.isdiagonal:
            return type(self)(x=np.sqrt(self.__x), isdiagonal=True,
                              row_names=self.row_names,
                              col_names=self.col_names,
                              autoalign=self.autoalign)
        elif self.shape[1] == 1: #a vector
            return type(self)(x=np.sqrt(self.__x), isdiagonal=False,
                              row_names=self.row_names,
                              col_names=self.col_names,
                              autoalign=self.autoalign)
        else:
            return type(self)(x=la.sqrtm(self.__x), row_names=self.row_names,
                              col_names=self.col_names,
                              autoalign=self.autoalign)
    @property
    def full_s(self):
        """ Get the full singular value matrix of self

        Returns
        -------
        Matrix : Matrix

        """
        x = np.zeros((self.shape),dtype=np.float32)

        x[:self.s.shape[0],:self.s.shape[0]] = self.s.as_2d
        s = Matrix(x=x, row_names=self.row_names,
                          col_names=self.col_names, isdiagonal=False,
                          autoalign=False)
        return s

    @property
    def s(self):
        """the singular value (diagonal) Matrix

        Returns
        -------
        Matrix : Matrix

        """
        if self.__s is None:
            self.__set_svd()
        return self.__s


    @property
    def u(self):
        """the left singular vector Matrix

        Returns
        -------
        Matrix : Matrix

        """
        if self.__u is None:
            self.__set_svd()
        return self.__u


    @property
    def v(self):
        """the right singular vector Matrix

        Returns
        -------
        Matrix : Matrix

        """
        if self.__v is None:
            self.__set_svd()
        return self.__v

    @property
    def zero2d(self):
        """ get an 2D instance of self with all zeros

        Returns
        -------
        Matrix : Matrix

        """
        return type(self)(x=np.atleast_2d(np.zeros((self.shape[0],self.shape[1]))),
                   row_names=self.row_names,
                   col_names=self.col_names,
                   isdiagonal=False)


    @staticmethod
    def find_rowcol_indices(names,row_names,col_names,axis=None):
        self_row_idxs = {row_names[i]: i for i in range(len(row_names))}
        self_col_idxs = {col_names[i]: i for i in range(len(col_names))}

        scol = set(col_names)
        srow = set(row_names)
        row_idxs = []
        col_idxs = []
        for name in names:
            name = name.lower()
            if name not in scol \
                    and name not in srow:
                raise Exception('Matrix.indices(): name not found: ' + name)
            if name in scol:
                col_idxs.append(self_col_idxs[name])
            if name.lower() in srow:
                row_idxs.append(self_row_idxs[name])
        if axis is None:
            return np.array(row_idxs, dtype=np.int32), \
                   np.array(col_idxs, dtype=np.int32)
        elif axis == 0:
            if len(row_idxs) != len(names):
                raise Exception("Matrix.indices(): " +
                                "not all names found in row_names")
            return np.array(row_idxs, dtype=np.int32)
        elif axis == 1:
            if len(col_idxs) != len(names):
                raise Exception("Matrix.indices(): " +
                                "not all names found in col_names")
            return np.array(col_idxs, dtype=np.int32)
        else:
            raise Exception("Matrix.indices(): " +
                            "axis argument must 0 or 1, not:" + str(axis))

    def indices(self, names, axis=None):
        """get the row and col indices of names. If axis is None, two ndarrays
                are returned, corresponding the indices of names for each axis

        Parameters
        ----------
        names : iterable
            column and/or row names
        axis : (int) (optional)
            the axis to search.

        Returns
        -------
        numpy.ndarray : numpy.ndarray
            indices of names.

        """
        return Matrix.find_rowcol_indices(names,self.row_names,self.col_names,axis=axis)




    def old_indices(self, names, axis=None):
        """get the row and col indices of names. If axis is None, two ndarrays
                are returned, corresponding the indices of names for each axis

        Parameters
        ----------
        names : iterable
            column and/or row names
        axis : (int) (optional)
            the axis to search.

        Returns
        -------
        numpy.ndarray : numpy.ndarray
            indices of names.

        """
        warnings.warn("Matrix.old_indices() is deprecated - only here for testing. Use Matrix.indices()")
        row_idxs, col_idxs = [], []
        for name in names:
            if name.lower() not in self.col_names \
                    and name.lower() not in self.row_names:
                raise Exception('Matrix.indices(): name not found: ' + name)
            if name.lower() in self.col_names:
                col_idxs.append(self.col_names.index(name))
            if name.lower() in self.row_names:
                row_idxs.append(self.row_names.index(name))
        if axis is None:
            return np.array(row_idxs, dtype=np.int32),\
                np.array(col_idxs, dtype=np.int32)
        elif axis == 0:
            if len(row_idxs) != len(names):
                raise Exception("Matrix.indices(): " +
                                "not all names found in row_names")
            return np.array(row_idxs, dtype=np.int32)
        elif axis == 1:
            if len(col_idxs) != len(names):
                raise Exception("Matrix.indices(): " +
                                "not all names found in col_names")
            return np.array(col_idxs, dtype=np.int32)
        else:
            raise Exception("Matrix.indices(): " +
                            "axis argument must 0 or 1, not:" + str(axis))


    def align(self, names, axis=None):
        """reorder self by names.  If axis is None, reorder both indices

        Parameters
        ----------
        names : iterable
            names in rowS and\or columnS
        axis : (int)
            the axis to reorder. if None, reorder both axes

        """
        if not isinstance(names, list):
            names = [names]
        row_idxs, col_idxs = self.indices(names)
        if self.isdiagonal or isinstance(self, Cov):
            assert row_idxs.shape == col_idxs.shape
            assert row_idxs.shape[0] == self.shape[0]
            if self.isdiagonal:
                self.__x = self.__x[row_idxs]
            else:
                self.__x = self.__x[row_idxs, :]
                self.__x = self.__x[:, col_idxs]
            row_names = []
            [row_names.append(self.row_names[i]) for i in row_idxs]
            self.row_names, self.col_names = row_names, row_names

        else:
            if axis is None:
                raise Exception("Matrix.align(): must specify axis in " +
                                "align call for non-diagonal instances")
            if axis == 0:
                assert row_idxs.shape[0] == self.shape[0], \
                    "Matrix.align(): not all names found in self.row_names"
                self.__x = self.__x[row_idxs, :]
                row_names = []
                [row_names.append(self.row_names[i]) for i in row_idxs]
                self.row_names = row_names
            elif axis == 1:
                assert col_idxs.shape[0] == self.shape[1], \
                    "Matrix.align(): not all names found in self.col_names"
                self.__x = self.__x[:, col_idxs]
                col_names = []
                [col_names.append(self.col_names[i]) for i in row_idxs]
                self.col_names = col_names
            else:
                raise Exception("Matrix.align(): axis argument to align()" +
                                " must be either 0 or 1")


    def get(self, row_names=None, col_names=None, drop=False):
        """get a new Matrix instance ordered on row_names or col_names

        Parameters
        ----------
        row_names : iterable
            row_names for new Matrix
        col_names : iterable
            col_names for new Matrix
        drop : bool
            flag to remove row_names and/or col_names

        Returns
        -------
        Matrix : Matrix

        """
        if row_names is None and col_names is None:
            raise Exception("Matrix.get(): must pass at least" +
                            " row_names or col_names")

        if row_names is not None and not isinstance(row_names, list):
            row_names = [row_names]
        if col_names is not None and not isinstance(col_names, list):
            col_names = [col_names]

        if isinstance(self,Cov) and (row_names is None or col_names is None ):
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
        return type(self)(x=self.newx,row_names=self.row_names,
                          col_names=self.col_names,
                          isdiagonal=self.isdiagonal,autoalign=self.autoalign)


    def drop(self, names, axis):
        """ drop elements from self in place

        Parameters
        ----------
        names : iterable
            names to drop
        axis : (int)
            the axis to drop from. must be in [0,1]

        """
        if axis is None:
            raise Exception("Matrix.drop(): axis arg is required")
        if not isinstance(names, list):
            names = [names]
        if axis == 1:
            assert len(names) < self.shape[1], "can't drop all names along axis 1"
        else:
            assert len(names) < self.shape[0], "can't drop all names along axis 0"

        idxs = self.indices(names, axis=axis)



        if self.isdiagonal:
            self.__x = np.delete(self.__x, idxs, 0)
            keep_names = [name for name in self.row_names if name not in names]
            assert len(keep_names) == self.__x.shape[0],"shape-name mismatch:"+\
                   "{0}:{0}".format(len(keep_names),self.__x.shape)
            self.row_names = keep_names
            self.col_names = copy.deepcopy(keep_names)
            # idxs = np.sort(idxs)
            # for idx in idxs[::-1]:
            #     del self.row_names[idx]
            #     del self.col_names[idx]
        elif isinstance(self,Cov):
            self.__x = np.delete(self.__x, idxs, 0)
            self.__x = np.delete(self.__x, idxs, 1)
            keep_names = [name for name in self.row_names if name not in names]

            assert len(keep_names) == self.__x.shape[0],"shape-name mismatch:"+\
                   "{0}:{0}".format(len(keep_names),self.__x.shape)
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
            assert len(keep_names) == self.__x.shape[0],"shape-name mismatch:"+\
                   "{0}:{0}".format(len(keep_names),self.__x.shape)
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
            assert len(keep_names) == self.__x.shape[1],"shape-name mismatch:"+\
                   "{0}:{0}".format(len(keep_names),self.__x.shape)
            self.col_names = keep_names
            # idxs = np.sort(idxs)
            # for idx in idxs[::-1]:
            #     del self.col_names[idx]
        else:
            raise Exception("Matrix.drop(): axis argument must be 0 or 1")


    def extract(self, row_names=None, col_names=None):
        """wrapper method that Matrix.gets() then Matrix.drops() elements.
        one of row_names or col_names must be not None.

        Parameters
        ----------
        row_names : iterable
            row names to extract
        col_names : (enumerate)
            col_names to extract

        Returns
        -------
        Matrix : Matrix

        """
        if row_names is None and col_names is None:
            raise Exception("Matrix.extract() " +
                            "row_names and col_names both None")
        extract = self.get(row_names, col_names, drop=True)
        return extract

    def get_diagonal_vector(self, col_name="diag"):
        """Get a new Matrix instance that is the diagonal of self.  The
        shape of the new matrix is (self.shape[0],1).  Self must be square

        Parameters:
            col_name : str
                the name of the column in the new Matrix

        Returns:
            Matrix : Matrix
        """
        assert self.shape[0] == self.shape[1]
        assert not self.isdiagonal
        assert isinstance(col_name,str)
        return type(self)(x=np.atleast_2d(np.diag(self.x)).transpose(),
                          row_names=self.row_names,
                          col_names=[col_name],isdiagonal=False)


    def to_coo(self,filename,droptol=None,chunk=None):
        """write a PEST-compatible binary file.  The data format is
        [int,int,float] for i,j,value.  It is autodetected during
        the read with Matrix.from_binary().

        Parameters
        ----------
        filename : str
            filename to save binary file
        droptol : float
            absolute value tolerance to make values smaller than zero.  Default is None
        chunk : int
            number of elements to write in a single pass.  Default is None

        """
        if self.isdiagonal:
            #raise NotImplementedError()
            self.__x = self.as_2d
            self.isdiagonal = False
        if droptol is not None:
            self.x[np.abs(self.x) < droptol] = 0.0
        f = open(filename, 'wb')
        #print("counting nnz")
        nnz = np.count_nonzero(self.x) #number of non-zero entries
        # write the header
        header = np.array((self.shape[1], self.shape[0], nnz),
                          dtype=self.binary_header_dt)
        header.tofile(f)
        # get the indices of non-zero entries
        #print("getting nnz idxs")
        row_idxs, col_idxs = np.nonzero(self.x)

        if chunk is None:
            flat = self.x[row_idxs, col_idxs].flatten()
            data = np.core.records.fromarrays([row_idxs,col_idxs,flat],dtype=self.coo_rec_dt)
            data.tofile(f)
        else:

            start,end = 0,min(chunk,row_idxs.shape[0])
            while True:
                #print(row_idxs[start],row_idxs[end])
                #print("chunk",start,end)
                flat = self.x[row_idxs[start:end],col_idxs[start:end]].flatten()
                data = np.core.records.fromarrays([row_idxs[start:end],col_idxs[start:end],
                                                   flat],
                                                  dtype=self.coo_rec_dt)
                data.tofile(f)
                if end == row_idxs.shape[0]:
                    break
                start = end
                end = min(row_idxs.shape[0],start + chunk)


        for name in self.col_names:
            if len(name) > self.par_length:
                name = name[:self.par_length - 1]
            elif len(name) < self.par_length:
                for i in range(len(name), self.par_length):
                    name = name + ' '
            f.write(name.encode())
        for name in self.row_names:
            if len(name) > self.obs_length:
                name = name[:self.obs_length - 1]
            elif len(name) < self.obs_length:
                for i in range(len(name), self.obs_length):
                    name = name + ' '
            f.write(name.encode())
        f.close()



    def to_binary(self, filename,droptol=None, chunk=None):
        """write a PEST-compatible binary file.  The format is the same
        as the format used to storage a PEST Jacobian matrix

        Parameters
        ----------
        filename : str
            filename to save binary file
        droptol : float
            absolute value tolerance to make values smaller than zero.  Default is None
        chunk : int
            number of elements to write in a single pass.  Default is None

        """
        if np.any(np.isnan(self.x)):
            raise Exception("Matrix.to_binary(): nans found")
        if self.isdiagonal:
            #raise NotImplementedError()
            self.__x = self.as_2d
            self.isdiagonal = False
        if droptol is not None:
            self.x[np.abs(self.x) < droptol] = 0.0
        f = open(filename, 'wb')
        nnz = np.count_nonzero(self.x) #number of non-zero entries
        # write the header
        header = np.array((-self.shape[1], -self.shape[0], nnz),
                          dtype=self.binary_header_dt)
        header.tofile(f)
        # get the indices of non-zero entries
        row_idxs, col_idxs = np.nonzero(self.x)
        icount = row_idxs + 1 + col_idxs * self.shape[0]
        # flatten the array
        #flat = self.x[row_idxs, col_idxs].flatten()
        # zip up the index position and value pairs
        #data = np.array(list(zip(icount, flat)), dtype=self.binary_rec_dt)


        if chunk is None:
            flat = self.x[row_idxs, col_idxs].flatten()
            data = np.core.records.fromarrays([icount, flat], dtype=self.binary_rec_dt)
            # write
            data.tofile(f)
        else:
            start,end = 0,min(chunk,row_idxs.shape[0])
            while True:
                #print(row_idxs[start],row_idxs[end])
                flat = self.x[row_idxs[start:end], col_idxs[start:end]].flatten()
                data = np.core.records.fromarrays([icount[start:end],
                                                   flat],
                                                  dtype=self.binary_rec_dt)
                data.tofile(f)
                if end == row_idxs.shape[0]:
                    break
                start = end
                end = min(row_idxs.shape[0],start + chunk)


        for name in self.col_names:
            if len(name) > self.par_length:
                name = name[:self.par_length - 1]
            elif len(name) < self.par_length:
                for i in range(len(name), self.par_length):
                    name = name + ' '
            f.write(name.encode())
        for name in self.row_names:
            if len(name) > self.obs_length:
                name = name[:self.obs_length - 1]
            elif len(name) < self.obs_length:
                for i in range(len(name), self.obs_length):
                    name = name + ' '
            f.write(name.encode())
        f.close()


    @classmethod
    def from_binary(cls,filename):
        """class method load from PEST-compatible binary file into a
        Matrix instance

        Parameters
        ----------
        filename : str
            filename to read

        Returns
        -------
        Matrix : Matrix

        """
        x,row_names,col_names = Matrix.read_binary(filename)
        if np.any(np.isnan(x)):
            warnings.warn("Matrix.from_binary(): nans in matrix")
        return cls(x=x, row_names=row_names, col_names=col_names)

    @staticmethod
    def read_binary(filename, sparse=False):


        f = open(filename, 'rb')
        # the header datatype
        itemp1, itemp2, icount = np.fromfile(f, Matrix.binary_header_dt, 1)[0]
        if itemp1 > 0 and itemp2 < 0 and icount < 0:
            print(" WARNING: it appears this file was \n" +\
                  " written with 'sequential` " +\
                  " binary fortran specification\n...calling " +\
                  " Matrix.from_fortranfile()")
            f.close()
            return Matrix.from_fortranfile(filename)
        ncol, nrow = abs(itemp1), abs(itemp2)
        if itemp1 >= 0:
            # raise TypeError('Matrix.from_binary(): Jco produced by ' +
            #                 'deprecated version of PEST,' +
            #                 'Use JcoTRANS to convert to new format')
            print("'COO' format detected...")

            data = np.fromfile(f, Matrix.coo_rec_dt, icount)
            if sparse:
                data = scipy.sparse.coo_matrix((data["dtemp"],(data["i"],data['j'])),shape=(nrow,ncol))
            else:
                x = np.zeros((nrow, ncol))
                x[data['i'], data['j']] = data["dtemp"]
                data = x
        else:

            # read all data records
            # using this a memory hog, but really fast
            data = np.fromfile(f, Matrix.binary_rec_dt, icount)
            icols = ((data['j'] - 1) // nrow) + 1
            irows = data['j'] - ((icols - 1) * nrow)
            if sparse:
                data = scipy.sparse.coo_matrix((data["dtemp"],(irows-1,icols-1)),shape=(nrow,ncol))
            else:
                x = np.zeros((nrow, ncol))
                x[irows - 1, icols - 1] = data["dtemp"]
                data = x
        # read obs and parameter names
        col_names = []
        row_names = []
        for j in range(ncol):
            name = struct.unpack(str(Matrix.par_length) + "s",
                                 f.read(Matrix.par_length))[0]\
                                  .strip().lower().decode()
            col_names.append(name)
        for i in range(nrow):
            name = struct.unpack(str(Matrix.obs_length) + "s",
                                 f.read(Matrix.obs_length))[0]\
                                  .strip().lower().decode()
            row_names.append(name)
        f.close()
        assert len(row_names) == data.shape[0],\
          "Matrix.read_binary() len(row_names) (" + str(len(row_names)) +\
          ") != x.shape[0] (" + str(data.shape[0]) + ")"
        assert len(col_names) == data.shape[1],\
          "Matrix.read_binary() len(col_names) (" + str(len(col_names)) +\
          ") != self.shape[1] (" + str(data.shape[1]) + ")"
        return data,row_names,col_names


    @classmethod
    def from_fortranfile(cls, filename):
        """ a binary load method to accommodate one of the many
            bizarre fortran binary writing formats

        Parameters
        ----------
        filename : str
            name of the binary matrix file

        Returns
        -------
        Matrix : Matrix

        """
        f = FortranFile(filename,mode='r')
        itemp1, itemp2 = f.read_ints()
        icount = f.read_ints()
        if itemp1 >= 0:
           raise TypeError('Matrix.from_binary(): Jco produced by ' +
                           'deprecated version of PEST,' +
                           'Use JcoTRANS to convert to new format')
        ncol, nrow = abs(itemp1), abs(itemp2)
        data = []
        for i in range(icount):
            d = f.read_record(Matrix.binary_rec_dt)[0]
            data.append(d)
        data = np.array(data,dtype=Matrix.binary_rec_dt)
        icols = ((data['j'] - 1) // nrow) + 1
        irows = data['j'] - ((icols - 1) * nrow)
        x = np.zeros((nrow, ncol))
        x[irows - 1, icols - 1] = data["dtemp"]
        row_names = []
        col_names = []
        for j in range(ncol):
            name = f.read_record("|S12")[0].strip().decode()
            col_names.append(name)
        #obs_rec = np.dtype((np.str_, self.obs_length))
        for i in range(nrow):
            name = f.read_record("|S20")[0].strip().decode()
            row_names.append(name)
        assert len(row_names) == x.shape[0],\
          "Matrix.from_fortranfile() len(row_names) (" + \
          str(len(row_names)) +\
          ") != self.shape[0] (" + str(x.shape[0]) + ")"
        assert len(col_names) == x.shape[1],\
          "Matrix.from_fortranfile() len(col_names) (" + \
          str(len(col_names)) +\
          ") != self.shape[1] (" + str(x.shape[1]) + ")"
        return cls(x=x,row_names=row_names,col_names=col_names)

    def to_ascii(self, out_filename, icode=2):
        """write a PEST-compatible ASCII Matrix/vector file

        Parameters
        ----------
        out_filename : str
            output filename
        icode : (int)
            PEST-style info code for Matrix style

        """
        nrow, ncol = self.shape
        f_out = open(out_filename, 'w')
        f_out.write(' {0:7.0f} {1:7.0f} {2:7.0f}\n'.
                    format(nrow, ncol, icode))
        f_out.close()
        f_out = open(out_filename,'ab')
        if self.isdiagonal:
            x = np.diag(self.__x[:, 0])
        else:
            x = self.__x
        np.savetxt(f_out, x, fmt='%15.7E', delimiter='')
        f_out.close()
        f_out = open(out_filename,'a')
        if icode == 1:
            f_out.write('* row and column names\n')
            for r in self.row_names:
                f_out.write(r + '\n')
        else:
            f_out.write('* row names\n')
            for r in self.row_names:
                f_out.write(r + '\n')
            f_out.write('* column names\n')
            for c in self.col_names:
                f_out.write(c + '\n')
            f_out.close()


    @classmethod
    def from_ascii(cls,filename):
        """load a pest-compatible ASCII Matrix/vector file into a
        Matrix instance

        Parameters
        ----------
        filename : str
            name of the file to read

        """
        x,row_names,col_names,isdiag = Matrix.read_ascii(filename)
        return cls(x=x,row_names=row_names,col_names=col_names,isdiagonal=isdiag)


    @staticmethod
    def read_ascii(filename):

        f = open(filename, 'r')
        raw = f.readline().strip().split()
        nrow, ncol, icode = int(raw[0]), int(raw[1]), int(raw[2])
        #x = np.fromfile(f, dtype=self.double, count=nrow * ncol, sep=' ')
        # this painfully slow and ugly read is needed to catch the
        # fortran floating points that have 3-digit exponents,
        # which leave out the base (e.g. 'e') : "-1.23455+300"
        count = 0
        x = []
        while True:
            line = f.readline()
            if line == '':
                raise Exception("Matrix.from_ascii() error: EOF")
            raw = line.strip().split()
            for r in raw:
                try:
                    x.append(float(r))
                except:
                    # overflow
                    if '+' in r:
                        x.append(1.0e+30)
                    # underflow
                    elif '-' in r:
                        x.append(0.0)
                    else:
                        raise Exception("Matrix.from_ascii() error: " +
                                        " can't cast " + r + " to float")
                count += 1
                if count == (nrow * ncol):
                    break
            if count == (nrow * ncol):
                    break

        x = np.array(x,dtype=Matrix.double)
        x.resize(nrow, ncol)
        line = f.readline().strip().lower()
        if not line.startswith('*'):
            raise Exception('Matrix.from_ascii(): error loading ascii file," +\
                "line should start with * not ' + line)
        if 'row' in line and 'column' in line:
            assert nrow == ncol
            names = []
            for i in range(nrow):
                line = f.readline().strip().lower()
                names.append(line)
            row_names = copy.deepcopy(names)
            col_names = names

        else:
            names = []
            for i in range(nrow):
                line = f.readline().strip().lower()
                names.append(line)
            row_names = names
            line = f.readline().strip().lower()
            assert "column" in line, \
                "Matrix.from_ascii(): line should be * column names " +\
                "instead of: " + line
            names = []
            for j in range(ncol):
                line = f.readline().strip().lower()
                names.append(line)
            col_names = names
        f.close()
        # test for diagonal
        isdiagonal=False
        if nrow == ncol:
            diag = np.diag(np.diag(x))
            diag_tol = 1.0e-6
            diag_delta = np.abs(diag.sum() - x.sum())
            if diag_delta < diag_tol:
                isdiagonal = True
                x = np.atleast_2d(np.diag(x)).transpose()
        return x,row_names,col_names,isdiagonal

    def df(self):
        """wrapper of Matrix.to_dataframe()
        """
        return self.to_dataframe()

    @classmethod
    def from_dataframe(cls, df):
        """ class method to create a new Matrix instance from a
         pandas.DataFrame

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        Matrix : Matrix

        """
        assert isinstance(df, pandas.DataFrame)
        row_names = copy.deepcopy(list(df.index))
        col_names = copy.deepcopy(list(df.columns))
        return cls(x=df.as_matrix(),row_names=row_names,col_names=col_names)


    @classmethod
    def from_names(cls,row_names,col_names,isdiagonal=False,autoalign=True, random=False):
        """ class method to create a new Matrix instance from
        row names and column names, filled with trash

        Parameters
        ----------
            row_names : iterable
                row names for the new matrix
            col_names : iterable
                col_names for the new matrix
            isdiagonal : bool
                flag for diagonal matrix. Default is False
            autoalign : bool
                flag for autoaligning new matrix
                during linear algebra calcs. Default
                is True
            random : bool
                flag for contents of the trash matrix.
                If True, fill with random numbers, if False, fill with zeros
                Default is False
        Returns
        -------
            mat : Matrix
                the new Matrix instance

        """
        if random:
            return cls(x=np.random.random((len(row_names), len(col_names))), row_names=row_names,
                       col_names=col_names, isdiagonal=isdiagonal, autoalign=autoalign)
        else:
            return cls(x=np.empty((len(row_names),len(col_names))),row_names=row_names,
                      col_names=col_names,isdiagonal=isdiagonal,autoalign=autoalign)


    def to_dataframe(self):
        """return a pandas.DataFrame representation of the Matrix object

        Returns
        -------
        pandas.DataFrame : pandas.DataFrame

        """
        if self.isdiagonal:
            x = np.diag(self.__x[:, 0])
        else:
            x = self.__x
        return pandas.DataFrame(data=x,index=self.row_names,columns=self.col_names)


    def to_sparse(self, trunc=0.0):
        """get the COO sparse Matrix representation of the Matrix

        Returns
        -------
        scipy.sparse.Matrix : scipy.sparse.Matrix

        """
        try:
            import scipy.sparse as sparse
        except:
            raise Exception("mat.to_sparse() error importing scipy.sparse")
        iidx, jidx = [], []
        data = []
        nrow, ncol = self.shape
        for i in range(nrow):
            for j in range(ncol):
                val = self.x[i,j]
                if val > trunc:
                    iidx.append(i)
                    jidx.append(j)
                    data.append(val)
        # csr_Matrix( (data,(row,col)), shape=(3,3)
        return sparse.coo_matrix((data, (iidx, jidx)), shape=(self.shape))


    def extend(self,other,inplace=False):
        """ extend self with the elements of other.

        Parameters
        ----------
        other : (Matrix)
            the Matrix to extend self by
        inplace : bool
            inplace = True not implemented

        Returns
        -------
        Matrix : Matrix
            if not inplace

        """
        if inplace == True:
            raise NotImplementedError()
        assert len(set(self.row_names).intersection(set(other.row_names))) == 0
        assert len(set(self.col_names).intersection(set(other.col_names))) == 0
        assert type(self) == type(other)
        new_row_names = copy.copy(self.row_names)
        new_row_names.extend(other.row_names)
        new_col_names = copy.copy(self.col_names)
        new_col_names.extend(other.col_names)

        new_x = np.zeros((len(new_row_names),len(new_col_names)))
        new_x[0:self.shape[0],0:self.shape[1]] = self.as_2d
        new_x[self.shape[0]:self.shape[0]+other.shape[0],
              self.shape[1]:self.shape[1]+other.shape[1]] = other.as_2d
        isdiagonal = True
        if not self.isdiagonal or not other.isdiagonal:
            isdiagonal = False

        return type(self)(x=new_x,row_names=new_row_names,
                           col_names=new_col_names,isdiagonal=isdiagonal)





class Jco(Matrix):
    """a thin wrapper class to get more intuitive attribute names.  Functions
    exactly like Matrix
    """
    def __init(self, **kwargs):
        """ Jco constuctor takes the same arguments as Matrix.

        Parameters
        ----------
        **kwargs : (dict)
            constructor arguments for Matrix

        Returns
        -------
        Jco : Jco

        """

        super(Jco, self).__init__(kwargs)


    @property
    def par_names(self):
        """ thin wrapper around Matrix.col_names

        Returns
        -------
        list : list
            parameter names

        """
        return self.col_names


    @property
    def obs_names(self):
        """ thin wrapper around Matrix.row_names

        Returns
        -------
        list : list
            observation names

        """
        return self.row_names


    @property
    def npar(self):
        """ number of parameters in the Jco

        Returns
        -------
        int : int
            number of parameters (columns)

        """
        return self.shape[1]


    @property
    def nobs(self):
        """ number of observations in the Jco

        Returns
        -------
        int : int
            number of observations (rows)

        """
        return self.shape[0]

    def replace_cols(self, other, parnames=None):
        """
        Replaces columns in one Matrix with columns from another.
        Intended for Jacobian matrices replacing parameters.

        Parameters
        ----------
        other: Matrix
            Matrix to use for replacing columns in self

        parnames: list
            parameter (column) names to use in other.  If None, all
            columns in other are used

        """
        assert len(set(self.col_names).intersection(set(other.col_names))) > 0
        if not parnames:
            parnames = other.col_names
        assert len(set(self.col_names).intersection(set(other.col_names))) == len(parnames)

        assert len(set(self.row_names).intersection(set(other.row_names))) == len(self.row_names)
        assert type(self) == type(other)

        # re-sort other by rows to be sure they line up with self
        try:
            other = other.get(row_names=self.row_names)
        except:
            raise Exception('could not align rows of the two matrices')

        # replace the columns in self with those from other
        selfobs = np.array(self.col_names)
        otherobs = np.array(other.col_names)
        selfidx = [np.where(np.array(selfobs) == i)[0][0] for i in parnames]
        otheridx = [np.where(np.array(otherobs) == i)[0][0] for i in parnames]
        self.x[:,selfidx] = other.x[:,otheridx]


    @classmethod
    def from_pst(cls,pst, random=False):
        """construct a new empty Jco from a control file filled
        with trash

        Parameters
        ----------
            pst : Pst
                a control file instance.  If type is 'str',
                Pst is loaded from filename
            random : bool
                flag for contents of the trash matrix.
                If True, fill with random numbers, if False, fill with zeros
                Default is False
        Return
        ------
            jco : Jco
                the new Jco instance

        """

        if isinstance(pst,str):
            pst = Pst(pst)

        return Jco.from_names(pst.obs_names, pst.adj_par_names, random=random)

class Cov(Matrix):
    """a subclass of Matrix for handling diagonal or dense Covariance matrices
        todo:block diagonal
    """
    def __init__(self, x=None, names=[], row_names=[], col_names=[],
                 isdiagonal=False, autoalign=True):
        """ Cov constructor.


        Parameters
        ----------
        x : numpy.ndarray
            elements in Cov
        names : iterable
            names for both columns and rows
        row_names : iterable
            names for rows
        col_names : iterable
            names for columns
        isdiagonal : bool
            diagonal Matrix flag
        autoalign : bool
            autoalignment flag

        Returns
        -------
        Cov : Cov

        """
        self.__identity = None
        self.__zero = None
        #if len(row_names) > 0 and len(col_names) > 0:
        #    assert row_names == col_names
        if len(names) != 0 and len(row_names) == 0:
            row_names = names
        if len(names) != 0 and len(col_names) == 0:
            col_names = names
        super(Cov, self).__init__(x=x, isdiagonal=isdiagonal,
                                  row_names=row_names,
                                  col_names=col_names,
                                  autoalign=autoalign)


    @property
    def identity(self):
        """get an identity Matrix like self

        Returns
        -------
        Cov : Cov

        """
        if self.__identity is None:
            self.__identity = Cov(x=np.atleast_2d(np.ones(self.shape[0]))
                                  .transpose(), names=self.row_names,
                                  isdiagonal=True)
        return self.__identity


    @property
    def zero(self):
        """ get an instance of self with all zeros

        Returns
        -------
        Cov : Cov

        """
        if self.__zero is None:
            self.__zero = Cov(x=np.atleast_2d(np.zeros(self.shape[0]))
                              .transpose(), names=self.row_names,
                              isdiagonal=True)
        return self.__zero


    def condition_on(self,conditioning_elements):
        """get a new Covariance object that is conditional on knowing some
            elements.  uses Schur's complement for conditional Covariance
            propagation

        Parameters
        ----------
        conditioning_elements : iterable
            names of elements to condition on

        Returns
        -------
        Cov : Cov
        """
        if not isinstance(conditioning_elements,list):
            conditioning_elements = [conditioning_elements]
        for iname, name in enumerate(conditioning_elements):
            conditioning_elements[iname] = name.lower()
            assert name.lower() in self.col_names,\
                "Cov.condition_on() name not found: " + name
        keep_names = []
        for name in self.col_names:
            if name not in conditioning_elements:
                keep_names.append(name)
        #C11
        new_Cov = self.get(keep_names)
        if self.isdiagonal:
            return new_Cov
        #C22^1
        cond_Cov = self.get(conditioning_elements).inv
        #C12
        upper_off_diag = self.get(keep_names, conditioning_elements)
        #print(new_Cov.shape,upper_off_diag.shape,cond_Cov.shape)
        return new_Cov - (upper_off_diag * cond_Cov * upper_off_diag.T)

    def draw(self, mean=1.0):
        """Obtain a random draw from a covariance matrix either with mean==1
        or with specified mean vector

        Parameters
        ----------
        mean: scalar of enumerable of length self.shape[0]
            mean values. either a scalar applied to to the entire
            vector of length N or an N-length vector

        Returns
        -------
        numpy.nparray : numpy.ndarray
            A vector of conditioned values, sampled
            using the covariance matrix (self) and applied to the mean

        """
        if np.isscalar(mean):
            mean = np.ones(self.ncol) * mean
        else:
            assert len(mean) == self.ncol, "mean vector must be {0} elements. {1} were provided".\
                format(self.ncol, len(mean))

        return(np.random.multivariate_normal(mean, self.as_2d))



    @property
    def names(self):
        """wrapper for getting row_names.  row_names == col_names for Cov

        Returns
        -------
        list : list
            names

        """
        return self.row_names


    def replace(self,other):
        """replace elements in the covariance matrix with elements from other.
        if other is not diagonal, then self becomes non diagonal

        Parameters
        -----------
        other : Cov
            the Cov to replace elements in self with

        Note
        ----
            operates in place

        """
        assert isinstance(other,Cov),"Cov.replace() other must be Cov, not {0}".\
            format(type(other))
        # make sure the names of other are in self
        missing = [n for n in other.names if n not in self.names]
        if len(missing) > 0:
            raise Exception("Cov.replace(): the following other names are not" +\
                            " in self names: {0}".format(','.join(missing)))
        self_idxs = self.indices(other.names,0)
        other_idxs = other.indices(other.names,0)

        if self.isdiagonal and other.isdiagonal:
            self._Matrix__x[self_idxs] = other.x[other_idxs]
            return
        if self.isdiagonal:
            self._Matrix__x = self.as_2d
            self.isdiagonal = False

        #print("allocating other_x")
        other_x = other.as_2d
        #print("replacing")
        for i,ii in zip(self_idxs,other_idxs):
            self._Matrix__x[i,self_idxs] = other_x[ii,other_idxs].copy()
        #print("resetting")
        #self.reset_x(self_x)
        #self.isdiagonal = False

    def to_uncfile(self, unc_file, covmat_file="Cov.mat", var_mult=1.0):
        """write a PEST-compatible uncertainty file

        Parameters
        ----------
        unc_file : str
            filename of the uncertainty file
        covmat_file : str Covariance Matrix filename. Default is
        "Cov.mat".  If None, and Cov.isdiaonal, then a standard deviation
        form of the uncertainty file is written.  Exception raised if None
        and not Cov.isdiagonal
        var_mult : float
            variance multiplier for the covmat_file entry

        """
        assert len(self.row_names) == self.shape[0], \
            "Cov.to_uncfile(): len(row_names) != x.shape[0] "
        if covmat_file:
            f = open(unc_file, 'w')
            f.write("START COVARIANCE_MATRIX\n")
            f.write(" file " + covmat_file + "\n")
            f.write(" variance_multiplier {0:15.6E}\n".format(var_mult))
            f.write("END COVARIANCE_MATRIX\n")
            f.close()
            self.to_ascii(covmat_file, icode=1)
        else:
            if self.isdiagonal:
                f = open(unc_file, 'w')
                f.write("START STANDARD_DEVIATION\n")
                for iname, name in enumerate(self.row_names):
                    f.write("  {0:20s}  {1:15.6E}\n".
                            format(name, np.sqrt(self.x[iname, 0])))
                f.write("END STANDARD_DEVIATION\n")
                f.close()
            else:
                raise Exception("Cov.to_uncfile(): can't write non-diagonal " +
                                "object as standard deviation block")

    @classmethod
    def from_obsweights(cls, pst_file):
        """instantiates a  Cov instance from observation weights in
        a PEST control file.  Calls Cov.from_observation_data()

        Parameters
        ----------
        pst_file : str
            pest control file name

        Returns
        -------
        Cov : Cov
        """
        if not pst_file.endswith(".pst"):
            pst_file += ".pst"
        return Cov.from_observation_data(Pst(pst_file))

    @classmethod
    def from_observation_data(cls, pst):
        """instantiates a  Cov from a pandas dataframe
                of pyemu.Pst.observation_data

        Parameters
        ----------
        pst : pyemu.Pst

        Returns
        -------
        Cov : Cov

        """
        nobs = pst.observation_data.shape[0]
        x = np.zeros((nobs, 1))
        onames = []
        ocount = 0
        for idx,row in pst.observation_data.iterrows():
            w = float(row["weight"])
            w = max(w, 1.0e-30)
            x[ocount] = (1.0 / w) ** 2
            ocount += 1
            onames.append(row["obsnme"].lower())
        return cls(x=x,names=onames,isdiagonal=True)

    @classmethod
    def from_parbounds(cls, pst_file, sigma_range = 4.0,scale_offset=True):
        """Instantiates a  Cov from a pest control file parameter data section.
        Calls Cov.from_parameter_data()

        Parameters
        ----------
        pst_file : str
            pest control file name
        sigma_range: float
            defines range of upper bound - lower bound in terms of standard
            deviation (sigma). For example, if sigma_range = 4, the bounds
            represent 4 * sigma.  Default is 4.0, representing approximately
            95% confidence of implied normal distribution
        scale_offset : bool
            flag to apply scale and offset to parameter upper and lower
            bounds before calculating varaince.  Default is True

        Returns
        -------
        Cov : Cov

        """
        if not pst_file.endswith(".pst"):
            pst_file += ".pst"
        new_pst = Pst(pst_file)
        return Cov.from_parameter_data(new_pst, sigma_range)

    @classmethod
    def from_parameter_data(cls, pst, sigma_range = 4.0, scale_offset=True):
        """load Covariances from a pandas dataframe of
        pyemu.Pst.parameter_data

        Parameters
        ----------
        pst : (pyemu.Pst)
        sigma_range: float
            defines range of upper bound - lower bound in terms of standard
            deviation (sigma). For example, if sigma_range = 4, the bounds
            represent 4 * sigma.  Default is 4.0, representing approximately
            95% confidence of implied normal distribution
        scale_offset : bool
            flag to apply scale and offset to parameter upper and lower
            bounds before calculating varaince.  Default is True

        Returns
        -------
        Cov : Cov

        """
        npar = pst.npar_adj
        x = np.zeros((npar, 1))
        names = []
        idx = 0
        for i, row in pst.parameter_data.iterrows():
            t = row["partrans"]
            if t in ["fixed", "tied"]:
                continue
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
                raise Exception("Cov.from_parameter_data() error: " +\
                                "variance for parameter {0} is nan".\
                                format(row["parnme"]))
            if (var == 0.0):
                raise Exception("Cov.from_parameter_data() error: " +\
                                "variance for parameter {0} is 0.0".\
                                format(row["parnme"]))
            x[idx] = var
            names.append(row["parnme"].lower())
            idx += 1

        return cls(x=x,names=names,isdiagonal=True)

    @classmethod
    def from_uncfile(cls, filename):
        """instaniates a Cov from a PEST-compatible uncertainty file

        Parameters
        ----------
        filename : str
            uncertainty file name

        Returns
        -------
        Cov : Cov

        """

        nentries = Cov.get_uncfile_dimensions(filename)
        x = np.zeros((nentries, nentries))
        row_names = []
        col_names = []
        f = open(filename, 'r')
        isdiagonal = True
        idx = 0
        while True:
            line = f.readline().lower()
            if len(line) == 0:
                break
            line = line.strip()
            if 'start' in line:
                if 'standard_deviation' in line:
                    std_mult = 1.0
                    while True:
                        line2 = f.readline().strip().lower()
                        if line2.strip().lower().startswith("end"):
                            break


                        raw = line2.strip().split()
                        name,val = raw[0], float(raw[1])
                        if name == "std_multiplier":
                            std_mult = val
                        else:
                            x[idx, idx] = (val*std_mult)**2
                            if name in row_names:
                                raise Exception("Cov.from_uncfile():" +
                                                "duplicate name: " + str(name))
                            row_names.append(name)
                            col_names.append(name)
                            idx += 1

                elif 'covariance_matrix' in line:
                    isdiagonal = False
                    var = 1.0
                    while True:
                        line2 = f.readline().strip().lower()
                        if line2.strip().lower().startswith("end"):
                            break
                        if line2.startswith('file'):
                            cov = Matrix.from_ascii(line2.split()[1])

                        elif line2.startswith('variance_multiplier'):
                            var = float(line2.split()[1])
                        else:
                            raise Exception("Cov.from_uncfile(): " +
                                            "unrecognized keyword in" +
                                            "std block: " + line2)
                    if var != 1.0:
                        cov *= var
                    for name in cov.row_names:
                        if name in row_names:
                            raise Exception("Cov.from_uncfile():" +
                                            " duplicate name: " + str(name))
                    row_names.extend(cov.row_names)
                    col_names.extend(cov.col_names)

                    for i, rname in enumerate(cov.row_names):
                        x[idx + i,idx:idx + cov.shape[0]] = cov.x[i, :].copy()
                    idx += cov.shape[0]
                else:
                    raise Exception('Cov.from_uncfile(): ' +
                                    'unrecognized block:' + str(line))
        f.close()
        if isdiagonal:
            x = np.atleast_2d(np.diag(x)).transpose()
        return cls(x=x,names=row_names,isdiagonal=isdiagonal)

    @staticmethod
    def get_uncfile_dimensions(filename):
        """quickly read an uncertainty file to find the dimensions

        Parameters
        ----------
        filename : str
            uncertainty filename

        Returns
        -------
        nentries : int
            number of elements in file
        """
        f = open(filename, 'r')
        nentries = 0
        while True:
            line = f.readline().lower()
            if len(line) == 0:
                break
            line = line.strip()
            if 'start' in line:
                if 'standard_deviation' in line:
                    while True:
                        line2 = f.readline().strip().lower()
                        if "std_multiplier" in line2:
                            continue
                        if line2.strip().lower().startswith("end"):
                            break
                        nentries += 1

                elif 'covariance_matrix' in line:
                    while True:
                        line2 = f.readline().strip().lower()
                        if line2.strip().lower().startswith("end"):
                            break
                        if line2.startswith('file'):
                            cov = Matrix.from_ascii(line2.split()[1])
                            nentries += len(cov.row_names)
                        elif line2.startswith('variance_multiplier'):
                            var = float(line2.split()[1])
                        else:
                            raise Exception('Cov.get_uncfile_dimensions(): ' +
                            'unrecognized keyword in Covariance block: ' +
                                            line2)
                else:
                    raise Exception('Cov.get_uncfile_dimensions():' +
                                    'unrecognized block:' + str(line))
        f.close()
        return nentries

    @classmethod
    def identity_like(cls,other):
        """ Get an identity matrix Cov instance like other

        Parameters
        ----------
        other : Matrix
            must be square

        Returns
        -------
        Cov : Cov

        """
        assert other.shape[0] == other.shape[1]
        x = np.identity(other.shape[0])
        return cls(x=x,names=other.row_names,isdiagonal=False)

    def to_pearson(self):
        """ Convert Cov instance to Pearson correlation coefficient
        matrix

        Returns
        -------
        Matrix : Matrix
            this is on purpose so that it is clear the returned
            instance is not a Cov

        """
        std_dict = self.get_diagonal_vector().to_dataframe()["diag"].\
            apply(np.sqrt).to_dict()
        pearson = self.identity.as_2d
        if self.isdiagonal:
            return Matrix(x=pearson,row_names=self.row_names,
                          col_names=self.col_names)
        df = self.to_dataframe()
        # fill the lower triangle
        for i,iname in enumerate(self.row_names):
            for j,jname in enumerate(self.row_names[i+1:]):
                # cv = df.loc[iname,jname]
                # std1,std2 = std_dict[iname],std_dict[jname]
                # cc = cv / (std1*std2)
                # v1 = np.sqrt(df.loc[iname,iname])
                # v2 = np.sqrt(df.loc[jname,jname])
                pearson[i,j+i+1] = df.loc[iname,jname] / (std_dict[iname] * std_dict[jname])

        # replicate across diagonal
        for i,iname in enumerate(self.row_names[:-1]):
             pearson[i+1:,i] = pearson[i,i+1:]
        return Matrix(x=pearson,row_names=self.row_names,
                      col_names=self.col_names)



class SparseMatrix(object):
    """Note: less rigid about references since this class is for big matrices and
    don't want to be making copies"""
    def __init__(self,x,row_names,col_names):
        assert isinstance(x,scipy.sparse.coo_matrix)
        assert x.shape[0] == len(row_names)
        assert x.shape[1] == len(col_names)
        self.x = x
        self.row_names = list(row_names)
        self.col_names = list(col_names)


    @property
    def shape(self):
        return self.x.shape

    @classmethod
    def from_binary(cls,filename):
        x,row_names,col_names = Matrix.read_binary(filename,sparse=True)
        return cls(x=x,row_names=row_names,col_names=col_names)



    def to_coo(self,filename):
        save_coo(self.x,row_names=self.row_names,col_names=self.col_names,filename=filename)



    @classmethod
    def from_coo(cls,filename):
        return SparseMatrix.from_binary(filename)

    # @classmethod
    # def from_csv(cls,filename):
    #     pass
    #
    #
    # def to_csv(self,filename):
    #     pass


    def to_matrix(self):
        x = np.zeros(self.shape)
        for i,j,d in zip(self.x.row,self.x.col,self.x.data):
            x[i,j] = d
        return Matrix(x=x,row_names=self.row_names,col_names=self.col_names)


    @classmethod
    def from_matrix(cls, matrix, droptol=None):
        iidx,jidx = matrix.as_2d.nonzero()
        coo = scipy.sparse.coo_matrix((matrix.as_2d[iidx,jidx],(iidx,jidx)),shape=matrix.shape)
        return cls(x=coo,row_names=matrix.row_names,col_names=matrix.col_names)


    def block_extend_ip(self,other):
        """designed for combining dense martrices into a sparse block diagonal matrix.
        other must not have any rows or columns in common with self """
        ss = set(self.row_names)
        os = set(other.row_names)
        inter = ss.intersection(os)
        if len(inter) > 0:
            raise Exception("SparseMatrix.block_extend_ip(): error shares the following rows:{0}".
                            format(','.join(inter)))
        ss = set(self.col_names)
        os = set(other.col_names)
        inter = ss.intersection(os)
        if len(inter) > 0:
            raise Exception("SparseMatrix.block_extend_ip(): error shares the following cols:{0}".
                            format(','.join(inter)))

        if isinstance(other,Matrix):
            iidx,jidx = other.as_2d.nonzero()
            # this looks terrible but trying to do this as close to "in place" as possible
            self.x = scipy.sparse.coo_matrix((np.append(self.x.data,other.as_2d[iidx,jidx]),
                                              (np.append(self.x.row,(iidx+self.shape[0])),
                                               np.append(self.x.col,(jidx+self.shape[1])))),
                                             shape=(self.shape[0]+other.shape[0],self.shape[1]+other.shape[1]))
            self.row_names.extend(other.row_names)
            self.col_names.extend(other.col_names)

        elif isinstance(other,SparseMatrix):
            self.x = scipy.sparse.coo_matrix((np.append(self.x.data, other.x.data),
                                              (np.append(self.x.row, (self.shape[0] + other.x.row)),
                                               np.append(self.x.col, (self.shape[1] + other.x.col)))),
                                             shape=(self.shape[0] + other.shape[0], self.shape[1] + other.shape[1]))
            self.row_names.extend(other.row_names)
            self.col_names.extend(other.col_names)


        else:
            raise NotImplementedError("SparseMatrix.block_extend_ip() 'other' arg only supports Matrix types ")


    def get_matrix(self,row_names,col_names):
        if not isinstance(row_names,list):
            row_names = [row_names]
        if not isinstance(col_names,list):
            col_names = [col_names]

        iidx = Matrix.find_rowcol_indices(row_names,self.row_names,self.col_names,axis=0)
        jidx = Matrix.find_rowcol_indices(col_names,self.row_names,self.col_names,axis=1)

        imap = {ii:i for i,ii in enumerate(iidx)}
        jmap = {jj:j for j,jj in enumerate(jidx)}

        iset = set(iidx)
        jset = set(jidx)

        x = np.zeros((len(row_names),len(col_names)))
        # for i,idx in enumerate(iidx):
        #     for j,jdx in enumerate(jidx):
        #         if jdx in jset and idx in iset:
        #             x[i,j] = self.x[idx,jdx]

        for i,j,d in zip(self.x.row,self.x.col,self.x.data):
            if i in iset and j in jset:
                x[imap[i],jmap[j]] = d
        return Matrix(x=x,row_names=row_names,col_names=col_names)

    def get_sparse_matrix(self,row_names,col_names):
        if not isinstance(row_names,list):
            row_names = [row_names]
        if not isinstance(col_names,list):
            col_names = [col_names]

        iidx = Matrix.find_rowcol_indices(row_names,self.row_names,self.col_names,axis=0)
        jidx = Matrix.find_rowcol_indices(col_names,self.row_names,self.col_names,axis=1)

        imap = {ii:i for i,ii in enumerate(iidx)}
        jmap = {jj:j for j,jj in enumerate(jidx)}

        iset = set(iidx)
        jset = set(jidx)

        x = np.zeros((len(row_names),len(col_names)))
        # for i,idx in enumerate(iidx):
        #     for j,jdx in enumerate(jidx):
        #         if jdx in jset and idx in iset:
        #             x[i,j] = self.x[idx,jdx]
        ii,jj,data = [],[],[]
        for i,j,d in zip(self.x.row,self.x.col,self.x.data):
            if i in iset and j in jset:
                ii.append(i)
                jj.append(j)
                data.append(d)
        coo = scipy.sparse.coo_matrix((data,(ii,jj)),shape=(len(row_names),len(col_names)))
        return SparseMatrix(x=coo,row_names=row_names,col_names=col_names)




