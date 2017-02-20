from __future__ import print_function, division
import copy
import struct
from datetime import datetime
import numpy as np
import pandas
import scipy.linalg as la
from scipy.io import FortranFile

from pyemu.pst.pst_handler import Pst

def concat(mats):
    """Concatenate Matrix objects.  Tries either axis.
    Parameters:
    ----------
        mats: an enumerable of Matrix objects
    Returns:
    -------
        Matrix
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
        x = mats[0].newx
        for mat in mats[1:]:
            mat.align(mats[0].row_names, axis=0)
            other_x = mat.newx
            x = np.append(x, other_x, axis=1)

    else:
        col_names = copy.deepcopy(mats[0].col_names)
        row_names = []
        for mat in mats:
            row_names.extend(copy.deepcopy(mat.row_names))
        x = mat[0].newx
        for mat in mats[1:]:
            mat.align(mats[0].col_names, axis=1)
            other_x = mat.newx
            x = np.append(x, other_x, axis=0)
    return Matrix(x=x, row_names=row_names, col_names=col_names)


def get_common_elements(list1, list2):
    """find the common elements in two lists.  used to support auto align
        might be faster with sets
    Parameters:
    ----------
        list1 : a list of objects
        list2 : a list of objects
    Returns:
    -------
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
    Attributes:
        x : ndarray
            numpy ndarray
        row_names : list(str)
            names of the rows in the matrix
        col_names : list(str)
            names of the columns in the matrix
        shape : tuple
            shape of the matrix
        isdiagonal : bool
            diagonal matrix flag
    Notes:
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

    par_length = 12
    obs_length = 20

    def __init__(self, x=None, row_names=[], col_names=[], isdiagonal=False,
                 autoalign=True):
        """constructor for Matrix objects
        Args:
            x : numpy array for the Matrix entries
            row_names : list of Matrix row names
            col_names : list of Matrix column names
            isdigonal : bool to determine if the Matrix is diagonal
            autoalign: bool used to control the autoalignment of Matrix objects
                during linear algebra operations
        Returns:
            None
        """
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

    def __str__(self):
        s = "shape:{0}:{1}".format(*self.shape)+" row names: " + str(self.row_names) + \
            '\n' + "col names: " + str(self.col_names) + '\n' + str(self.__x)
        return s

    def __getitem__(self, item):
        """a very crude overload of getitem - not trying to parse item,
            instead relying on shape of submat
        Parameters:
        ----------
            item : an enumerable that can be used as an index
        Returns:
        -------
            a Matrix object that is a subMatrix of self
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
        """overload of __pow__ operator
        Parameters:
        ----------
            power: int or float.  interpreted as follows:
                -1 = inverse of self
                -0.5 = sqrt of inverse of self
                0.5 = sqrt of self
                all other positive ints = elementwise self raised to power
        Returns:
        -------
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
        """
            subtraction overload.  tries to speedup by checking for scalars of
            diagonal matrices on either side of operator
        Parameters:
        ----------
            other : [scalar,numpy.ndarray,Matrix object]
        Returns:
        -------
            Matrix object
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
        """addition overload.  tries to speedup by checking for
            scalars of diagonal matrices on either side of operator
        Parameters:
        ----------
            other : [scalar,numpy.ndarray,Matrix object]
        Returns:
        -------
            Matrix
        """
        if np.isscalar(other):
            return type(self)(x=self.x + other)
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
        """element-wise multiplication.  tries to speedup by checking for
            scalars of diagonal matrices on either side of operator
        Parameters:
        ----------
            other : [scalar,numpy.ndarray,Matrix object]
        Returns:
        -------
            Matrix
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
        """multiplication overload.  tries to speedup by checking for scalars or
            diagonal matrices on either side of operator
        Parameters:
        ----------
            other : [scalar,numpy.ndarray,Matrix object]
        Returns:
        -------
            Matrix object
        """
        if np.isscalar(other):
            return type(self)(x=self.__x.copy() * other,
                              row_names=self.row_names,
                              col_names=self.col_names)
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
        if np.isscalar(other):
            return type(self)(x=self.__x.copy() * other,row_names=self.row_names,\
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
        """private method to set SVD components
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
        """check if matrices are aligned for multiplication
        Parameters:
        ----------
            other : Matrix
        Returns:
        -------
            True if aligned
            False if not aligned
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
        Parameters:
        ----------
            other : Matrix
        Returns:
        -------
            True if aligned
            False if not aligned
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
        """
        return self.__x.copy()


    @property
    def x(self):
        """return a reference to x
        """
        return self.__x

    @property
    def as_2d(self):
        if not self.isdiagonal:
            return self.x
        return np.diag(self.x.flatten())

    @property
    def shape(self):
        """get the shape of x
        Parameters:
        ----------
            None
        Returns:
        -------
            tuple of ndims
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
        return self.shape[1]

    @property
    def nrow(self):
        return self.shape[0]

    @property
    def T(self):
        """wrapper function for transpose
        """
        return self.transpose


    @property
    def transpose(self):
        """transpose operation
        Parameters:
        ----------
            None
        Returns:
        -------
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
        """inversion operation
        Parameters:
        ----------
            None
        Returns:
        -------
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
        sthresh =np.abs((self.s.x / self.s.x[0]) - eigthresh)
        return max(1,np.argmin(sthresh))

    def pseudo_inv_components(self,maxsing=None,eigthresh=1.0e-5):
        if maxsing is None:
            maxsing = self.get_maxsing(eigthresh=eigthresh)

        s = self.s[:maxsing,:maxsing]
        v = self.v[:,:maxsing]
        u = self.u[:,:maxsing]
        return u,s,v

    def pseudo_inv(self,maxsing=None,eigthresh=1.0e-5):
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
        Parameters:
        ----------
            None
        Returns:
        -------
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
        x = np.zeros((self.shape),dtype=np.float32)

        x[:self.s.shape[0],:self.s.shape[0]] = self.s.as_2d
        s = Matrix(x=x, row_names=self.row_names,
                          col_names=self.col_names, isdiagonal=False,
                          autoalign=False)
        return s




    @property
    def s(self):
        """the singular value (diagonal) Matrix
        """
        if self.__s is None:
            self.__set_svd()
        return self.__s


    @property
    def u(self):
        """the left singular vector Matrix
        """
        if self.__u is None:
            self.__set_svd()
        return self.__u


    @property
    def v(self):
        """the right singular vector Matrix
        """
        if self.__v is None:
            self.__set_svd()
        return self.__v

    @property
    def zero2d(self):
        """ get an instance of self with all zeros
        """
        return type(self)(x=np.atleast_2d(np.zeros((self.shape[0],self.shape[1]))),
                   row_names=self.row_names,
                   col_names=self.col_names,
                   isdiagonal=False)

    def indices(self, names, axis=None):
        """get the row and col indices of names
        Parameters:
        ----------
            names : [enumerable] column and/or row names
            axis : [int] the axis to search.
        Returns:
        -------
            numpy.ndarray : indices of names.  if axis is None, two ndarrays
                are returned, corresponding the indices of names for each axis
        """
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
        """reorder self by names
        Parameters:
        ----------
            names : [enumerable] names in row and\or column names
            axis : [int] the axis to reorder. if None, reorder both axes
        Returns:
        -------
            None
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
        """get a (sub)Matrix ordered on row_names or col_names
        Parameters:
        ----------
            row_names : [enumerable] row_names for new Matrix
            col_names : [enumerable] col_names for new Matrix
            drop : [bool] flag to remove row_names and/or col_names
        Returns:
        -------
            Matrix
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
                extract = extract[:, idxs.copy()]
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


    def drop(self, names, axis):
        """ drop elements from self
        Parameters:
        ----------
            names : [enumerable] names to drop
            axis : [int] the axis to drop from. must be in [0,1]
        Returns:
        -------
            None
        """
        if axis is None:
            raise Exception("Matrix.drop(): axis arg is required")
        if not isinstance(names, list):
            names = [names]
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
        """wrapper method that gets then drops elements
        """
        if row_names is None and col_names is None:
            raise Exception("Matrix.extract() " +
                            "row_names and col_names both None")
        extract = self.get(row_names, col_names, drop=True)
        return extract

    def get_diagonal_vector(self, col_name="diag"):
        assert self.shape[0] == self.shape[1]
        assert not self.isdiagonal
        assert isinstance(col_name,str)
        return type(self)(x=np.atleast_2d(np.diag(self.x)).transpose(),
                          row_names=self.row_names,
                          col_names=[col_name],isdiagonal=False)

    def to_binary(self, filename):
        """write a pest-compatible binary file
        Parameters:
        ----------
            filename : [str] filename to save binary file
        Returns:
        -------
            None
        """
        if self.isdiagonal:
            raise NotImplementedError()
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
        flat = self.x[row_idxs, col_idxs].flatten()
        # zip up the index position and value pairs
        #data = np.array(list(zip(icount, flat)), dtype=self.binary_rec_dt)
        data = np.core.records.fromarrays([icount,flat],dtype=self.binary_rec_dt)
        # write
        data.tofile(f)

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
    def from_binary(cls, filename):
        """load from pest-compatible binary file
        Parameters:
        ----------
            filename : [str] filename to save binary file
        Returns:
        -------
            None
        """

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

        if itemp1 >= 0:
           raise TypeError('Matrix.from_binary(): Jco produced by ' +
                           'deprecated version of PEST,' +
                           'Use JcoTRANS to convert to new format')
        #icount = np.fromfile(f,np.int32,1)
        #print(itemp1,itemp2,icount)
        ncol, nrow = abs(itemp1), abs(itemp2)
        x = np.zeros((nrow, ncol))
        # read all data records
        # using this a memory hog, but really fast
        data = np.fromfile(f, Matrix.binary_rec_dt, icount)
        icols = ((data['j'] - 1) // nrow) + 1
        irows = data['j'] - ((icols - 1) * nrow)
        x[irows - 1, icols - 1] = data["dtemp"]
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
        assert len(row_names) == x.shape[0],\
          "Matrix.from_binary() len(row_names) (" + str(len(row_names)) +\
          ") != x.shape[0] (" + str(x.shape[0]) + ")"
        assert len(col_names) == x.shape[1],\
          "Matrix.from_binary() len(col_names) (" + str(len(col_names)) +\
          ") != self.shape[1] (" + str(x.shape[1]) + ")"
        return cls(x=x,row_names=row_names,col_names=col_names)

    @classmethod
    def from_fortranfile(cls, filename):
        """ a binary load method to accommodate one of the many
            bizzare fortran binary writing formats
        Parameters:
        ----------
            filename : str
                name of the binary matrix file
        Returns:
        -------
            None
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
        """write a pest-compatible ASCII Matrix/vector file
        Parameters:
        ----------
            out_filename : [str] output filename
            icode : [int] pest-style info code for Matrix style
        Returns:
        -------
            None
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
    def from_ascii(cls, filename):
        """load a pest-compatible ASCII Matrix/vector file
        Parameters:
        ----------
            filename : str
                name of the file to read
        Returns:
        -------
            None
        """
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
        return cls(x=x,row_names=row_names,col_names=col_names,
                   isdiagonal=isdiagonal)

    def df(self):
        return self.to_dataframe()

    @classmethod
    def from_dataframe(cls, df):
        """ populate self with dataframe information
        Parameters:
        ----------
            df : pandas dataframe

        Returns:
        -------
            None

        """
        assert isinstance(df, pandas.DataFrame)
        row_names = copy.deepcopy(list(df.index))
        col_names = copy.deepcopy(list(df.columns))
        return cls(x=df.as_matrix(),row_names=row_names,col_names=col_names)

    def to_dataframe(self):
        """return a pandas dataframe of the Matrix object
        Parameters:
        ----------
            None
        Returns:
        -------
            pandas dataframe
        """
        if self.isdiagonal:
            x = np.diag(self.__x[:, 0])
        else:
            x = self.__x
        return pandas.DataFrame(data=x,index=self.row_names,columns=self.col_names)


    def to_sparse(self, trunc=0.0):
        """get the CSR sparse Matrix representation of Matrix
        Parameters:
        ----------
            None
        Returns:
        -------
            scipy sparse Matrix object
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
        return sparse.csr_matrix((data, (iidx, jidx)), shape=(self.shape))


    def extend(self,other,inplace=False):
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
    """a thin wrapper class to get more intuitive attribute names
    """
    def __init(self, **kwargs):
        super(Jco, self).__init__(kwargs)


    @property
    def par_names(self):
        return self.col_names


    @property
    def obs_names(self):
        return self.row_names


    @property
    def npar(self):
        return self.shape[1]


    @property
    def nobs(self):
        return self.shape[0]



class Cov(Matrix):
    """a subclass of Matrix for handling diagonal or dense Covariance matrices
        todo:block diagonal
    """
    def __init__(self, x=None, names=[], row_names=[], col_names=[],
                 isdiagonal=False, autoalign=True):
        """
        Parameters:
            x : numpy.ndarray
            names : [enumerable] names for both columns and rows
            row_names : [enumerable] names for rows
            col_names : [enumerable] names for columns
            isdiagonal : [bool] diagonal Matrix flag
            autoalign : [bool] autoalignment flag
        Returns:
            None
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
    #     self.__eigvals = None
    #     self.__eigvecs = None
    #
    # @property
    # def eigvals(self):
    #     if self.__eigvals is None:
    #         self.__eig()
    #     return self.__eigvals
    #
    # @property
    # def eigvecs(self):
    #     if self.__eigvecs is None:
    #         self.__eig()
    #     return self.__eigvecs
    #
    # def __eig(self):
    #     try:
    #         vals,vecs = np.linalg.eigh(self.x)
    #     except Exception as e:
    #         raise Exception("Cov.__eig() error:{0}".format(str(e)))
    #     names = ["eig{0}".format(i) for i in range(vals.shape[0])]
    #     self.__eigvals = Matrix(x=np.atleast_2d(vals),row_names=names,col_names=names,
    #                             isdiagonal=True)


    @property
    def identity(self):
        """get an identity Matrix like self
        """
        if self.__identity is None:
            self.__identity = Cov(x=np.atleast_2d(np.ones(self.shape[0]))
                                  .transpose(), names=self.row_names,
                                  isdiagonal=True)
        return self.__identity


    @property
    def zero(self):
        """ get an instance of self with all zeros
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
        Parameters:
        ----------
            conditioning_elements : [enumerable] names of elements to
                                    condition on
        Returns:
        -------
            Cov object
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


    def to_uncfile(self, unc_file, covmat_file="Cov.mat", var_mult=1.0):
        """write a pest-compatible uncertainty file
        Parameters:
        ----------
            unc_file : [str] filename
            Covmat : [str] Covariance Matrix filename
            var_mult : [float] variance multiplier
        Returns:
        -------
            None
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
        """load Covariance from observation weights
        Parameters:
        ----------
            pst_file : [str] pest control file name
        Returns:
        -------
            None
        """
        if not pst_file.endswith(".pst"):
            pst_file += ".pst"
        return Cov.from_observation_data(Pst(pst_file))

    @classmethod
    def from_observation_data(cls, pst):
        """load Covariances from a pandas dataframe
                of the pst observation data section
        Parameters:
        ----------
            pst : [pst object]
        Returns:
        -------
            None
        """
        nobs = pst.observation_data.shape[0]
        # if pst.mode == "estimation":
        #     nobs += pst.nprior
        x = np.zeros((nobs, 1))
        onames = []
        ocount = 0
        for idx,row in pst.observation_data.iterrows():
            w = float(row["weight"])
            w = max(w, 1.0e-30)
            x[ocount] = (1.0 / w) ** 2
            ocount += 1
            onames.append(row["obsnme"].lower())
        # leave the prior info out of the obsCov
        # if pst.mode == "estimation" and pst.nprior > 0:
        #     for iidx, row in pst.prior_information.iterrows():
        #         w = float(row["weight"])
        #         w = max(w, 1.0e-30)
        #         x[ocount] = (1.0 / w) ** 2
        #         ocount += 1
        #         onames.append(row["pilbl"].lower())

        return cls(x=x,names=onames,isdiagonal=True)

    @classmethod
    def from_parbounds(cls, pst_file):
        """load Covariances from a pest control file parameter data section
        Parameters:
        ----------
            pst_file : [str] pest control file name
        Returns:
        -------
            None
        """
        if not pst_file.endswith(".pst"):
            pst_file += ".pst"
        new_pst = Pst(pst_file)
        return Cov.from_parameter_data(new_pst)

    @classmethod
    def from_parameter_data(cls, pst):
        """load Covariances from a pandas dataframe of the
                pst parameter data section
        Parameters:
        ----------
            pst : [pst object]
        Returns:
        -------
            None
        """
        npar = pst.npar_adj
        x = np.zeros((npar, 1))
        names = []
        idx = 0
        for i, row in pst.parameter_data.iterrows():
            t = row["partrans"]
            if t in ["fixed", "tied"]:
                continue
            lb = row.parlbnd * row.scale + row.offset
            ub = row.parubnd * row.scale + row.offset

            if t == "log":
                var = ((np.log10(np.abs(ub)) - np.log10(np.abs(lb))) / 4.0) ** 2
            else:
                var = ((ub - lb) / 4.0) ** 2
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
        """load Covariances from a pest-compatible uncertainty file
        Parameters:
        ----------
            filename : [str] uncertainty file name
        Returns:
        -------
            None
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
                    while True:
                        line2 = f.readline().strip().lower()
                        if line2.strip().lower().startswith("end"):
                            break
                        raw = line2.strip().split()
                        name,val = raw[0], float(raw[1])
                        x[idx, idx] = val**2
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
                        x *= var
                    for name in cov.row_names:
                        if name in row_names:
                            raise Exception("Cov.from_uncfile():" +
                                            " duplicate name: " + str(name))
                    row_names.extend(cov.row_names)
                    col_names.extend(cov.col_names)

                    for i, rname in enumerate(cov.row_names):
                        x[idx + i,idx:idx + cov.shape[0]] = cov.x[i, :]
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
        Parameters:
        ----------
            filename : [str] uncertainty filename
        Returns:
        -------
            nentries : [int] number of elements in file
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
        assert other.shape[0] == other.shape[1]
        x = np.identity(other.shape[0])
        return cls(x=x,names=other.row_names,isdiagonal=True)

    def to_pearson(self):

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
