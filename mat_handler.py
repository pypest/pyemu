import copy
import struct
import numpy as np
import pandas
import scipy.linalg as la
import pst_handler as phand

def concat(mats):
    """Concatenate matrix objects.  Tries either axis.
    Args:
        mats: an enumerable of matrix objects
    Returns:
        matrix
    Raises
        NotImplementedError is a diagonal matrix object is in mats
        Exception if all objects in mats are not aligned by
            eithers rows or columns
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
        raise Exception("mat_handler.concat(): all matrix objects"+\
                        "must share either rows or cols")

    if row_match and col_match:
        raise Exception("mat_handler.concat(): all matrix objects"+\
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
            x = np.append(x,other_x, axis=1)

    else:
        col_names = copy.deepcopy(mats[0].col_names)
        row_names = []
        for mat in mats:
            row_names.extend(copy.deepcopy(mat.row_names))
        x = mat[0].newx
        for mat in mats[1:]:
            mat.align(mats[0].col_names, axis=1)
            other_x = mat.newx
            x = np.append(x,other_x, axis=0)
    return matrix(x=x, row_names=row_names, col_names=col_names)


def get_common_elements(list1, list2):
    """find the common elements in two lists.  used to support auto align
        might be faster with sets
    Args:
        list1 : a list of objects
        list2 : a list of objects
    Returns:
        list of common objects shared by list1 and list2
    Raises:
        None
    """
    result = []
    for item in list1:
        if item in list2:
            result.append(item)
    return result



class matrix(object):
    """a class for easy linear algebra
    Attributes:
        too many to list...
    Notes:
        this class makes heavy use of property decorators to encapsulate
        private attributes

    """
    def __init__(self, x=None, row_names=[], col_names=[], isdiagonal=False,
                 autoalign=True):
        """constructor for matrix objects
        Args:
            x : numpy array for the matrix entries
            row_names : list of matrix row names
            col_names : list of matrix column names
            isdigonal : bool to determine if the matrix is diagonal
            autoalign: bool used to control the autoalignment of matrix objects
                during linear algebra operations
        Returns:
            None
        Raises:
            AssertionErrors is x.shape  len(row_names) and len(col_names)
                don't agree
        """
        self.col_names, self.row_names = [], []
        [self.col_names.append(c.lower()) for c in col_names]
        [self.row_names.append(r.lower()) for r in row_names]
        self.__x = None
        self.__u = None
        self.__s = None
        self.__v = None
        if x is not None:
            x = np.atleast_2d(x)
            if isdiagonal and len(row_names) > 0:
                assert len(row_names) == x.shape[0],\
                    'matrix.__init__(): diagonal shape[1] != len(row_names) ' +\
                    str(x.shape) + ' ' + str(len(row_names))
            else:
                if len(row_names) > 0:
                    assert len(row_names) == x.shape[0],\
                        'matrix.__init__(): shape[0] != len(row_names) ' +\
                        str(x.shape) + ' ' + str(len(row_names))
                if len(col_names) > 0:
                    #--if this a row vector
                    if len(row_names) == 0 and x.shape[1] == 1:
                        x.transpose()
                    assert len(col_names) == x.shape[1],\
                        'matrix.__init__(): shape[1] != len(col_names) ' + \
                        str(x.shape) + ' ' + str(len(col_names))
            self.__x = x
        self.integer = np.int32
        self.double = np.float64
        self.char = np.uint8
        self.isdiagonal = bool(isdiagonal)
        self.autoalign = bool(autoalign)

        self.binary_header_dt = np.dtype([('itemp1', self.integer),
                                          ('itemp2', self.integer),
                                          ('icount', self.integer)])
        self.binary_rec_dt = np.dtype([('j', self.integer),
                                       ('dtemp', self.double)])
        self.par_length = 12
        self.obs_length = 20

    def __str__(self):
        s = "row names: " + str(self.row_names) + \
            '\n' + "col names: " + str(self.col_names) + '\n' + str(self.__x)
        return s


    def __getitem__(self, item):
        """a very crude overload of getitem - not trying to parse item,
            instead relying on shape of submat
        Args:
            item : an enumerable that can be used as an index
        Returns:
            a matrix object that is a submatrix of self
        Raises:
            None
        """
        if self.isdiagonal and isinstance(item, tuple):
            submat = np.atleast_2d((self.__x[item[0]]))
        else:
            submat = np.atleast_2d(self.__x[item])
        #--transpose a row vector to a column vector
        if submat.shape[0] == 1:
            submat = submat.transpose()
        row_names = self.row_names[:submat.shape[0]]
        if self.isdiagonal:
            col_names = row_names
        else:
            col_names = self.col_names[:submat.shape[1]]
        return matrix(x=submat, isdiagonal=self.isdiagonal, row_names=row_names,
                      col_names=col_names, autoalign=self.autoalign)


    def __pow__(self, power):
        """overload of __pow__ operator
        Args:
            power: int or float.  interpreted as follows:
                -1 = inverse of self
                -0.5 = sqrt of inverse of self
                0.5 = sqrt of self
                all other positive ints = elementwise self raised to power
        Returns:
            a new matrix object
        Raises:
            NotImplementedError for non-supported negative and float powers
        """
        if power < 0:
            if power == -1:
                return self.inv
            elif power == -0.5:
                return (self.inv).sqrt
            else:
                raise NotImplementedError("matrix.__pow__() not implemented " +
                                          "for negative powers except for -1")

        elif int(power) != float(power):
            if power == 0.5:
                return self.sqrt
            else:
                raise NotImplementedError("matrix.__pow__() not implemented " +
                                          "for fractional powers except 0.5")
        else:
            return matrix(self.__x**power, row_names=self.row_names,
                          col_names=self.col_names, isdiagonal=self.isdiagonal)


    def __sub__(self, other):
        """
            subtraction overload.  tries to speedup by checking for scalars of
            diagonal matrices on either side of operator
        Args:
            other : [scalar,numpy.ndarray,matrix object]
        Returns:
            matrix object
        Raises:
            AssertionError if other is not aligned with self
            Exception is other is not in supported types
        """
        if np.isscalar(other):
            return matrix(x=self.x - other, row_names=self.row_names,
                          col_names=self.col_names,
                          isdiagonal=self.isdiagonal)
        else:
            if isinstance(other, np.ndarray):
                assert self.shape == other.shape, "matrix.__sub__() shape" +\
                                                  "mismatch: " +\
                                                  str(self.shape) + ' ' + \
                                                  str(other.shape)
                if self.isdiagonal:
                    elem_sub = -1.0 * other
                    for j in xrange(self.shape[0]):
                        elem_sub[j, j] += self.x[j]
                    return matrix(x=elem_sub, row_names=self.row_names,
                                  col_names=self.col_names)
                else:
                    return matrix(x=self.x - other, row_names=self.row_names,
                                  col_names=self.col_names)
            elif isinstance(other, matrix):
                if self.autoalign and other.autoalign \
                        and not self.element_isaligned(other):
                    common_rows = get_common_elements(self.row_names,
                                                      other.row_names)
                    common_cols = get_common_elements(self.col_names,
                                                      other.col_names)
                    first = self.get(row_names=common_rows,
                                     col_names=common_cols)
                    second = other.get(row_names=common_rows,
                                       col_names=common_cols)
                else:
                    assert self.shape == other.shape, \
                        "matrix.__sub__():shape mismatch: "+\
                        str(self.shape) + ' ' + str(other.shape)
                    first = self
                    second = other

                if first.isdiagonal and second.isdiagonal:
                    return matrix(x=first.x - second.x, isdiagonal=True,
                                  row_names=first.row_names,
                                  col_names=first.col_names)
                elif first.isdiagonal:
                    elem_sub = -1.0 * second.newx
                    for j in xrange(first.shape[0]):
                        elem_sub[j, j] += first.x[j, 0]
                    return matrix(x=elem_sub, row_names=first.row_names,
                                  col_names=first.col_names)
                elif second.isdiagonal:
                    elem_sub = first.newx
                    for j in xrange(second.shape[0]):
                        elem_sub[j, j] -= second.x[j, 0]
                    return matrix(x=elem_sub, row_names=first.row_names,
                                  col_names=first.col_names)
                else:
                    return matrix(x=first.x - second.x,
                                  row_names=first.row_names,
                                  col_names=first.col_names)


    def __add__(self, other):
        """addition overload.  tries to speedup by checking for
            scalars of diagonal matrices on either side of operator
        Args:
            other : [scalar,numpy.ndarray,matrix object]
        Returns:
            matrix
        Raises:
            AssertionError if other is not aligned with self
            Exception is other is not in supported types
            NotImplementedError for diagonal matrices
        """
        if np.isscalar(other):
            return matrix(x=self.x + other)
        if isinstance(other, np.ndarray):
            assert self.shape == other.shape, \
                "matrix.__add__(): shape mismatch: "+\
                str(self.shape) + ' ' + str(other.shape)
            if self.isdiagonal:
                raise NotImplementedError("matrix.__add__ not supported for" +
                                          "diagonal self")
            else:
                return matrix(x=self.x + other, row_names=self.row_names,
                              col_names=self.col_names)
        elif isinstance(other, matrix):
            if self.autoalign and other.autoalign \
                    and not self.element_isaligned(other):
                common_rows = get_common_elements(self.row_names,
                                                  other.row_names)
                common_cols = get_common_elements(self.col_names,
                                                  other.col_names)
                first = self.get(row_names=common_rows, col_names=common_cols)
                second = other.get(row_names=common_rows, col_names=common_cols)
            else:
                assert self.shape == other.shape, \
                    "matrix.__add__(): shape mismatch: " +\
                    str(self.shape) + ' ' + str(other.shape)
                first = self
                second = other
            if first.isdiagonal and second.isdiagonal:
                return matrix(x=first.x + second.x, isdiagonal=True,
                              row_names=first.row_names,
                              col_names=first.col_names)
            elif first.isdiagonal:
                ox = second.newx
                for j in xrange(first.shape[0]):
                    ox[j, j] += first.__x[j]
                return matrix(x=ox, row_names=first.row_names,
                              col_names=first.col_names)
            elif second.isdiagonal:
                x = first.x
                for j in xrange(second.shape[0]):
                    x[j, j] += second.x[j]
                return matrix(x=x, row_names=first.row_names,
                              col_names=first.col_names)
            else:
                return matrix(x=first.x + second.x, row_names=first.row_names,
                              col_names=first.col_names)
        else:
            raise Exception("matrix.__add__(): unrecognized type for " +
                            "other in __add__: " + str(type(other)))


    def __mul__(self, other):
        """multiplication overload.  tries to speedup by checking for scalars or
            diagonal matrices on either side of operator
        Args:
            other : [scalar,numpy.ndarray,matrix object]
        Returns:
            matrix object
        Raises:
            AssertionError if other is not aligned with self
            Exception is other is not in supported types
        """
        if np.isscalar(other):
            return matrix(x=self.__x.copy() * other)
        elif isinstance(other, np.ndarray):
            assert self.shape[1] == other.shape[0], \
                "matrix.__mul__(): matrices are not aligned: "+\
                str(self.shape) + ' ' + str(other.shape)
            if self.isdiagonal:
                return matrix(x=np.dot(np.diag(self.__x).transpose(), other))
            else:
                return matrix(x=np.dot(self.__x, other))
        elif isinstance(other, matrix):
            if self.autoalign and other.autoalign \
                    and not self.mult_isaligned(other):
                common = get_common_elements(self.col_names,other.row_names)
                assert len(common) > 0,"matrix.__mult__():self.col_names "+\
                                       "and other.row_names"+\
                                       "don't share any common elements"
                #--these should be aligned
                if isinstance(self, cov):
                    first = self.get(row_names=common, col_names=common)
                else:
                    first = self.get(row_names=self.row_names, col_names=common)
                if isinstance(other, cov):
                    second = other.get(row_names=common, col_names=common)
                else:
                    second = other.get(row_names=common,
                                       col_names=other.col_names)

            else:
                assert self.shape[1] == other.shape[0], \
                    "matrix.__mul__(): matrices are not aligned: "+\
                    str(self.shape) + ' ' + str(other.shape)
                first = self
                second = other
            if first.isdiagonal and second.isdiagonal:
                elem_prod = matrix(x=first.x.transpose() * second.x,
                                   row_names=first.row_names,
                                   col_names=second.col_names)
                elem_prod.isdiagonal = True
                return elem_prod
            elif first.isdiagonal:
                ox = second.newx
                for j in range(first.shape[0]):
                    ox[j, :] *= first.x[j]
                return matrix(x=ox, row_names=first.row_names,
                              col_names=second.col_names)
            elif second.isdiagonal:
                x = first.newx
                ox = second.x
                for j in range(first.shape[1]):
                    x[:, j] *= ox[j]
                return matrix(x=x, row_names=first.row_names,
                              col_names=second.col_names)
            else:
                return matrix(np.dot(first.x, second.x),
                              row_names=first.row_names,
                              col_names=second.col_names)
        else:
            raise Exception("matrix.__mul__(): unrecognized " +
                            "other arg type in __mul__: " + str(type(other)))


    def __rmul__(self, other):
        raise NotImplementedError()


    def __set_svd(self):
        """private method to set SVD components
        Args:
            None
        Returns:
            None
        Raises:
            Exception is SVD process fails
        """
        if self.isdiagonal:
            x = np.diag(self.x.flatten())
        else:
            #--just a pointer to x
            x = self.x
        try:
            u, s, v = la.svd(x, full_matrices=True)
            v = v.transpose()
        except:
            try:
                v, s, u = la.svd(x.transpose(), full_matrices=True)
                u = u.transpose()
            except:
                raise Exception("matrix.__set_svd(): " +
                                "unable to compute SVD of self.x")
        col_names = []
        [col_names.append("left_sing_vec_" + str(i + 1))
         for i in xrange(u.shape[1])]
        self.__u = matrix(x=u, row_names=self.row_names,
                          col_names=col_names, autoalign=False)
        sing_names = []
        [sing_names.append("sing_val_" + str(i + 1))
         for i in xrange(s.shape[0])]
        self.__s = matrix(x=np.atleast_2d(s).transpose(), row_names=sing_names,
                          col_names=sing_names, isdiagonal=True,
                          autoalign=False)
        col_names = []
        [col_names.append("right_sing_vec_" + str(i + 1))
         for i in xrange(v.shape[0])]
        self.__v = matrix(v, row_names=self.col_names, col_names=col_names,
                          autoalign=False)


    def mult_isaligned(self, other):
        """check if matrices are aligned for multiplication
        Args:
            other : matrix
        Returns:
            True if aligned
            False if not aligned
        Raises:
            Exception if other is not a matrix object
        """
        assert isinstance(other, matrix), \
            "matrix.isaligned(): other argumnent must be type matrix, not: " +\
            str(type(other))
        if self.col_names == other.row_names:
            return True
        else:
            return False


    def element_isaligned(self, other):
        """check if matrices are aligned for element-wise operations
        Args:
            other : matrix
        Returns:
            True if aligned
            False if not aligned
        Raises:
            Exception if other is not a matrix object
        """
        assert isinstance(other, matrix), \
            "matrix.isaligned(): other argument must be type matrix, not: " +\
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
            return self.__x
        return np.diag(self.__x.flatten())

    @property
    def shape(self):
        """get the shape of x
        Args:
            None
        Returns:
            tuple of ndims
        Raises:
            Exception if self is not 2D
        """
        if self.__x is not None:
            if self.isdiagonal:
                return (max(self.__x.shape), max(self.__x.shape))
            if len(self.__x.shape) == 1:
                raise Exception("matrix.shape: matrix objects must be 2D")
            return self.__x.shape
        return None


    @property
    def T(self):
        """wrapper function for transpose
        """
        return self.transpose


    @property
    def transpose(self):
        """transpose operation
        Args:
            None
        Returns
            transpose of self
        Raises:
            None
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
        Args:
            None
        Returns
            inverse of self
        Raises:
            None
        """
        if self.isdiagonal:
            return type(self)(x=1.0 / self.__x, isdiagonal=True,
                              row_names=self.row_names,
                              col_names=self.col_names,
                              autoalign=self.autoalign)
        else:
            return type(self)(x=la.inv(self.__x), row_names=self.row_names,
                              col_names=self.col_names,
                              autoalign=self.autoalign)

    @property
    def sqrt(self):
        """square root operation
        Args:
            None
        Returns:
            square root of self
        Raises:
            None
        """
        if self.isdiagonal:
            return type(self)(x=np.sqrt(self.__x), isdiagonal=True,
                              row_names=self.row_names,
                              col_names=self.col_names,
                              autoalign=self.autoalign)
        else:
            return type(self)(x=la.sqrtm(self.__x), row_names=self.row_names,
                              col_names=self.col_names,
                              autoalign=self.autoalign)


    @property
    def s(self):
        """the singular value (diagonal) matrix
        """
        if self.__s is None:
            self.__set_svd()
        return self.__s


    @property
    def u(self):
        """the left singular vector matrix
        """
        if self.__u is None:
            self.__set_svd()
        return self.__u


    @property
    def v(self):
        """the right singular vector matrix
        """
        if self.__v is None:
            self.__set_svd()
        return self.__v


    def indices(self, names, axis=None):
        """get the row and col indices of names
        Args:
            names : [enumerable] column and/or row names
            axis : [int] the axis to search.
        Returns:
            numpy.ndarray : indices of names.  if axis is None, two ndarrays
                are returned, corresponding the indices of names for each axis
        Raises:
            Exception if a name is not found
            Exception if axis not in [0,1]
        """
        row_idxs, col_idxs = [], []
        for name in names:
            if name.lower() not in self.col_names \
                    and name.lower() not in self.row_names:
                raise Exception('matrix.indices(): name not found: ' + name)
            if name.lower() in self.col_names:
                col_idxs.append(self.col_names.index(name))
            if name.lower() in self.row_names:
                row_idxs.append(self.row_names.index(name))
        if axis is None:
            return np.array(row_idxs, dtype=np.int32),\
                np.array(col_idxs, dtype=np.int32)
        elif axis == 0:
            if len(row_idxs) != len(names):
                raise Exception("matrix.indices(): " +
                                "not all names found in row_names")
            return np.array(row_idxs, dtype=np.int32)
        elif axis == 1:
            if len(col_idxs) != len(names):
                raise Exception("matrix.indices(): " +
                                "not all names found in col_names")
            return np.array(col_idxs, dtype=np.int32)
        else:
            raise Exception("matrix.indices(): " +
                            "axis argument must 0 or 1, not:" + str(axis))


    def align(self, names, axis=None):
        """reorder self by names
        Args:
            names : [enumerable] names in row and\or column names
            axis : [int] the axis to reorder. if None, reorder both axes
        Returns:
            None
        Raises:
            Exception if axis not passed and needed
            Exception if axis not in [0,1]
            AssertionError if name(s) not found
        """
        if not isinstance(names, list):
            names = [names]
        row_idxs,col_idxs = self.indices(names)
        if self.isdiagonal or isinstance(self, cov):
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
                raise Exception("matrix.align(): must specify axis in " +
                                "align call for non-diagonal instances")
            if axis == 0:
                assert row_idxs.shape[0] == self.shape[0], \
                    "matrix.align(): not all names found in self.row_names"
                self.__x = self.__x[row_idxs, :]
                row_names = []
                [row_names.append(self.row_names[i]) for i in row_idxs]
                self.row_names = row_names
            elif axis == 1:
                assert col_idxs.shape[0] == self.shape[1], \
                    "matrix.align(): not all names found in self.col_names"
                self.__x = self.__x[:, col_idxs]
                col_names = []
                [col_names.append(self.col_names[i]) for i in row_idxs]
                self.col_names = col_names
            else:
                raise Exception("matrix.align(): axis argument to align()" +
                                " must be either 0 or 1")


    def get(self,row_names=None, col_names=None, drop=False):
        """get a (sub)matrix ordered on row_names or col_names
        Args:
            row_names : [enumerable] row_names for new matrix
            col_names : [enumerable] col_names for new matrix
            drop : [bool] flag to remove row_names and/or col_names
        Returns:
            matrix
        Raises:
            Exception if row_names and col_names are both None
        """
        if row_names is None and col_names is None:
            raise Exception("matrix.get(): must pass at least" +
                            " row_names or col_names")

        if row_names is not None and not isinstance(row_names, list):
            row_names = [row_names]
        if col_names is not None and not isinstance(col_names, list):
            col_names = [col_names]

        if isinstance(self,cov):
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
            return cov(x=extract, names=names, isdiagonal=self.isdiagonal)
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
        Args:
            names : [enumerable] names to drop
            axis : [int] the axis to drop from. must be in [0,1]
        Returns:
            None
        Raises:
            Exception if axis is None
            Exception if all the names along an axis are in names arg
            Exception if names aren't found
        """
        if axis is None:
            raise Exception("matrix.drop(): axis arg is required")
        if not isinstance(names, list):
            names = [names]
        idxs = self.indices(names, axis=axis)

        if self.isdiagonal or isinstance(self, cov):
            self.__x = np.delete(self.__x, idxs, 0)
            idxs = np.sort(idxs)
            for idx in idxs[::-1]:
                del self.row_names[idx]
                del self.col_names[idx]
        elif axis == 0:
            if idxs.shape[0] == self.shape[0]:
                raise Exception("matrix.drop(): can't drop all rows")
            elif idxs.shape == 0:
                raise Exception("matrix.drop(): nothing to drop on axis 0")
            self.__x = np.delete(self.__x, idxs, 0)
            idxs = np.sort(idxs)
            for idx in idxs[::-1]:
                del self.row_names[idx]
        elif axis == 1:
            if idxs.shape[0] == self.shape[1]:
                raise Exception("matrix.drop(): can't drop all cols")
            if idxs.shape == 0:
                raise Exception("matrix.drop(): nothing to drop on axis 1")
            self.__x = np.delete(self.__x, idxs, 1)
            idxs = np.sort(idxs)
            for idx in idxs[::-1]:
                del self.col_names[idx]
        else:
            raise Exception("matrix.drop(): axis argument must be 0 or 1")


    def extract(self, row_names=None, col_names=None):
        """wrapper method that gets then drops elements
        """
        if row_names is None and col_names is None:
            raise Exception("matrix.extract() " +
                            "row_names and col_names both None")
        extract = self.get(row_names, col_names, drop=True)
        return extract


    def to_binary(self, filename):
        """write a pest-compatible binary file
        Args:
            filename : [str] filename to save binary file
        Returns:
            None
        Raises:
            None
        """
        f = open(filename, 'wb')
        nnz = np.count_nonzero(self.x) #number of non-zero entries
        #--write the header
        header = np.array((-self.shape[1], -self.shape[0], nnz),
                          dtype=self.binary_header_dt)
        header.tofile(f)
        #--get the indices of non-zero entries
        row_idxs, col_idxs = np.nonzero(self.x)
        icount = row_idxs + 1 + col_idxs * self.shape[0]
        #--flatten the array
        flat = self.x[row_idxs, col_idxs].flatten()
        #--zip up the index position and value pairs
        data = np.array(zip(icount, flat), dtype=self.binary_rec_dt)
        #--write
        data.tofile(f)

        for name in self.col_names:
            if len(name) > self.par_length:
                name = name[:self.par_length - 1]
            elif len(name) < self.par_length:
                for i in range(len(name), self.par_length):
                    name = name + ' '
            f.write(name)
        for name in self.row_names:
            if len(name) > self.obs_length:
                name = name[:self.obs_length - 1]
            elif len(name) < self.obs_length:
                for i in range(len(name), self.obs_length):
                    name = name + ' '
            f.write(name)
        f.close()


    def from_binary(self, filename):
        """load from pest-compatible binary file
        Args:
            filename : [str] filename to save binary file
        Returns:
            None
        Raises:
            TypeError if the binary file is deprecated version
        """
        f = open(filename, 'rb')
        #--the header datatype
        itemp1, itemp2, icount = np.fromfile(f, self.binary_header_dt, 1)[0]
        if itemp1 >= 0:
            raise TypeError('matrix.from_binary(): Jco produced by ' +
                            'deprecated version of PEST,' +
                            'Use JCOTRANS to convert to new format')
        ncol, nrow = abs(itemp1), abs(itemp2)
        self.__x = np.zeros((nrow, ncol))
        #--read all data records
        #--using this a memory hog, but really fast
        data = np.fromfile(f, self.binary_rec_dt, icount)
        icols = ((data['j'] - 1) / nrow) + 1
        irows = data['j'] - ((icols - 1) * nrow)
        self.__x[irows - 1, icols - 1] = data["dtemp"]
        #--read obs and parameter names
        for j in xrange(self.shape[1]):
            name = struct.unpack(str(self.par_length) + "s",
                                 f.read(self.par_length))[0].strip().lower()
            self.col_names.append(name)
        for i in xrange(self.shape[0]):
            name = struct.unpack(str(self.obs_length) + "s",
                                 f.read(self.obs_length))[0].strip().lower()
            self.row_names.append(name)
        f.close()
        assert len(self.row_names) == self.shape[0],\
          "matrix.from_binary() len(row_names) (" + str(len(self.row_names)) +\
          ") != self.shape[0] (" + str(self.shape[0]) + ")"
        assert len(self.col_names) == self.shape[1],\
          "matrix.from_binary() len(col_names) (" + str(len(self.col_names)) +\
          ") != self.shape[1] (" + str(self.shape[1]) + ")"

    def to_ascii(self, out_filename, icode=2):
        """write a pest-compatible ASCII matrix/vector file
        Args:
            out_filename : [str] output filename
            icode : [int] pest-style info code for matrix style
        Returns:
            None
        Raises:
            None
        """
        nrow, ncol = self.shape
        f_out = open(out_filename, 'w')
        f_out.write(' {0:7.0f} {1:7.0f} {2:7.0f}\n'.
                    format(nrow, ncol, icode))
        if self.isdiagonal:
            x = np.diag(self.__x[:, 0])
        else:
            x = self.__x
        np.savetxt(f_out, x, fmt='%15.7E', delimiter='')
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


    def from_ascii(self, filename):
        """load a pest-compatible ASCII matrix/vector file
        Args:
            filename : [str] name of the file to read
        Returns:
            None
        Raises:
            Exception if file is not correct
        """
        f = open(filename, 'r')
        raw = f.readline().strip().split()
        nrow, ncol, icode = int(raw[0]), int(raw[1]), int(raw[2])
        #x = np.fromfile(f, dtype=self.double, count=nrow * ncol, sep=' ')
        # this painfully slow and ungly read is needed to catch the
        # fortran floating points that have 3-digit exponents,
        # which leave out the base (e.g. 'e') : "-1.23455+300"
        count = 0
        x = []
        while True:
            line = f.readline()
            if line == '':
                raise Exception("matrix.from_ascii() error: EOF")
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
                        raise Exception("matrix.from_ascii() error: " +
                                        " can't cast " + r + " to float")
                count += 1
                if count == (nrow * ncol):
                    break
            if count == (nrow * ncol):
                    break
        x = np.array(x,dtype=self.double)
        x.resize(nrow, ncol)
        self.__x = x
        line = f.readline().strip().lower()
        if not line.startswith('*'):
            raise Exception('matrix.from_ascii(): error loading ascii file," +\
                "line should start with * not ' + line)
        if 'row' in line and 'column' in line:
            assert nrow == ncol
            names = []
            for i in range(nrow):
                line = f.readline().strip().lower()
                names.append(line)
            self.row_names = copy.deepcopy(names)
            self.col_names = names
            self.isdiagonal = True
        else:
            names = []
            for i in range(nrow):
                line = f.readline().strip().lower()
                names.append(line)
            self.row_names = names
            line = f.readline().strip().lower()
            assert "column" in line, \
                "matrix.from_ascii(): line should be * column names " +\
                "instead of: " + line
            names = []
            for j in range(ncol):
                line = f.readline().strip().lower()
                names.append(line)
            self.col_names = names
        f.close()


    def df(self):
        return self.to_dataframe()


    def to_dataframe(self):
        """return a pandas dataframe of the matrix object
        Args:
            None
        Returns:
            pandas dataframe
        Raises:
            None
        """
        if self.isdiagonal:
            x = np.diag(self.__x[:, 0])
        else:
            x = self.__x
        return pandas.DataFrame(data=x,index=self.row_names,columns=self.col_names)

    def to_sparse(self,trunc = 0.0):
        """get the CSR sparse matrix representation of matrix
        Args:
            None
        Returns:
            scipy sparse matrix object
        Raises:
            Exception rethrow on scipy.sparse import failure
        """
        try:
            import scipy.sparse as sparse
        except:
            raise Exception("mat.to_sparse() error importing scipy.sparse")
        iidx, jidx = [], []
        data = []
        nrow, ncol = self.shape
        for i in xrange(nrow):
            for j in xrange(ncol):
                val = self.x[i,j]
                if val > trunc:
                    iidx.append(i)
                    jidx.append(j)
                    data.append(val)
        # csr_matrix( (data,(row,col)), shape=(3,3)
        return sparse.csr_matrix((data, (iidx, jidx)), shape=(self.shape))



class jco(matrix):
    """a thin wrapper class to get more intuitive attribute names
    """
    def __init(self, **kwargs):
        super(jco, self).__init__(kwargs)


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



class cov(matrix):
    """a subclass of matrix for handling diagonal or dense covariance matrices
        todo:block diagonal
    """
    def __init__(self, x=None, names=[], row_names=[], col_names=[],
                 isdiagonal=False, autoalign=True):
        """constructor for cov
        Args:
            x : numpy.ndarray
            names : [enumerable] names for both columns and rows
            row_names : [enumerable] names for rows
            col_names : [enumerable] names for columns
            isdiagonal : [bool] diagonal matrix flag
            autoalign : [bool] autoalignment flag
        Returns:
            None
        Raises
            None
        """
        self.__identity = None
        self.__zero = None
        if len(names) != 0 and len(row_names) == 0:
            row_names = names
        if len(names) != 0 and len(col_names) == 0:
            col_names = names
        super(cov, self).__init__(x=x, isdiagonal=isdiagonal,
                                  row_names=row_names,
                                  col_names=col_names,
                                  autoalign=autoalign)


    @property
    def identity(self):
        """get an identity matrix like self
        """
        if self.__identity is None:
            self.__identity = cov(x=np.atleast_2d(np.ones(self.shape[0]))
                                  .transpose(), names=self.row_names,
                                  isdiagonal=True)
        return self.__identity


    @property
    def zero(self):
        if self.__zero is None:
            self.__zero = cov(x=np.atleast_2d(np.zeros(self.shape[0]))
                              .transpose(), names=self.row_names,
                              isdiagonal=True)
        return self.__zero


    def condition_on(self,conditioning_elements):
        """get a new covariance object that is conditional on knowing some
            elements.  uses Schur's complement for conditional covariance
            propagation
        Args:
            conditioning_elements : [enumerable] names of elements to
                                    condition on
        Returns:
            Cov object
        Raises:
            AssertionError is conditioning element not found
        """
        for iname, name in enumerate(conditioning_elements):
            conditioning_elements[iname] = name.lower()
            assert name.lower() in self.col_names,\
                "cov.condition_on() name not found: " + name
        keep_names = []
        for name in self.col_names:
            if name not in conditioning_elements:
                keep_names.append(name)
        #C11
        new_cov = self.get(keep_names)
        if self.isdiagonal:
            return new_cov
        #C22^1
        cond_cov = self.get(conditioning_elements).inv()
        #C12
        upper_off_diag = self.get(keep_names, conditioning_elements)
        return new_cov - (upper_off_diag * cond_cov * upper_off_diag.T)


    def to_uncfile(self, unc_file, covmat_file="cov.mat", var_mult=1.0):
        """write a pest-compatible uncertainty file
        Args:
            unc_file : [str] filename
            covmat : [str] covariance matrix filename
            var_mult : [float] variance multiplier
        Returns:
            None
        Raises
            AssertionError is self is out of alignment
            Exception if covmat_file is None and self is not diagonal
        """
        assert len(self.row_names) == self.shape[0], \
            "cov.to_uncfile(): len(row_names) != x.shape[0] "
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
                raise Exception("cov.to_uncfile(): can't write non-diagonal " +
                                "object as standard deviation block")


    def from_obsweights(self, pst_file):
        """load covariance from observation weights
        Args:
            pst_file : [str] pest control file name
        Returns:
            None
        Raises:
            None
        """
        if not pst_file.endswith(".pst"):
            pst_file += ".pst"
        pst = phand.pst(pst_file)
        self.from_observation_data(pst)


    def from_observation_data(self, pst):
        """load covariances from a pandas dataframe
                of the pst obseravtion data section
        Args:
            pst : [pst object]
        Returns:
            None
        Raises:
            None
        """
        nobs = pst.observation_data.shape[0]
        if pst.mode == "estimation":
            nobs += pst.nprior
        x = np.zeros((nobs, 1))
        onames = []
        ocount = 0
        for idx,row in pst.observation_data.iterrows():
            w = float(row["weight"])
            w = max(w, 1.0e-30)
            x[ocount] = (1.0 / w) ** 2
            ocount += 1
            onames.append(row["obsnme"].lower())
        if pst.mode == "estimation" and pst.nprior > 0:
            for iidx, row in pst.prior_information.iterrows():
                w = float(row["weight"])
                w = max(w, 1.0e-30)
                x[ocount] = (1.0 / w) ** 2
                ocount += 1
                onames.append(row["pilbl"].lower())
        self._matrix__x = x
        self.row_names = copy.deepcopy(onames)
        self.col_names = onames
        self.isdiagonal = True


    def from_parbounds(self, pst_file):
        """load covariances from a pest control file parameter data section
        Args:
            pst_file : [str] pest control file name
        Returns:
            None
        Raises:
            None
        """
        if not pst_file.endswith(".pst"):
            pst_file += ".pst"
        pst = phand.pst(pst_file)
        self.from_parameter_data(pst)


    def from_parameter_data(self, pst):
        """load covariances from a pandas dataframe of the
                pst parameter data section
        Args:
            pst : [pst object]
        Returns:
            None
        Raises:
            None
        """
        npar = pst.npar_adj
        x = np.zeros((npar, 1))
        names = []
        idx = 0
        for i, row in pst.parameter_data.iterrows():
            t = row["partrans"]
            #if t in ["fixed", "tied"]:
            #    continue
            lb, ub = row["parlbnd"], row["parubnd"]
            if t == "log":
                var = ((np.log10(ub) - np.log10(lb)) / 4.0) ** 2
            else:
                var = ((ub - lb) / 4.0) ** 2
            x[idx] = var
            names.append(row["parnme"].lower())
            idx += 1
        self._matrix__x = x
        assert len(names) == x.shape[0]
        self.row_names = copy.deepcopy(names)
        self.col_names = copy.deepcopy(names)
        self.isdiagonal = True


    def from_uncfile(self, filename):
        """load covariances from a pest-compatible uncertainty file
        Args:
            filename : [str] uncertainty file name
        Returns:
            None
        Raises:
            Exception for duplicate entries
            Exception for wrong file structure
        """
        nentries = self.get_uncfile_dimensions(filename)
        self._matrix__x = np.zeros((nentries, nentries))
        f = open(filename, 'r')
        self.isdiagonal = True
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
                        self._matrix__x[idx, idx] = val**2
                        if name in self.row_names:
                            raise Exception("cov.from_uncfile():" +
                                            "duplicate name: " + str(name))
                        self.row_names.append(name)
                        self.col_names.append(name)
                        idx += 1

                elif 'covariance_matrix' in line:
                    self.isdiagonal = False
                    var = 1.0
                    while True:
                        line2 = f.readline().strip().lower()
                        if line2.strip().lower().startswith("end"):
                            break
                        if line2.startswith('file'):
                            cov = matrix()
                            cov.from_ascii(line2.split()[1])

                        elif line2.startswith('variance_multiplier'):
                            var = float(line2.split()[1])
                        else:
                            raise Exception("cov.from_uncfile(): " +
                                            "unrecognized keyword in" +
                                            "std block: " + line2)
                    if var != 1.0:
                        cov.__x *= var
                    for name in cov.row_names:
                        if name in self.row_names:
                            raise Exception("cov.from_uncfile():" +
                                            " duplicate name: " + str(name))
                    self.row_names.extend(cov.row_names)
                    self.col_names.extend(cov.col_names)

                    for i, rname in enumerate(cov.row_names):
                        self._matrix__x[idx + i,
                                        idx:idx + cov.shape[0]] = cov.x[i, :]
                    idx += cov.shape[0]
                else:
                    raise Exception('cov.from_uncfile(): ' +
                                    'unrecognized block:' + str(line))
        f.close()


    def get_uncfile_dimensions(self,filename):
        """quickly read an uncertainty file to find the dimensions
        Args:
            filename : [str] uncertainty filename
        Returns:
            nentries : [int] number of elements in file
        Raises:
            Exception for wrong file structure
        """
        f = open(filename,'r')
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
                            cov = matrix()
                            cov.from_ascii(line2.split()[1])
                            nentries += len(cov.row_names)
                        elif line2.startswith('variance_multiplier'):
                            var = float(line2.split()[1])
                        else:
                            raise Exception('cov.get_uncfile_dimensions(): ' +
                            'unrecognized keyword in covariance block: ' +
                                            line2)
                else:
                    raise Exception('cov.get_uncfile_dimensions():' +
                                    'unrecognized block:' + str(line))
        f.close()
        return nentries


def test():
    arr = np.arange(0,12)
    arr.resize(4,3)
    first = jco(x=arr,col_names=["p1","p2","p3"],row_names=["o1","o2","o3","o4"])
    first.to_binary("test.bin")
    first.from_binary("test.bin")

    first = jco(x=np.ones((4,3)),col_names=["p1","p2","p3"],row_names=["o1","o2","o3","o4"])
    second = cov(x=np.ones((3,3))+1.0,names=["p1","p2","p3"],isdiagonal=False)
    #second = cov(x=np.ones((3,1))+1.0,names=["p1","p2","p3"],isdiagonal=True)
    third = cov(x=np.ones((4,1))+2.0,names=["o1","o2","o3","o4"],isdiagonal=True)
    si = second.identity
    result = second - second.identity
    #--add and sub
    newfirst = first.get(row_names=["o1"],col_names="p1")
    result = newfirst - first
    result = first - newfirst
    result = newfirst + first
    result = first + newfirst
    newfirst = first.get(row_names=["o1","o2"],col_names="p1")
    result = newfirst - first
    result = first - newfirst
    result = newfirst + first
    result = first + newfirst
    newfirst = first.get(row_names=["o1","o2"],col_names=["p1","p3"])
    result = newfirst - first
    result = first - newfirst
    result = newfirst + first
    result = first + newfirst



    #--mul test
    result = first.T * third * first
    result = first * second

    newfirst = first.get(col_names="p1")
    result = newfirst * second
    result = second * newfirst.T

    newfirst = first.get(col_names=["p1","p2"])
    result = newfirst * second
    result = second * newfirst.T

    newfirst = first.get(row_names=["o1"])
    result = newfirst * second
    result = second * newfirst.T

    newfirst = first.get(row_names=["o1","o2"])
    result = newfirst * second
    result = second * newfirst.T
    result = newfirst.T * third * newfirst

    newthird = third.get(row_names=["o1"])
    result = first.T * newthird * first

if __name__ == "__main__":
    #test()
    # a = np.random.random((10, 5))
    # row_names = []
    # [row_names.append("row_{0:02d}".format(i)) for i in xrange(10)]
    # col_names = []
    # [col_names.append("col_{0:02d}".format(i)) for i in xrange(5)]
    # m = matrix(x=a, row_names=row_names, col_names=col_names)
    # print (m.T * m).inv
    #
    # m.to_binary("mat_test.bin")
    # m1 = matrix()
    # m1.from_binary("mat_test.bin")
    # print m1.row_names
    m = cov()
    m.from_ascii("post.cov")

