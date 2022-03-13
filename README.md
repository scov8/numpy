# Introduction to Numpy

Vectors and matrices are not natively managered in Python. Although a list of lists can be used as a matrix, we need a set of function to easily work with it. For this reason, the Numpy library is essential for dealing with vectors and matrices. The Numpy array (`numpy.ndarray`) is generally used for vectors, matrices, and multi-dimensional data sets. Respect to a list, the elements of a Numpy array must be homogeneous. In other words, all elements of a Numpy array must be of the same type that is defined when the array is created.

To begin to use the Numpy library, we need to import the `numpy` module.

``` {.python}
import numpy as np
```
Note that we have used an alias. In this way, we will access only using `np.` to the variables and functions of the Numpy library.

For convenience, the Numpy array will be called simply array. An array can be created in different ways. A first way is to create if from a list using the function `np.array( )`.

``` {.python}
# to create new vector from a lists we can use the `np.array` function.
l1 = [ 4, 5, 1, 3, 4]  # it is a list
a1 = np.array(l1)      # it is an array
print('Vector:', a1)

l2 = [ [5.0, 1.4,], [5.5, 2.1]]  # it is a list of lists
a2 = np.array(l2)                # it is an array
print('Matrix:\n', a2)

# We can explicitly define the type of data using the `dtype` argument: 
a3 = np.array(l2, dtype=int)     
print('Matrix of integeres:\n', a3)
```

## Data Type

Numpy supportd differents data types, such as: `np.uint8`, `np.int64`, `np.float32`, `np.float64`, `np.complex64`, `np.bool`.

Using the `dtype` property of an array, we can know the data type.

``` {.python}
print("The data type of 'a1' is ", a1.dtype)
print("The data type of 'a2' is ", a2.dtype)
print("The data type of 'a3' is ", a3.dtype)
```

## Properties of arrays

An array has other useful properties:

-   `.ndim` : returns the number of dimensions.
-   `.shape` : returns a tuple where each element is the length along that dimension.
-   `.size` : returns the number of elements of the array.
-   `.nbytes`: returns the bytes of the array.

``` {.python}
print("The  ndim of 'a1' is ", a1.ndim )
print("The  ndim of 'a2' is ", a2.ndim )
print("The shape of 'a1' is ", a1.shape)
print("The shape of 'a2' is ", a2.shape)
print("The  size of 'a1' is ", a1.size )
print("The  size of 'a2' is ", a2.size )
```

# Creating Arrays

There are many functions that generate arrays of different forms. For examples:

-   `np.array( )`: constructs an array from a list.
-   `np.zeros( )`: creates an array with zeros of a given shape.
-   `np.ones( )`: creates an array with ones of a given shape.
-   `np.arange( )`: creates a mono-dimensional array that contains a sequence of numbers.
-   `np.copy( )`: creates an array replicating another array.
-   `np.random.rand( )`: creates an array with randomly-generated values from a uniform distribution.
-   `np.random.randn( )`: creates an array with randomly-generated values from a normal distribution.

``` {.python}
# to create an array with zeros of shape (2,1,3) and dtype uint8
z = np.zeros((2,1,3), dtype=np.uint8)
print('array with zeros:\n', z)

# to create an array with ones of shape (3,2) and dtype float32
o = np.ones((3,2), dtype=np.float32)
print('array with oness:\n', o)

# to create an array with the even numbers from 0 to 10 (10 is excluded).
r = np.arange(0, 10, 2) # arguments: start, stop, step
print('array with a sequence:\n', r)

# to create an array of shape (4,4) with uniform random numbers in [0,1[
u = np.random.rand(4,4)
print('random array:\n', u)
# Note that the result of 'np.random.rand( )' is different for each execution 
```

# Indexing and Slicing

We can access a single element of an array or a sub-array using the square brackets likewise to the indexing and slicing of Python list.
Moreover, Numpy also supports the Fancy-Indexing and the Boolean-Indexing, for a complete guide see the [official page](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html).

Example of single element indexing:

``` {.python}
matrix = np.random.rand(5,4)
element = matrix[0,1] # get the element in first row and second column
matrix[2,0] = 5.6     # set the element in third row and first column
```

Example of simple slicing:

``` {.python}
matrix = np.random.rand(5,4)
row = matrix[2,:]   # get the third row
col = matrix[:,-1]  # get the last column
```

Note that in the previous example the arrays `row` and `col` are the same data type of `matrix` but a different number of dimensions.

Also for the arrays, we can use the syntax `start:stop:step`, for examples:

``` {.python}
matrix = np.random.rand(5,4)
a = matrix[:5:2,:]      # sub-array with the first two rows
b = matrix[1:3,2:4]   # sub-array 2x2
#matrix[:,1:2] = 0     # set to zero the second and third columns
print(matrix)
print(a)
print(b)
```

# Element-wise operations and functions

Numpy library provides efficient element-wise operations and functions applied across one or more arrays. For examples:

-   Arithmetic Operators (`+`, `-`, `*`, `/`, \...)
-   Comparisons (`==`, `>`, `!=`, \...)
-   Boolean Functions (`np.logical_and( )`, `np.logical_not( )`, \...)
-   Math Functions (`np.maximum( )`, `np.exp( )`, `np.sin( )`, \...)

``` {.python}
l1 = [ 4, 5, 1, 3, 4]  # it is a list
a1 = np.array(l1)      # it is an array

l2 = [ 0, 3, 2, 7, 9]  # it is a list
a2 = np.array(l2)      # it is an array

# the sum of two lists is different from the sum of two arrays
print('the sum of the two lists :', l1+l2)
print('the sum of the two arrays:', a1+a2)
```

``` {.python}
a1 = np.array([[4, 5], [1, 3]] ) # fist array
a2 = np.array([[1, 3], [2, 7]] ) # second array

# Some examples of element-wise operations and functions 
print('the maximum  of the two arrays:\n', np.maximum(a1, a2) )
print('the prodoct  of the two arrays:\n', a1 * a2) # it is the element-wise prodoct
print('the division of the two arrays:\n', a1 / a2) # it is the element-wise division
print('the floor-division of the two arrays:\n', a1 // a2) # it is the element-wise floor-division
```

# Stacking & Reshaping & Transposing 

Numpy provides functions to stack, to reshape and to transpose the arrays.

## `np.concatenate(arrays_list, axis)`

joins a list of arrays along an axis.

``` {.python}
a0 = np.array([[1, 2], [3, 4]])  # it is an array 2x2
a1 = np.array([[5, 6]])          # it is an array 1x2
a2 = np.array([[5, ], [6,]])     # it is an array 2x1
print('matrix "a0":\n', a0)
print('shape of "a0":', a0.shape)
print('matrix "a1":\n', a1)
print('shape of "a1":', a1.shape)
print('matrix "a2":\n', a2)
print('shape of "a2":', a1.shape)
print()
```

``` {.python}
# Concatenation along the rows
b = np.concatenate([a0,a1], 0) # it is an array 3x2
print('matrix  "b":\n', b)
print('shape of  "b"', b.shape)
```

``` {.python}
# Concatenation along the columns
c = np.concatenate([a0,a2], 1) # it is an array 2x3
print('matrix  "c":\n',c)
print('shape of  "c"', c.shape)
```

## `np.stack(arrays_list, axis)` 

joins a list of arrays along a new axis. if axis=0 the new axis is created at the beginning while if axis=-1 the new axis is created at the end.

``` {.python}
a0 = np.array([[1, 2, 4], [3, 4, 8]])  # it is a matrix 2x3
a1 = np.array([[5, 6, 7], [5, 7, 9]])  # it is a matrix 2x3
print('matrix "a0":\n', a0)
print('shape of "a0":', a0.shape)
print('matrix "a1":\n', a1)
print('shape of "a1":', a1.shape)
print()

# stacking 
b = np.stack([a0,a1],0)      # it is an array 2x2x3
print('array  "b":\n', b)
print('shape of  "b"', b.shape) 

# stacking
c = np.stack([a0,a1], -1)   # it is an array 2x3x2
print('array  "c":\n', c)
print('shape of  "c"', c.shape) 
```

## `np.reshape(array, newshape, order='C')` 

modifies the shape to the array without changing its data.

``` {.python}
vector = np.random.rand(12)
matrix = np.reshape(vector, (4,3))

print("vector:\n", vector)
print("matrix:\n", matrix)

# "vector" and "matrix" have same data but different shapes
```

Note, by default the reshape follows the C-like index order (the rows of the matrix are placed in contiguous indexes like in C/C++).

While, executing the reshape with `order='F'`, it follows the F-like index order (the columns of the matrix are placed in contiguous indexes like in Fortran and Matlab).

``` {.python}
matrixF = np.reshape(vector, (4,3), order='F')

print("matrix in C-like order:\n", matrix)
print("matrix in F-like order:\n", matrixF)
```

## `np.transpose(array, axes)` 

swaps the dimensions of the array.

``` {.python}
# transpose of a matrix
matrix = np.random.rand(4,3)  # a bidimensional matrix

# swap the first dimension with the second one
matrixT = np.transpose(matrix, (1,0) )  

print("original  matrix:\n", matrix)
print("shape of original  matrix:", matrix.shape)
print()

print("transpose matrix:\n", matrixT)
print("shape of transpose matrix:", matrixT.shape)
print()
```

``` {.python}
# transpose of a multidimensional array
array = np.random.rand(4,2,3)  # an array with three dimensions

# swap the first dimension with the third one
arrayT = np.transpose(array, (2,1,0) )  

print("original  array:\n", array)
print("shape of original  array:", array.shape)
print()

print("transpose array:\n", arrayT)
print("shape of transpose array:", arrayT.shape)
print()
```

# Reductions

Numpy provides functions to \"aggregate\" our data along a particular axis or the whole array, such as `np.sum( )`, `np.mean( )`, `np.var( )`, `np.median( )`, `np.min( )`, `np.max( )`.

``` {.python}
a1 = np.array([[4.4, 5.0], [1.0, 3.2], [1.7, 2.4]] )  # array 3x2

print("sum of the whole array:", np.sum(a1) )
print("sum of the array along the columns:", np.sum(a1, axis=0) )
print("sum of the array along the rows:", np.sum(a1, 1) )
```

The other functions to \"aggregate\" have the same syntax.

# Save & Load {#save--load}

To save and load an array, we can use the functions `np.save( )` and `np.load( )`.

``` {.python}
a1 = np.array([[4.4, 5.0], [1.0, 3.2], [1.7, 2.4]] )  # array 3x2

np.save("/content/drive/MyDrive/array_file.npy", a1)

del a1

a1 = np.load("/content/drive/MyDrive/array_file.npy")
print(a1)

```

File extension used by Numpy to store an array is `.npy`

