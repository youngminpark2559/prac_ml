# 003_001_numpy_array

# ======================================================================
# Numpy manages data as array and performs operations in array
# At this moment, array can be considered as vector or matrix mathematically

# ======================================================================
import numpy as np

# np.array() can have list or other numpy array as arguments

# c data1: created list
data1=[5,5.4,9,0,1]

# c arr1: created numpy array using list
arr1=np.array(data1)
print(arr1)
# array([5. , 5.4, 9. , 0. , 1. ])

# c arr1: 1D array in Numpy or list perspectives, 5D vector mathematically 
print(arr1.shape)
# (5,)

# 2D list   
data2=[[5,5.4,9,0,1],
       [3,5.2,5,2,0]]
arr2=np.array(data2)
print(arr2)
# array([[5. , 5.4, 9. , 0. , 1. ],
#        [3. , 5.2, 5. , 2. , 0. ]])

# c arr2: 2D array in Numpy or list perspectives, (2,5) matrix mathematically 
print(arr2.shape)
# (2, 5)

# ======================================================================
# (3,6) 2D array or (3,6) matrix filled with all 0s
print(np.zeros((3,6)))
# array([[0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0.]])

# 10 length 1D array or 10 dimensional vector
print(np.ones(10))
# array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

# 15 length 1D array or 15 dimensional vector
print(np.arange(15))
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])

# ======================================================================
# Numpy automatically sets datatype of elements in np array by using most wide datatype
print(arr1.dtype)
# dtype('float64')

print(arr2.dtype)
# dtype('float64')

# arr1 and arr2 has same wide datatype

# --------------------------------------------------
# You also can manually set datatype
# 64 means size of bit which each element in np array will use in memory
# When you need use larger datatype, you can use larger number
arr=np.array([1,2,3,4,5],dtype=np.int64)
print(arr)
# array([1, 2, 3, 4, 5])

print(arr.dtype)
# dtype('int64')

# ======================================================================
# You can change datatype which was already determined

# c float_arr: float datatype np array
float_arr=arr.astype(np.float64)
print(float_arr)
# [1. 2. 3. 4. 5.]

print(float_arr.dtype)
# float64

# ======================================================================
# Let's talk about operation in numpy

arr1=np.array(
    [[1,2,3],
     [4,5,6]],
    dtype=np.float64)

arr2=np.array(
    [[7,8,9],
     [10,11,12]],
    dtype=np.float64)

# Try these calculations
arr1+arr2
arr1-arr2
# Note that this performs element-wise multiplication, 
# not matrix multiplication
arr1*arr2
arr1/arr2
arr1*2
arr1**0.5
1/arr1
