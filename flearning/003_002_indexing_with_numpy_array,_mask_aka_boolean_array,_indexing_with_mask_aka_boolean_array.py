# 003_002_indexing_with_numpy_array,_mask_aka_boolean_array,_indexing_with_mask_aka_boolean_array.py

# ======================================================================
# You can use boolean indexing and boolean array (=mask)

# c names: names np array
names=np.array(
    ["Charles","Young","Alice","Young","Alice","Young","Young"])
names
# array(['Charles', 'Young', 'Alice', 'Young', 'Alice', 'Young', 'Young'], dtype='<U7')

# c data: (7,4) 2D array
data=np.random.randn(7,4)
data
# array([[-0.52135367,  1.2418411 ,  0.38879725, -0.21458105],
#        [ 0.74671105, -0.1206586 ,  0.57081313,  0.83424216],
#        [ 1.81281524,  0.67417278, -0.28299278, -0.07114104],
#        [-0.04997379, -0.79321175, -1.61146587, -0.08362419],
#        [ 1.0932172 , -0.14965469,  0.23705274,  0.09749227],
#        [-1.32798821,  1.63438081,  0.9191975 , -0.65666315],
#        [ 1.98234468,  0.61819292, -0.63243926, -1.34409189]])

# c names: created mask (boolean array) of names np array, 
# whose all elements represent "element is Young or not"
names=="Young"
# array([False,  True, False,  True, False,  True,  True])

# In this array, ["Charles","Young","Alice","Young","Alice","Young","Young"]
# Charles is not Young, so it's False
# Young is Young, so it's True
# Alice is not Young, so it's False
# ...

# ======================================================================
# You can perform "boolean indexing", or "indexing by mask (boolean array)"
mask_array=names=="Young"
# array([False,  True, False,  True, False,  True,  True])

# You can select row by using mask [False,  True, False,  True, False,  True,  True],
data[mask_array]
data[mask_array,:]
# array([[ 0.74671105, -0.1206586 ,  0.57081313,  0.83424216],
#        [-0.04997379, -0.79321175, -1.61146587, -0.08362419],
#        [-1.32798821,  1.63438081,  0.9191975 , -0.65666315],
#        [ 1.98234468,  0.61819292, -0.63243926, -1.34409189]])
