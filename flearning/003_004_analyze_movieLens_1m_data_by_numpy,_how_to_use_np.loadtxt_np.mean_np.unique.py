# 003_004_analyze_movieLens_1m_data_by_numpy,_how_to_use_np.loadtxt_np.mean_np.unique.py

import numpy as np

# You can download movielens dataset from here
# https://grouplens.org/datasets/movielens/
# ml-20m.zip (size: 190 MB, checksum)

data=np.loadtxt("./data/movielens-1m/ratings.dat",delimiter="::",dtype=np.int64)

data.shape
# (1000209, 4)
# data is (1000209, 4) 2D array

# In 0 axis (1000209 rows), select start (0) row to (5-1) row
# In 1 axis (4 columns), select all columns
data[:5,:]
# array([[1,  1193,  5,  978300760],
#        [1,   661,  3,  978302109],
#        [1,   914,  3,  978301968],
#        [1,  3408,  4,  978300275],
#        [1,  2355,  5,  978824291]])

# ======================================================================
# You can find averge of rating score of entire dataset
rating_score_data=data[:,2]
# array([5, 3, 3, ..., 5, 4, 4])

mean_rating_total=rating_score_data.mean()
mean_rating_total
# 3.581564453029317

# You can find averge of rating score from each user
# You first extract each user
all_ids=data[:,0]
# array([   1,    1,    1, ..., 6040, 6040, 6040])

all_ids.shape
# (1000209,)
# There are 1000209 ids

user_ids=np.unique(all_ids)
# array([   1,    2,    3, ..., 6038, 6039, 6040])
# You can see there are 6040 user ids

# ======================================================================
# c mean_rating_by_user_list: created empty list
mean_rating_by_user_list=[]

# Iterates all unique user ids
for user_id in user_ids:
    all_ids=data[:,0]
    mask=all_ids==user_id
    # c userTorF: created mask, user true or false
    userTorF=mask
    # You perform indexing with mask
    dataForEachUser=data[userTorF,:]
    meanRatingForEachUser=dataForEachUser[:,2].mean()
    mean_rating_by_user_list.append([user_id,meanRatingForEachUser])

mean_rating_by_user_list[:5]
# [[1, 4.188679245283019],
#  [2, 3.7131782945736433],
#  [3, 3.9019607843137254],
#  [4, 4.190476190476191],
#  [5, 3.1464646464646466]]

# ======================================================================
# You will convert list into np array to easily deal with

meanRatingFromEachUserArray=np.array(
    mean_rating_by_user_list,dtype=np.float32)

meanRatingFromEachUserArray[:5]
# array([[1. , 4.188679 ],
#        [2. , 3.7131784],
#        [3. , 3.9019608],
#        [4. , 4.1904764],
#        [5. , 3.1464646]], dtype=float32)

meanRatingFromEachUserArray.shape
# (6040, 2)

# You will export array as csv file
np.savetxt("meanRatingFromEachUserArray.csv",meanRatingFromEachUserArray,fmt="%.3f",delimiter=",")
