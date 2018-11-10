# 004_001_pandas,_series,_dataframe.py

import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)    # Sets number of maximum columns in terminal
pd.set_option('display.max_columns', None)   # Sets maximum width of each single field in terminal

series=pd.Series([4,7,-5,3])
series
# Automatically generated indices (0 to 3)
# Data (4 to 3)
# 0    4
# 1    7
# 2   -5
# 3    3

# Get only values (without indices) from Series
series.values
# array([ 4,  7, -5,  3])

# Get only indices from Series
series.index
# You will see index information without value
# RangeIndex(start=0, stop=4, step=1)

series.dtype
# dtype('int64')

# ======================================================================
# You can insert your own index instead of default integer index
data=[4,7,-5,3]
custom_index=["d","b","a","c"]
series2=pd.Series(data,index=custom_index)
series2
# d    4
# b    7
# a   -5
# c    3

# ======================================================================
# You can pass dictionary into Series

data_dic={"Charles":35000,"Young":40000,"Alice":30000,"Maria":50000}

series3=pd.Series(data_dic)
series3
# Alice      30000
# Charles    35000
# Maria      50000
# Young      40000

# ======================================================================
# You can give name to Series
series3.name="Salary"
series3
# Charles    35000
# Young      40000
# Alice      30000
# Maria      50000
# Name: Salary, dtype: int64

# ======================================================================
# You can give name to index
series3.index.name="names"
series3
# names
# Alice      30000
# Charles    35000
# Maria      50000
# Young      40000
# Name: Salary, dtype: int64

# ======================================================================
# You can redefine name of index
series3.index=["A","B","C","D"]
series3
# A    30000
# B    35000
# C    50000
# D    40000
# Name: Salary, dtype: int64

# ======================================================================
dic_data_for_dataframe={"names":["Flash","Flash","Flash","Alice","Alice"],
                        "year":[2014,2015,2016,2015,2016],
                        "points":[1.5,1.7,3.6,2.4,2.9]}
df=pd.DataFrame(dic_data_for_dataframe)
df
#      names    points    year
# 0    Flash    1.5       2014
# 1    Flash    1.7       2015
# 2    Flash    3.6       2016
# 3    Alice    2.4       2015
# 4    Alice    2.9       2016

# ======================================================================
# DataFrame is composed of index (height) and column (width)

df.index
# RangeIndex(start=0, stop=5, step=1)

df.columns
# Index(['names', 'points', 'year'], dtype='object')

# We will see only values in 2D array
df.values
# array([['Flash', 1.5, 2014],
#        ['Flash', 1.7, 2015],
#        ['Flash', 3.6, 2016],
#        ['Alice', 2.4, 2015],
#        ['Alice', 2.9, 2016]], dtype=object)

# You change index name of dataframe
df.index.name="Numer"
df
#        names  year  points
# Numer
# 0      Flash  2014  1.5
# 1      Flash  2015  1.7
# 2      Flash  2016  3.6
# 3      Alice  2015  2.4
# 4      Alice  2016  2.9

# You change column name of dataframe
df.columns.name="information"
df
# information  names  year  points
# Numer
# 0            Flash  2014  1.5
# 1            Flash  2015  1.7
# 2            Flash  2016  3.6
# 3            Alice  2015  2.4
# 4            Alice  2016  2.9

# ======================================================================
# You can give column name and index name,
# when you create dataframe

column_data=["year","name","points","penalty"]
index_data=["one","two","three","four","five"]

dataframe2=pd.DataFrame(
    dic_data_for_dataframe,
    columns=column_data,
    index=index_data)
dataframe2
#        year name  points penalty
# one    2014  NaN  1.5     NaN
# two    2015  NaN  1.7     NaN
# three  2016  NaN  3.6     NaN
# four   2015  NaN  2.4     NaN
# five   2016  NaN  2.9     NaN

# ======================================================================
# You can see brief statistics of dataframe
dataframe2.describe()
#        year          points
# count  5.00000       5.000000
# mean   2015.20000    2.420000
# std    0.83666       0.864292
# min    2014.00000    1.500000
# 25%    2015.00000    1.700000
# 50%    2015.00000    2.400000
# 75%    2016.00000    2.900000
# max    2016.00000    3.600000
