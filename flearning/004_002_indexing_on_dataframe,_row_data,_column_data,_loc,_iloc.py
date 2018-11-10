# 004_002_indexing_on_dataframe,_row_data,_column_data,_loc,_iloc.py

import pandas as pd
import numpy as np

dic_data_for_dataframe={"names":["Flash","Flash","Flash","Alice","Alice"],
                        "year":[2014,2015,2016,2015,2016],
                        "points":[1.5,1.7,3.6,2.4,2.9]}
columns_data=["year","names","points","penalty"]
index_data=["one","two","three","four","five"]
dataframe1=pd.DataFrame(
    dic_data_for_dataframe,
    columns=columns_data,
    index=index_data)
dataframe1
#        year    names    points penalty
# one    2014    Flash    1.5    NaN
# two    2015    Flash    1.7    NaN
# three  2016    Flash    3.6    NaN
# four   2015    Alice    2.4    NaN
# five   2016    Alice    2.9    NaN

# ======================================================================
dataframe1["year"]
# one      2014
# two      2015
# three    2016
# four     2015
# five     2016

# --------------------------------------------------
# You can obtain same result with following code
dataframe1.year
# one      2014
# two      2015
# three    2016
# four     2015
# five     2016

# ======================================================================
dataframe1[["year","points"]]
#        year    points
# one    2014    1.5
# two    2015    1.7
# three  2016    3.6
# four   2015    2.4
# five   2016    2.9

# ======================================================================
dataframe1["penalty"]=0.5
dataframe1
#        year    names    points penalty
# one    2014    Flash    1.5    0.5
# two    2015    Flash    1.7    0.5
# three  2016    Flash    3.6    0.5
# four   2015    Alice    2.4    0.5
# five   2016    Alice    2.9    0.5

# --------------------------------------------------
dataframe1["penalty"]=[0.1,0.2,0.3,0.4,0.5]
dataframe1
#        year    names    points penalty
# one    2014    Flash    1.5    0.1
# two    2015    Flash    1.7    0.2
# three  2016    Flash    3.6    0.3
# four   2015    Alice    2.4    0.4
# five   2016    Alice    2.9    0.5

# --------------------------------------------------
dataframe1["penalty"]=np.arange(5)
dataframe1
#        year    names    points penalty
# one    2014    Flash    1.5    0
# two    2015    Flash    1.7    1
# three  2016    Flash    3.6    2
# four   2015    Alice    2.4    3
# five   2016    Alice    2.9    4

# ======================================================================
# You can use series, when you insert value into dataframe

data=[-1.2,-1.5,-1.7]
series_data_for_dataframe=pd.Series(
    data,
    index=["two","four","five"])
dataframe1["debt"]=series_data_for_dataframe
dataframe1
#        year    names    points penalty  debt
# one    2014    Flash    1.5    0        NaN
# two    2015    Flash    1.7    1        -1.2
# three  2016    Flash    3.6    2        NaN
# four   2015    Alice    2.4    3        -1.5
# five   2016    Alice    2.9    4        -1.7

# ======================================================================
# Calculation

df1_po=dataframe1["points"]
df1_po
# one      1.5
# two      1.7
# three    3.6
# four     2.4
# five     2.9
# Name: points, dtype: float64

df1_pe=dataframe1["penalty"]
df1_pe
# one      0
# two      1
# three    2
# four     3
# five     4

dataframe1["net_points"]=df1_po-df1_pe
dataframe1
#        year  names  points  penalty  debt  net_points
# one    2014  Flash  1.5     0       NaN    1.5
# two    2015  Flash  1.7     1       -1.2   0.7
# three  2016  Flash  3.6     2       NaN    1.6
# four   2015  Alice  2.4     3       -1.5  -0.6
# five   2016  Alice  2.9     4       -1.7  -1.1

# ======================================================================
df1_np_gt_2=dataframe1["net_points"]>2.0
# one      False
# two      False
# three    False
# four     False
# five     False
# Name: net_points, dtype: bool

dataframe1["high_points"]=df1_np_gt_2
dataframe1
#       year    names   points  penalty  debt   net_points  high_points
# one   2014    Flash   1.5     0        NaN    1.5         False
# two   2015    Flash   1.7     1        -1.2   0.7         False
# three 2016    Flash   3.6     2        NaN    1.6         False
# four  2015    Alice   2.4     3        -1.5   -0.6        False
# five  2016    Alice   2.9     4        -1.7   -1.1        False

# ======================================================================
del dataframe1["high_points"]
del dataframe1["net_points"]
dataframe1
#       year    names    points  penalty   debt
# one   2014    Flash    1.5     0         NaN
# two   2015    Flash    1.7     1         -1.2
# three 2016    Flash    3.6     2         NaN
# four  2015    Alice    2.4     3         -1.5
# five  2016    Alice    2.9     4         -1.7

# ======================================================================
dataframe1.columns
# Index(['year', 'names', 'points', 'penalty', 'debt'], dtype='object')

# ======================================================================
dataframe1.index.name="Order"
dataframe1.columns.name="Info"
dataframe1
# Information   year    names    points   penalty   debt
# Order                    
# one           2014    Flash    1.5      0         NaN
# two           2015    Flash    1.7      1         -1.2
# three         2016    Flash    3.6      2         NaN
# four          2015    Alice    2.4      3         -1.5
# five          2016    Alice    2.9      4         -1.7

# ======================================================================
# Unlike numpy, there are much more ways of indexing in Pandas

dataframe1
# Info   year  names  points  penalty  debt
# Order
# one    2014  Flash  1.5     0       NaN
# two    2015  Flash  1.7     1       -1.2
# three  2016  Flash  3.6     2       NaN
# four   2015  Alice  2.4     3       -1.5
# five   2016  Alice  2.9     4       -1.7

dataframe1[0:3]
# Info   year  names  points  penalty  debt
# Order
# one    2014  Flash  1.5     0       NaN
# two    2015  Flash  1.7     1       -1.2
# three  2016  Flash  3.6     2       NaN

dataframe1["two":"four"]
# Info   year  names  points  penalty  debt
# Order
# two    2015  Flash  1.7     1       -1.2
# three  2016  Flash  3.6     2       NaN
# four   2015  Alice  2.4     3       -1.5

# ======================================================================
# Since there are too much ways of indexing, you can be confused by numerous ways
# So, using following 2 ways are recommended when indexing dataframe
# loc[], iloc[]

# I personally prefer iloc[] because it's similar to numpy indexing

# ======================================================================
dataframe1
# Info   year  names  points  penalty  debt
# Order
# one    2014  Flash  1.5     0       NaN
# two    2015  Flash  1.7     1       -1.2
# three  2016  Flash  3.6     2       NaN
# four   2015  Alice  2.4     3       -1.5
# five   2016  Alice  2.9     4       -1.7

# --------------------------------------------------
dataframe1.loc["two"]
# Information
# year        2015
# names      Flash
# points       1.7
# penalty        1
# debt        -1.2

# --------------------------------------------------
dataframe1.loc["two":"four"]
# Information  year    names    points penalty  debt
# Order                  
# two          2015    Flash    1.7    1        -1.2
# three        2016    Flash    3.6    2        NaN
# four         2015    Alice    2.4    3        -1.5

# --------------------------------------------------
dataframe1.loc["two":"four","points"]

# --------------------------------------------------
dataframe1.loc[:,"year"]

# --------------------------------------------------
dataframe1.loc[:,["year","names"]]

# --------------------------------------------------
dataframe1.loc[:,"year":"penalty"]

# --------------------------------------------------
dataframe1.loc[["three","five"],["year","names"]]

# --------------------------------------------------
dataframe1.loc["six",:]=[2013,"Alice",3.0,0.1,2.1]
# Information  year    names   points   penalty   debt
# Order                    
# one          2014.0  Flash   1.5      0.0       NaN
# two          2015.0  Flash   1.7      1.0       -1.2
# three        2016.0  Flash   3.6      2.0       NaN
# four         2015.0  Alice   2.4      3.0       -1.5
# five         2016.0  Alice   2.9      4.0       -1.7
# six          2013.0  Alice   3.0      0.1       2.1

# ======================================================================
# Let's talk about iloc[]
# iloc[] is used when you bring row and column,
# by numpy array indexing way

dataframe1
# Info     year  names  points  penalty  debt
# Order
# one    2014.0  Flash  1.5     0.0     NaN
# two    2015.0  Flash  1.7     1.0     -1.2
# three  2016.0  Flash  3.6     2.0     NaN
# four   2015.0  Alice  2.4     3.0     -1.5
# five   2016.0  Alice  2.9     4.0     -1.7
# six    2013.0  Alice  3.0     0.1      2.1

dataframe1.iloc[3]
# Information
# year         2015
# names       Alice
# points        2.4
# penalty         3
# debt         -1.5

dataframe1.iloc[3:5,0:2]
# Information	year	names
# Order		
# four	        2015.0	Alice
# five	        2016.0	Alice

dataframe1.iloc[[0,1,3],[1,2]]
# Information  names  points
# Order
# one          Flash  1.5
# two          Flash  1.7
# four         Alice  2.4

dataframe1[:,1:4]

dataframe1[1,1]

# ======================================================================
dataframe1
# Info     year  names  points  penalty  debt
# Order
# one    2014.0  Flash  1.5     0.0     NaN
# two    2015.0  Flash  1.7     1.0     -1.2
# three  2016.0  Flash  3.6     2.0     NaN
# four   2015.0  Alice  2.4     3.0     -1.5
# five   2016.0  Alice  2.9     4.0     -1.7
# six    2013.0  Alice  3.0     0.1      2.1

df1_y_gt_2014=dataframe1["year"]>2014
# Order
# one      False
# two       True
# three     True
# four      True
# five      True
# six      False
# Name: year, dtype: bool


dataframe1.loc[df1_y_gt_2014,:]
# Information year      names    points penalty debt
# Order                    
# two         2015.0    Flash    1.7    1.0    -1.2
# three       2016.0    Flash    3.6    2.0    NaN
# four        2015.0    Alice    2.4    3.0    -1.5
# five        2016.0    Alice    2.9    4.0    -1.7

# --------------------------------------------------
df1_n_eq_Alice=dataframe1["names"]=="Alice"
# Order
# one      False
# two      False
# three    False
# four     True
# five     True
# six      True
# Name: names, dtype: bool

dataframe1.loc[df1_n_eq_Alice,["names","points"]]
# Information  names    points
# Order        
# four         Alice    2.4
# five         Alice    2.9
# six          Alice    3.0

# --------------------------------------------------
# dataframe1
# Info     year  names  points  penalty  debt
# Order
# one    2014.0  Flash  1.5     0.0     NaN
# two    2015.0  Flash  1.7     1.0     -1.2
# three  2016.0  Flash  3.6     2.0     NaN
# four   2015.0  Alice  2.4     3.0     -1.5
# five   2016.0  Alice  2.9     4.0     -1.7
# six    2013.0  Alice  3.0     0.1      2.1

df1_po_gt_2=dataframe1["points"]>2
# Order
# one      False
# two      False
# three    True
# four     True
# five     True
# six      True
# Name: points, dtype: bool

dt1_po_lt_3=dataframe1["points"]<3
# Order
# one      True
# two      True
# three    False
# four     True
# five     True
# six      False
# Name: points, dtype: bool

merged_condi=df1_po_gt_2&dt1_po_lt_3
# Order
# one      False
# two      False
# three    False
# four     True
# five     True
# six      False
# Name: points, dtype: bool

dataframe1.loc[merged_condi,:]
# Information  year    names  points penalty debt
# Order                   
# four         2015.0  Alice  2.4    3.0     -1.5
# five         2016.0  Alice  2.9    4.0     -1.7

# --------------------------------------------------
# dataframe1
# Info     year  names  points  penalty  debt
# Order
# one    2014.0  Flash  1.5     0.0     NaN
# two    2015.0  Flash  1.7     1.0     -1.2
# three  2016.0  Flash  3.6     2.0     NaN
# four   2015.0  Alice  2.4     3.0     -1.5
# five   2016.0  Alice  2.9     4.0     -1.7
# six    2013.0  Alice  3.0     0.1      2.1

df1_po_gt_3=dataframe1["points"]>3
# Order
# one      False
# two      False
# three    True
# four     False
# five     False
# six      False
# Name: points, dtype: bool

selected=dataframe1.loc[df1_po_gt_3,"penalty"]
selected=0
dataframe1
# Information  year    names   points penalty  debt
# Order                    
# one          2014.0  Flash   1.5    0.0      NaN
# two          2015.0  Flash   1.7    1.0      -1.2
# three        2016.0  Flash   3.6    0.0      NaN
# four         2015.0  Alice   2.4    3.0      -1.5
# five         2016.0  Alice   2.9    4.0      -1.7
# six          2013.0  Alice   3.0    0.1      2.1
