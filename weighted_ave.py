#!/usr/bin/env python3
"""
Functions to compute weighted means/medians, which aren't automatically available in python
"""

import datetime
from itertools import product
import numpy as np
import pandas as pd

def weighted_mean(df, valname, weightname):
    if weightname not in df:
        raise ValueError('weightname not in df: ' + str(weightname) + '.')

    df2 = df.copy()
    # drop weightnames where valname is missing
    # df2[weightname] = df2[weightname]*pd.notnull(df2[valname])
    df2.loc[df2[valname].isna(), weightname] = np.nan

    d = df2[valname]
    w = df2[weightname]
    if w.sum() == 0:
        return(np.nan)
    else:
        return((d * w).sum() / w.sum())


def weighted_quantile(df, valname, weightname, quantile):
    """
    Quantile should be between 0 and 1
    For median, quantile would be 0.5
    """
    if weightname not in df:
        raise ValueError('weightname not in df: ' + str(weightname) + '.')

    df2 = df.copy()
    # drop weightnames where valname is missing
    # df2[weightname] = df2[weightname]*pd.notnull(df2[valname])
    df2.loc[df2[valname].isna(), weightname] = np.nan

    df2 = df2.sort_values(valname)
    cumsum = df2[weightname].cumsum()
    cutoff = df2[weightname].sum() * quantile

    # case where zero weightnames possible
    # return nan in this case
    if cutoff == 0:
        return(np.nan)
    else:
        return(df2[cumsum > cutoff - 1e-8][valname].iloc[0])


def weighted_median(df, valname, weightname):
    return(weighted_quantile(df, valname, weightname, 0.5))


def weighted_ave_test():
    # without nan
    df = pd.DataFrame({'val': [2, 0, 1], 'weight': [0.3, 0.2, 0]})

    if weighted_mean(df, 'val', 'weight') != 1.2:
        raise ValueError('Weighted mean failed.')
    if weighted_median(df, 'val', 'weight') != 2:
        raise ValueError('Weighted median failed.')

    # with nan
    df = pd.DataFrame({'val': [2, np.nan, 1], 'weight': [0.6, np.nan, 0]})

    if weighted_mean(df, 'val', 'weight') != 2:
        raise ValueError('Weighted mean (nan) failed.')
    if weighted_median(df, 'val', 'weight') != 2:
        raise ValueError('Weighted median (nan) failed.')


def groupby_weighted_mean(df, group_col, data_col, weight_col):
    """
    Gets a groupby weighted mean

    data_col is a list of the columns over which to compute the means. It can also be a single string if there is only one column.
    weight_col is a list of the columns for the weights. Alternatively, a string can be specified in which case only one weight column can be used for all the weights.
    group_col is a string or list for the groups i.e. 'year' or ['year'] or ['year', 'eli']
    """
    # get groups
    g = df.groupby(group_col)

    if isinstance(data_col, str):
        # ensure data_col is a list
        data_col = [data_col]
    if isinstance(weight_col, str):
        # if weight_col is not a list then set it to be a list of same length as data_col
        weight_col = [weight_col] * len(data_col)

    df2list = []

    for i in range(len(data_col)):
        df['_data_times_weight'] = df[data_col[i]]*df[weight_col[i]]
        # important to drop this otherwise sum of weighted mean will be biased down if missing values in data column
        df['_weight_where_notnull'] = df[weight_col[i]]*pd.notnull(df[data_col[i]])
        result = g['_data_times_weight'].sum() / g['_weight_where_notnull'].sum()
        df2 = result.to_frame()
        df2.columns = [data_col[i]]
        df2list.append(df2)

    del df['_data_times_weight'], df['_weight_where_notnull']

    dfout = pd.concat(df2list, axis = 1)

    return(dfout)


def groupby_weighted_mean2(df, group_col, data_col, weight_col):
    """
    Much much slower than groupby_weighted_mean

    Gets a groupby weighted mean

    data_col is a list of the columns over which to compute the means. It can also be a single string if there is only one column.
    weight_col is a list of the columns for the weights. Alternatively, a string can be specified in which case only one weight column can be used for all the weights.
    group_col is a string or list for the groups i.e. 'year' or ['year'] or ['year', 'eli']
    """
    # get groups
    g = df.groupby(group_col)

    if isinstance(data_col, str):
        # ensure data_col is a list
        data_col = [data_col]
    if isinstance(weight_col, str):
        # if weight_col is not a list then set it to be a list of same length as data_col
        weight_col = [weight_col] * len(data_col)

    df2list = []

    for i in range(len(data_col)):
        weighted_mean_this = lambda dfthis: weighted_mean(dfthis, data_col[i], weight_col[i])
        result = g.apply(weighted_mean_this)

        df2 = result.to_frame()
        df2.columns = [data_col[i]]
        df2list.append(df2)

    dfout = pd.concat(df2list, axis = 1)

    return(dfout)


def groupby_weighted_quantile(df, group_col, data_col, weight_col, quantile):
    """
    Gets a groupby weighted quantile

    data_col is a list of the columns over which to compute the means. It can also be a single string if there is only one column.
    weight_col is a list of the columns for the weights. Alternatively, a string can be specified in which case only one weight column can be used for all the weights.
    group_col is a string or list for the groups i.e. 'year' or ['year'] or ['year', 'eli']
    """

    if isinstance(data_col, str):
        # ensure data_col is a list
        data_col = [data_col]
    if isinstance(weight_col, str):
        # if weight_col is not a list then set it to be a list of same length as data_col
        weight_col = [weight_col] * len(data_col)
    if isinstance(group_col, str):
        # ensure data_col is a list
        group_col = [group_col]

    df2list = []

    for i in range(len(data_col)):

        # sort
        dfsort = df.sort_values(data_col[i])
        # important to drop this otherwise sum of weighted mean will be biased down if missing values in data column
        dfsort['_weight_where_notnull'] = dfsort[weight_col[i]]*pd.notnull(dfsort[data_col[i]])

        # get groups
        g = dfsort.groupby(group_col)

        # divide by sum of group
        # dfsort['_sum_weight_where_notnull'] = dfsort.groupby(group_col)['_weight_where_notnull'].agg('sum')
        dfsort['_sum_weight_where_notnull'] = dfsort.groupby(group_col)['_weight_where_notnull'].transform('sum')
        dfsort['_adjusted_weight'] = dfsort['_weight_where_notnull'] / dfsort['_sum_weight_where_notnull']

        # get cumsum by group
        dfsort['_cumsum'] = g['_adjusted_weight'].cumsum()

        # keep values > 0.5
        dfsort = dfsort[dfsort['_cumsum'] > quantile - 1e-8]

        result = dfsort.groupby(group_col)[data_col[i]].first()

        df2 = result.to_frame()
        df2.columns = [data_col[i]]
        df2list.append(df2)

    # get empty dataframe containing all indexes (since some may have no data and be missing)
    dfindex = df = df[group_col].drop_duplicates().set_index(group_col)

    dfout = pd.concat([dfindex] + df2list, axis = 1)

    return(dfout)


def groupby_weighted_quantile2(df, group_col, data_col, weight_col, quantile):
    """
    Much, much slower than groupby_weighted_quantile

    Gets a groupby weighted quantile

    data_col is a list of the columns over which to compute the means. It can also be a single string if there is only one column.
    weight_col is a list of the columns for the weights. Alternatively, a string can be specified in which case only one weight column can be used for all the weights.
    group_col is a string or list for the groups i.e. 'year' or ['year'] or ['year', 'eli']
    """
    # get groups
    g = df.groupby(group_col)

    if isinstance(data_col, str):
        # ensure data_col is a list
        data_col = [data_col]
    if isinstance(weight_col, str):
        # if weight_col is not a list then set it to be a list of same length as data_col
        weight_col = [weight_col] * len(data_col)

    df2list = []

    for i in range(len(data_col)):
        weighted_quantile_this = lambda dfthis: weighted_quantile(dfthis, data_col[i], weight_col[i], quantile)
        result = g.apply(weighted_quantile_this)

        df2 = result.to_frame()
        df2.columns = [data_col[i]]
        df2list.append(df2)

    dfout = pd.concat(df2list, axis = 1)

    return(dfout)


def groupby_weighted_median(df, group_col, data_col, weight_col):
    return(groupby_weighted_quantile(df, group_col, data_col, weight_col, 0.5))


def groupby_weighted_median2(df, group_col, data_col, weight_col):
    return(groupby_weighted_quantile2(df, group_col, data_col, weight_col, 0.5))


def groupby_weighted_ave_test():
    df = pd.DataFrame({'group': ['a', 'a', 'a'], 'val': [2, 0, 1], 'weight': [0.3, 0.2, 0]})
    
    if groupby_weighted_mean(df, 'group', 'val', 'weight')['val'].to_list() != [1.2]:
        raise ValueError('Groupby weighted mean failed.')
    print(groupby_weighted_median(df, 'group', 'val', 'weight'))
    if groupby_weighted_median(df, 'group', 'val', 'weight')['val'].to_list() != [2]:
        raise ValueError('Groupby weighted median failed.')

    df = pd.DataFrame({'group': ['a', 'a', 'b', 'b', 'c'], 'val': [2, 0, 1, np.nan, 3], 'weight': [0.3, 0.2, 1, 0, 0]})
    
    groupby_weighted_mean_res = groupby_weighted_mean(df, 'group', 'val', 'weight')['val'].to_list()
    # need to replace nan with 999 before compare as nan does not equal nan in list
    groupby_weighted_mean_res = [999 if pd.isnull(val) else val for val in groupby_weighted_mean_res]
    if groupby_weighted_mean_res != [1.2, 1.0, 999]:
        raise ValueError('Groupby weighted mean failed.')

    groupby_weighted_median_res = groupby_weighted_median(df, 'group', 'val', 'weight')['val'].to_list()
    # need to replace nan with 999 before compare as nan does not equal nan in list
    groupby_weighted_median_res = [999 if pd.isnull(val) else val for val in groupby_weighted_median_res]
    if groupby_weighted_median_res != [2.0, 1.0, 999]:
        raise ValueError('Groupby weighted median failed.')

    # generate a large groupby example
    np.random.seed(0)
    index0 = list(range(100))
    index1 = list(range(100))
    df = pd.DataFrame(list(product(index0, index1)), columns = ['index0', 'index1'])
    numcols = 100
    data_cols = ['col' + str(i) for i in range(numcols)]
    df2 = pd.DataFrame(np.random.normal(size = (len(df), numcols)), columns = data_cols)
    df = pd.concat([df, df2], axis = 1)
    df['weight'] = 1

    # COMPUTE MEANS

    start = datetime.datetime.now()
    df2 = groupby_weighted_mean(df, 'index0', data_cols, 'weight')
    print(df)
    print(df2)
    print("Main mean method took: " + str((datetime.datetime.now() - start)) + ".")

    start = datetime.datetime.now()
    df2 = groupby_weighted_mean2(df, 'index0', data_cols, 'weight')
    print(df2)
    print("Alternative mean method took: " + str((datetime.datetime.now() - start)) + ".")

    # COMPUTE MEDIANS

    start = datetime.datetime.now()
    df2 = groupby_weighted_median(df, 'index0', data_cols, 'weight')
    print(df)
    print(df2)
    print("Main median method took: " + str((datetime.datetime.now() - start)) + ".")

    start = datetime.datetime.now()
    df2 = groupby_weighted_median2(df, 'index0', data_cols, 'weight')
    print(df2)
    print("Alternative median method took: " + str((datetime.datetime.now() - start)) + ".")

    

# Full:{{{1
def full_test():
    weighted_ave_test()
    groupby_weighted_ave_test()


# Run:{{{1
if __name__ == "__main__":
    full_test()
