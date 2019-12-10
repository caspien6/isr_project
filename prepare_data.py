import pickle
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score
from sklearn.decomposition import PCA


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def initialize_string_array(df, relevant_string_list):
    for my_str in relevant_string_list:
        df[my_str] = 0


def filter_strings(mydf, short_int = 1, long_int = 20):
    # Filter too short or too long strings
    # return a df with string-count relevant pairs
    
    df_strings = mydf['strings'].apply(lambda x: [k for k in x if (len(k) >=short_int and len(k) <= long_int)])
    strings_list = []
    for index, value in df_strings.items():
        strings_list.extend(value)
    strings_counted = Counter(strings_list)
    df_strings = pd.DataFrame.from_dict(strings_counted, orient='index', columns = ['count']).reset_index()
    return df_strings

def get_unique_values(mydf, colname):
    su_list = []
    for index, row in mydf.iterrows():
        su_list += row[colname]
    su_list = list(set(su_list))
    su_list.sort()
    return su_list

def delete_rare_strings_from_stringcount(mydf, z = 10):
    # The deleted part
    deleted_part = mydf[mydf['count'] <= z]
    print(deleted_part.count())
    # Real deletion part
    df_strings = mydf[mydf['count'] > z]
    print(df_strings.count())
    return df_strings, deleted_part

def fill_up_the_relevance_table(mydf, most_rel_strs, colname='strings'):
    
    def fill_one_row(row):
        for row_str in row[colname]:
            if row_str in most_rel_strs:
                row[row_str] = 1
        return row

    return mydf.apply(fill_one_row, axis=1)


def check_result(data, potential_clusters):
    vector_distances = pairwise_distances(data)
    print("Minimum distance between vectors:", vector_distances.min())
    print( "Mean distance between vectors:", vector_distances.mean())
    print("Maximum distance between vectors:", vector_distances.max())
    print("")
    print("Overall Silhouette Score", silhouette_score(vector_distances, potential_clusters))
    print("")
    print("")
    
def open_data(filename):
    datalist = pickle.load(open(filename, "rb"))
    # Calling DataFrame constructor on list
    return pd.DataFrame(datalist)


def prepare_string_column(df, short_int = 2, long_int = 22, rare_string_count = 15):
    # I need only the strings column now.
    df1 = pd.DataFrame(df['strings'])

    

    df_strings = filter_strings(df1, short_int, long_int)
    df_strings, tmp = delete_rare_strings_from_stringcount(df_strings, rare_string_count)

    # Filter the strings in the strings column
    set_deletable = set(tmp['index'].tolist())
    df1['strings'] = df1['strings'].apply(lambda x: set([k for k in x if k not in set_deletable]))

    # Order the most relevant strings by occurence (descending)
    most_relevant_strings = df_strings.sort_values(by=['count'], ascending = False).reset_index().iloc[:,1:]

    # Create a list from these strings
    most_relevant_strings = most_relevant_strings['index'].tolist()

    initialize_string_array(df1, most_relevant_strings)
    df1 = fill_up_the_relevance_table(df1, most_relevant_strings)
    return df1
