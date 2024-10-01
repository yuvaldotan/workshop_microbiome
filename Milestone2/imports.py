import pandas as pd
import numpy as np
from scipy.special import softmax

def transformation(df, type = 'counts'):
    
    return {'clr':clr_transformation(df), 'counts':counts_transformation(df)}[type]

def clr_transformation(df):
    dataf = df.copy()
    D = dataf.shape[1]
    values = dataf.values
    values[values == 0] = 10 ** - 100
    
    for i, row in enumerate(values):
        gp = np.sum(np.log(row**(1/D)))
        values[i] = np.log(row)-gp


    
    return pd.DataFrame(values, columns= dataf.columns, index=dataf.index)

def counts_transformation(df):
    counts = np.random.choice(pd.read_csv(r"count.csv", index_col="library_name")["read_count"].values, len(df.index), replace=True)
    dataf = df.copy()
    dataf =dataf.multiply(counts, axis=0)

    return dataf

def to_composition(arr, type = 'counts'):
    if type == 'clr':
        return softmax(arr)
    if type == 'counts':
        return arr/arr.sum(axis=1).reshape(-1,1)