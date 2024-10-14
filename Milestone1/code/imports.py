import pandas as pd
import numpy as np
from scipy.special import softmax

def transformation(df, type = 'counts'):
    
    return {'clr':clr_transformation, 'counts':counts_transformation}[type](df)

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
        if arr.ndim == 1:
            total_sum = arr.sum()
            if total_sum == 0:  # Avoid division by zero
                return np.zeros_like(arr)
            return arr / total_sum
        
        row_sums = arr.sum(axis=1).reshape(-1, 1)
        row_sums[row_sums == 0] = 1  # Replace zero sums with 1 to avoid division by zero
        return arr / row_sums