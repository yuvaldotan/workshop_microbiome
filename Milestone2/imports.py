import pandas as pd
import numpy as np

def clr_normalization(df):
    dataf = df.copy()
    D = dataf.shape[1]
    values = dataf.values
    values[values == 0] = 10 ** - 100
    
    for i, row in enumerate(values):
        gp = np.sum(np.log(row**(1/D)))
        values[i] = np.log(row)-gp


    
    return pd.DataFrame(values, columns= dataf.columns, index=dataf.index)


