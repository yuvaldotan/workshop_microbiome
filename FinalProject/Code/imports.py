import sys
import pandas as pd
import numpy as np
import math
from scipy.special import softmax
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.distance import braycurtis
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from imports import *
import matplotlib.pyplot as plt
import seaborn as sns
import time

def to_composition(arr, type = 'counts'):
    if type == 'clr':
        return softmax(arr)
    if type == 'counts':
        if arr.ndim == 1:
            total_sum = arr.sum()
            if total_sum == 0:  # Avoid division by zero
                # print(arr)
                return np.zeros_like(arr)
            return arr / total_sum
        
        row_sums = arr.sum(axis=1).reshape(-1, 1)
        row_sums[row_sums == 0] = 1  # Replace zero sums with 1 to avoid division by zero
        return arr / row_sums