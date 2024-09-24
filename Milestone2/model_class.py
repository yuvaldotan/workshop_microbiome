import sys
import pandas as pd
import numpy as np
import math
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.spatial.distance import braycurtis
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, r'C:\Users\tomer\Desktop\BSc\year3\sem B\workshop_microbiome\code')
# sys.path.insert(0, r'C:\Users\yuvald\Documents\Uni\סמסטר ב\workshop_microbiome\code')

from imports import *


class BaboonModel:
    def __init__(self, baboon_id, data, metadata):
        self.baboon_id = baboon_id
        self.alpha_ = np.zeros((61,61))
        self.beta_ = np.zeros((61,61))
        self.data = data
        self.metadata = metadata
        self.normalized_data = clr_normalization(self.data)
        self.df_cumulative_mean = self.normalized_data.expanding().mean()
        self.delta_t = self.metadata['collection_date'].diff()
    
    def fit(self, lambda_):
        # calculate optimised alpha for a given lambda
        def objective(alpha_beta, lambda_):
            # calculate the objective function

            '''for the ith row in self.data (from row 3)
            1. calculate mean of previous i-2 rows = d_mean
            2. calculate time difference between ith row and i-1th row = d_time
            calculate the prediction for the ith row using the formula'''


            '''
            calculate difference between prediction and actual value using bray-curtis dissimilarity and return the sum/mean -TBD'''

            alpha = alpha_beta[:61*61].reshape(61,-1)
            beta = alpha_beta[61*61:].reshape(61,-1)


            D_t1 = self.normalized_data[1:-1].values
            D_mean = self.df_cumulative_mean[:-2].values
            cos = np.cos((2*np.pi*self.delta_t[2:].values)/365)
            exp = np.exp(-lambda_*self.delta_t[2:].values)

            f = alpha@(exp*cos*D_t1.T) + beta@((1-exp*cos)*D_mean.T)

            f = softmax(f.T, axis = 1) # transpose f to match the shape of D - each row is a sample

            # calculate bray-curtis dissimilarity
            bc =  np.array([braycurtis(self.data.values[i+2], f[i]) for i in range(len(f))])

            return bc.mean()

        
        # optimise alpha using scipy.optimize.minimize
        
        optimezed_alpha = minimize(lambda a: objective(a,lambda_), x0 = [0]*(61*61*2), method="L-BFGS-B", bounds=[(-1,1)]*(61*61*2), tol = 1e-3)

        self.alpha_ = optimezed_alpha.x[:61*61].reshape(61,-1)
        self.beta_ = optimezed_alpha.x[61*61:].reshape(61,-1)


        return self.alpha_, optimezed_alpha.fun       


    def predict(self, other, lambda_):
        known_data = other.data
        known_metadata = other.metadata
        return non_iterative_predictor(known_data, known_metadata, self.alpha_, self.beta_, lambda_)


class superModel:
    def __init__(self, data_path, metadata_path):
        data_df = pd.read_csv(data_path, index_col="sample")
        metadata_df = pd.read_csv(metadata_path,  index_col="sample")
        metadata_df["collection_date"] = (pd.to_datetime(metadata_df['collection_date']) - pd.Timestamp('1970-01-01')).dt.days
        # create baboon models
        self.baboons = []
        for baboon_id in metadata_df["baboon_id"].unique():
            baboon_metadata = metadata_df[metadata_df["baboon_id"] == baboon_id].sort_values("collection_date")
            baboon_data = data_df.loc[baboon_metadata.index]
            baboon = BaboonModel(baboon_id, baboon_data, baboon_metadata)
            self.baboons.append(baboon)
        self.lambda_ = 0
        

    def fit(self):
        def objective(lambda_):
            # calculate the objective function
            lambda_ = lambda_[0]
            sum = 0
            futures = []
            cpus = max(1, min(multiprocessing.cpu_count() - 2, len(self.baboons)))

            with ProcessPoolExecutor(cpus) as executor:
                for baboon in self.baboons:
                    fut = executor.submit(baboon.fit, lambda_)
                    futures.append(fut)
            
            for fut in futures:
                alpha, bc = fut.result()
                sum += bc/len(self.baboons)


            # for baboon in self.baboons:
            #     alpha, bc = baboon.fit(lambda_)
            #     sum += bc
            print(f"for lambda = {lambda_} the objective function is {sum}")
            return sum
        
        # optimise lambda using scipy.optimize.minimize
        
        optimezed_lambda = minimize(lambda l: objective(l), x0 = [0], method="L-BFGS-B", bounds=[(0,1)], tol = 1e-3)

        self.lambda_ = optimezed_lambda.x
        
        return objective(self.lambda_)

    def predict(self,  known_data, known_metadata, lambda_):
        # TODO: calculate weights
        weights = np.array([1/len(self.baboons)]*len(self.baboons))
        predictions = []
        for baboon in self.baboons:
            predictions.append(baboon.predict(known_data, known_metadata, lambda_))
        return predictions


def non_iterative_predictor(known_data, known_metadata, alpha, beta, lambda_):
        # calculate time difference between the last known sample and the unknown samples
        delta_t = known_metadata['collection_date'][len(known_data):] - known_metadata['collection_date'][len(known_data)-1]
        # calculate the prediction for the unknown samples using the formula
        D_t1 = np.repeat(clr_normalization(known_data)[-1].values, len(delta_t)).reshape(-1,len(delta_t))
        D_mean = np.repeat(clr_normalization(known_data)[:-2].mean().values, len(delta_t)).reshape(-1,len(delta_t))

        cos = np.cos((2*np.pi*delta_t)/365)
        exp = np.exp(-lambda_*delta_t)
        f = alpha@(exp*cos*D_t1) + beta@((1-exp*cos)*D_mean)
        
        # TODO: should we return the softmax or the CLR values?
        f = softmax(f.T, axis = 1) # transpose f to match the shape of D - each row is a sample

        return f

if __name__ == "__main__":
    data_path = r"C:\Users\tomer\Desktop\BSc\year3\sem B\workshop_microbiome\train_data.csv"
    metadata_path = r"C:\Users\tomer\Desktop\BSc\year3\sem B\workshop_microbiome\train_metadata.csv"
    model = superModel(data_path, metadata_path)
    model.baboons = [model.baboons[0]]
    model.fit()
    print(model.lambda_)