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

delta_t_social_group = 10
delta_t_other = 7

class BaboonModel:
    def __init__(self, baboon_id, data, metadata):
        self.baboon_id = baboon_id
        self.beta_ = np.zeros(61)
        self.metadata_I = metadata[metadata["baboon_id"]==baboon_id]
        self.data_I = data.loc[self.metadata_I.index]
        self.mean_social = self.data_I.copy()
        self.mean_other = self.data_I.copy()
        self.df_cumulative_mean = self.data_I.expanding().mean()
       
        self.delta_t = self.metadata_I['collection_date'].diff()
        for sample in self.metadata_I.index:
            social_group = self.metadata_I.loc[sample, 'social_group']
            date = pd.to_datetime(self.metadata_I.loc[sample, 'collection_date'])
            self.mean_social.loc[sample] = data[(metadata['social_group'] == social_group) & (metadata["baboon_id"]!=self.baboon_id) & (abs((pd.to_datetime(metadata['collection_date']) - date).dt.days)<=delta_t_social_group)].mean()
            self.mean_other.loc[sample] = data[(metadata['social_group'] != social_group) & (metadata["baboon_id"]!=self.baboon_id) & (abs((pd.to_datetime(metadata['collection_date']) - date).dt.days)<=delta_t_other)].mean()

        self.mean_social.fillna(0, inplace = True)
        self.mean_other.fillna(0, inplace = True)      

    
    def fit(self, alpha):
        # calculate optimised alpha for a given lambda
        def objective(alpha, beta):
            # calculate the objective function

            '''for the ith row in self.data (from row 3)
            1. calculate mean of previous i-1 rows = d_mean
            2. calculate mean of baboons in the same social_group with time difference of delta_t_social_group days
            3. calculate mean of baboons that are not in the same social_group with time difference of delta_t_other days
            calculate the prediction for the ith row using the formula'''

            '''
            calculate difference between prediction and actual value using bray-curtis dissimilarity and return the mean'''

            D_meanI = self.df_cumulative_mean[:-1].values
            D_meanS = self.mean_social[1:].values
            D_meanO = self.mean_other[1:].values

            f = alpha*D_meanO + (1-alpha-beta)*D_meanS + beta*D_meanI
            f = to_composition(f, type = 'counts') 
            # calculate bray-curtis dissimilarity
            bc =  np.array([braycurtis(self.data_I.values[i+1], f[i]) for i in range(len(f))])

            return bc.mean()

        

        
        optimezed_score = minimize(lambda beta: objective(alpha, beta), x0 = (1-alpha)/2 , method="L-BFGS-B", bounds=[(0,1-a) for a in alpha])

        self.beta_ = optimezed_score.x


        return self.beta_, optimezed_score.fun       


    def predict(self, other_data, other_metadata, weights,lambda_, iterative = False):
        if iterative:
            return iterative_predictor(other_data, other_metadata, self.alpha_, self.beta_, lambda_)*weights
        return (non_iterative_predictor(other_data, other_metadata, self.alpha_, self.beta_, lambda_).T * weights).T


class superModel:
    def __init__(self, data_path, metadata_path):
        metadata_df, data_df = preprocessing(data_path, metadata_path)
        # create baboon models
        self.baboons = []
        cpus = max(1, min(multiprocessing.cpu_count() - 2, len(metadata_df["baboon_id"].unique())))
        futures = []

        with ProcessPoolExecutor(cpus) as executor:
                for baboon_id in metadata_df["baboon_id"].unique():
                    futures.append(executor.submit(BaboonModel, baboon_id, data_df, metadata_df))
        
        for fut in futures:
            baboon = fut.result()                  
            self.baboons.append(baboon)
        self.alpha_ = np.array([0]*61)

    def fit(self):
        def objective(alpha):
            # calculate the objective function
            sum = 0
            futures = [] # process pool executor futures
            cpus = max(1, min(multiprocessing.cpu_count() - 2, len(self.baboons)))

            with ProcessPoolExecutor(cpus) as executor:
                for baboon in self.baboons:
                    fut = executor.submit(baboon.fit, alpha) # execute the fit function of each baboon in parallel using the global alpha
                    futures.append(fut)
            
            for fut in futures:
                beta,  bc = fut.result()
                sum += bc
            
            print(f"for alpha \n{alpha}\nscore is {sum}")
            return sum
        
        # optimise beta using scipy.optimize.minimize
        
        optimezed_alpha = minimize(lambda alpha: objective(alpha), x0 = self.alpha_, method="L-BFGS-B", bounds=[(0,1)])

        self.alpha_ = optimezed_alpha.x
        
        return objective(self.alpha_)

    def predict(self,  known_data, known_metadata, iterative = False):
        weights = np.zeros((len(self.baboons),len(known_metadata[len(known_data):])))
        # weights = np.array([1/len(self.baboons)]*len(self.baboons))

        for i, baboon in enumerate(self.baboons):
            weights[i,:] = baboon_similarity(baboon.metadata, known_metadata[len(known_data):])

        weights = weights.T

        for i in range(len(weights)):
            if weights[i].sum() == 0:
                weights[i] = np.array([1/len(self.baboons)]*len(self.baboons))
            else:
                weights[i] = weights[i]/weights[i].sum()


        predictions = np.zeros((len(known_metadata[len(known_data):]), 61))
        n = len(known_data)
        if not iterative:
            with ProcessPoolExecutor(multiprocessing.cpu_count()) as executor:
                futures = []
                for i, baboon in enumerate(self.baboons):
                    fut = executor.submit(baboon.predict, known_data, known_metadata, weights[:,i], self.lambda_, iterative)
                    futures.append(fut)
                for i, fut in enumerate(futures):
                    predictions += fut.result()
        else:           
            for j in range(len(known_data), len(known_metadata)):
                with ProcessPoolExecutor(multiprocessing.cpu_count()) as executor:
                    futures = []
                    for i, baboon in enumerate(self.baboons):
                        fut = executor.submit(baboon.predict, known_data, known_metadata, weights[j-n,i], self.lambda_, iterative = True)
                        futures.append(fut)
                    for i, fut in enumerate(futures):
                        predictions[j-n] += fut.result()
                known_data = pd.concat([known_data, pd.Series(predictions[j-n], index=known_data.columns).to_frame().T], ignore_index=True)
        pred_df = pd.DataFrame(predictions, columns = known_data.columns, index=known_metadata.index[n:])

        return pred_df

def non_iterative_predictor(known_data, known_metadata, alpha, beta, lambda_):
        # calculate time difference between the last known sample and the unknown samples
                
        delta_t = known_metadata['collection_date'].values[len(known_data):] - known_metadata['collection_date'].values[len(known_data)-1]
        # calculate the prediction for the unknown samples using the formula
        
        D_t1 = np.repeat(transformation(known_data, type = method).values[-1], len(delta_t)).reshape(-1,len(delta_t)).T

        if len(known_data)<=2:
            D_mean = np.repeat(transformation(known_data, type = method).values[0], len(delta_t)).reshape(-1,len(delta_t)).T
        else:
            D_mean = np.repeat(np.mean(transformation(known_data, type = method).values[:-2], axis = 0), len(delta_t)).reshape(-1,len(delta_t)).T

        cos = np.cos((2*np.pi*delta_t)/365.001)
        exp = np.exp(-lambda_*delta_t)
        f = alpha@(exp*cos*D_t1.T) + beta@((1-exp*cos)*D_mean.T)
        
        # TODO: should we return the normalized or the transformes values?
        f = to_composition(f.T, type= method) # transpose f to match the shape of D - each row is a sample

        return f


def iterative_predictor(known_data, known_metadata, alpha, beta, lambda_):
        # calculate time difference between the last known sample and the unknown samples
                
        delta_t = known_metadata['collection_date'].values[len(known_data)] - known_metadata['collection_date'].values[len(known_data)-1]
        # calculate the prediction for the unknown samples using the formula

        D_t1 = transformation(known_data, type = method).values[-1]
        if len(known_data)<=2:
            D_mean = transformation(known_data, type = method).values[0]
        else:
            D_mean = np.mean(transformation(known_data, type = method).values[:-2], axis = 0)

        cos = np.cos((2*np.pi*delta_t)/365.001)
        exp = np.exp(-lambda_*delta_t)
        f = (exp*cos*D_t1)@alpha + ((1-exp*cos)*D_mean.T)@beta
        
        # TODO: should we return the normalized or the transformes values?
        f = to_composition(f.T, type= method) # transpose f to match the shape of D - each row is a sample

        return f


def baboon_similarity(baboon1, baboon2):
    # score based on social group similarity
    # option1: for each sample in baboon2, take baboon1 if it was in baboon2's social group
    # option2: take the relative part of presence in the social group

    social_groups_baboon1 = set(baboon1['social_group'])

    # Create the 0/1 list based on presence of social group in baboon1
    presence_list = [1 if group in social_groups_baboon1 else 0 for group in baboon2['social_group']]

    return presence_list

def preprocessing(data_path, metadata_path):
    """Preprocess the data and metadata files to be used in.
    for samples from the same date and baboon_id, calculate the mean of the bacteria counts.
    """
    data_df = pd.read_csv(data_path)
    metadata_df = pd.read_csv(metadata_path)
    temp = pd.merge(data_df, metadata_df[["sample", "baboon_id", "collection_date"]], on="sample")
    bacteria_columns = temp.columns[1:-2]  # Adjust this depending on your actual column structure

    # Group by baboon_id and collection_date
    data_clean = temp.groupby(['baboon_id', 'collection_date']).agg({**{col: 'mean' for col in bacteria_columns}, 'sample': 'first'}).reset_index()
    data_clean.drop(['baboon_id', 'collection_date'], axis=1, inplace=True)
    chosen_samples = data_clean["sample"].unique()
    metadata_clean = metadata_df[metadata_df["sample"].isin(chosen_samples)]
    metadata_clean.set_index('sample', inplace=True)
    data_clean.set_index('sample', inplace=True)
    metadata_clean["collection_date"] = (pd.to_datetime(metadata_clean['collection_date']) - pd.Timestamp('1970-01-01')).dt.days
    return metadata_clean, data_clean
    


if __name__ == "__main__":
    data_path = r"train_data.csv"
    metadata_path = r"train_metadata.csv"
    model = superModel(data_path, metadata_path)
    # model.baboons = model.baboons[:3]
    # model.fit()
    print(model.lambda_)
    for baboon in model.baboons:
        baboon.alpha_ = np.zeros([61,61])
        baboon.beta_ = np.eye(61,61)
    baboons = model.baboons
    model.baboons = model.baboons[:60]
    Baboon_78 = baboons[-2]
    baboon_78_data = Baboon_78.data[2:]
    Baboon_78.data = Baboon_78.data[:2]
    iterative_res = model.predict(Baboon_78.data, Baboon_78.metadata, iterative=True)
    noninterative_res = model.predict(Baboon_78.data, Baboon_78.metadata, iterative=False)