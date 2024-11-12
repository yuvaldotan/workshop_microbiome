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
dataless_threshold = 10

# add delta_T
# add gamma
# change mean calculation
class BaboonModel:
    def __init__(self, baboon_id, data, metadata, fit=False, alpha = np.zeros(61), gamma =0):
        # print("init baboon model: ",baboon_id, len(metadata[metadata["baboon_id"]==baboon_id]))
        self.baboon_id = baboon_id
        self.beta_ = np.zeros(61)
        self.metadata_I = metadata[metadata["baboon_id"]==baboon_id].sort_values(by = 'collection_date')
        self.data_I = data.loc[[idx for idx in self.metadata_I.index if idx in data.index]] # add intersection with data index
        self.mean_social = self.data_I.copy()
        self.mean_other = self.data_I.copy()
       
        self.delta_t = self.metadata_I['collection_date'].diff().dt.days # maybe in the future


        for sample in self.metadata_I.index:
            social_group = self.metadata_I.loc[sample, 'social_group']
            date = self.metadata_I.loc[sample, 'collection_date']
            social_indicies = metadata[(metadata['social_group'] == social_group) & (metadata["baboon_id"]!=self.baboon_id) & (abs((metadata['collection_date'] - date).dt.days)<=delta_t_social_group)].index
            other_indicies = metadata[(metadata['social_group'] != social_group) & (metadata["baboon_id"]!=self.baboon_id) & (abs((metadata['collection_date'] - date).dt.days)<=delta_t_other)].index
            self.mean_social.loc[sample] = data.loc[np.intersect1d(data.index, social_indicies)].mean()
            self.mean_other.loc[sample] = data.loc[np.intersect1d(data.index, other_indicies)].mean()

        self.mean_social.fillna(0, inplace = True)
        self.mean_other.fillna(0, inplace = True)
        if fit:
            self.fit(alpha, gamma)  
    

    
    def fit(self, alpha, gamma):
        # calculate optimised alpha for a given lambda

        D_meanI = self.weighted_mean().values
        D_meanS = self.mean_social.loc[self.data_I.index][1:].values
        D_meanO = self.mean_other.loc[self.data_I.index][1:].values

        def objective(alpha, beta, gamma):
            # calculate the objective function

            '''for the ith row in self.data (from row 3)
            1. calculate mean of previous i-1 rows = d_mean
            2. calculate mean of baboons in the same social_group with time difference of delta_t_social_group days
            3. calculate mean of baboons that are not in the same social_group with time difference of delta_t_other days
            calculate the prediction for the ith row using the formula'''

            '''
            calculate difference between prediction and actual value using bray-curtis dissimilarity and return the mean'''

            

            f = alpha*D_meanO + (1-alpha-beta)*D_meanS + beta*D_meanI
            f = to_composition(f, type = 'counts') 
            # calculate bray-curtis dissimilarity
            bc =  np.array([braycurtis(self.data_I.values[i+1], f[i]) for i in range(len(f))])

            return bc.mean()
        
        optimezed_score = minimize(lambda beta: objective(alpha, beta, gamma), x0 = (1-alpha)/2 , method="L-BFGS-B", bounds=[(0,1-a) for a in alpha])

        self.beta_ = optimezed_score.x


        return self.beta_, optimezed_score.fun       


    def predict(self, alpha, iterative = False):
        # print(len(self.data_I), len(self.metadata_I))
        if iterative:
            return self.iterative_predictor(alpha)
        return self.non_iterative_predictor(alpha)
    
    def non_iterative_predictor(self, alpha):
        mask = self.metadata_I[~np.isin(self.metadata_I.index,self.data_I.index)].index # sample ids to predict
        D_I = self.weighted_mean(mask).values
        D_O = self.mean_other.loc[mask].values
        D_S = self.mean_social.loc[mask].values

        f = alpha*D_O + (1-alpha-self.beta_)*D_S + self.beta_*D_I
        
        # TODO: should we return the normalized or the transformes values?
        f = to_composition(f, type= "counts") 
        f_df = pd.DataFrame(f, columns = self.data_I.columns, index = mask)
        return f_df
    
    
    def iterative_predictor(self, alpha):
        mask = self.metadata_I[~np.isin(self.metadata_I.index,self.data_I.index)].index # sample ids to predict
        data = self.data_I.copy()
        for sample in mask:
            D_I = self.weighted_mean_row(sample)
            D_O = self.mean_other.loc[sample].values
            D_S = self.mean_social.loc[sample].values

            f = alpha*D_O + (1-alpha-self.beta_)*D_S + self.beta_*D_I
            # TODO: should we return the normalized or the transformes values?
            f = to_composition(f.T, type= "counts") # transpose f to match the shape of D - each row is a sample
            self.data_I.loc[sample] = f
        res = self.data_I.loc[mask]
        self.data_I = data
        return res
    

    def weighted_mean(self, indexes = None):
        res = pd.DataFrame(columns=self.data_df.columns) 
        if indexes is None:
            indexes = self.metadata_df.index[1:]
        for idx in indexes:
            res.loc[idx] = self.weighted_mean_row(idx)
        return res
    
    def weighted_mean_row(self, idx):

        row = np.zeros(61)

        loc = self.metadata_df.get_loc[idx]
        delta_t_samples = self.delta_t.iloc[:loc+1]
        relative_delta_t = np.nancumsum(delta_t_samples[::-1])[:-1][::-1]

        cos = 0.5+ 0.5*np.cos( 2 * np.pi * relative_delta_t/365)
        if len (cos) > len(self.data_I):
            cos = cos[:len(self.data_I)]
            return np.average(self.data_I.values, axis=0, weights=cos)
        else:
            return np.average(self.data_I.iloc[:len(cos)].values, axis=0, weights=cos)

        


            




        return row


class superModel:
    def __init__(self, data_path, metadata_path):
        self.metadata_df, self.data_df = preprocessing(data_path, metadata_path)
        # create baboon models
        self.baboons = dict()
        cpus = max(1, min(multiprocessing.cpu_count() - 2, len(self.metadata_df["baboon_id"].unique())))
        futures = []

        with ProcessPoolExecutor(cpus) as executor:
                for baboon_id in self.metadata_df["baboon_id"].unique():
                    futures.append(executor.submit(BaboonModel, baboon_id, self.data_df, self.metadata_df))
        
        for fut in futures:
            baboon = fut.result()                  
            self.baboons[baboon.baboon_id] = baboon
        self.alpha_ = np.zeros(61)
        self.gamma_ = 0
    def fit(self):
        def objective(alpha, gamma):
            # calculate the objective function
            bc_sum = 0
            futures = [] # process pool executor futures
            cpus = max(1, min(multiprocessing.cpu_count() - 2, len(self.baboons)))

            with ProcessPoolExecutor(cpus) as executor:
                for baboon in self.baboons.values():
                    fut = executor.submit(baboon.fit, alpha, gamma) # execute the fit function of each baboon in parallel using the global alpha
                    futures.append(fut)
            
            for fut in futures:
                beta,  bc = fut.result()
                bc_sum += bc
            
            # print(f"for alpha \n{alpha}\nscore is {bc_sum}")
            return bc_sum
        
        # optimise beta using scipy.optimize.minimize
        
        optimezed_gamma = minimize(lambda gamma: objective(self.alpha_, gamma), x0 = self.gamma_, method="L-BFGS-B", bounds=[(0,1)])

        self.gamma = optimezed_gamma.x
        
        return objective(self.alpha_, self.gamma_)
    




        # calculate the weighted mean of the data for the given indexes
        
    def add_new_data(self, data_path, metadata_path):
        metadata_df, data_df = preprocessing(data_path, metadata_path)
        baboons_to_add = metadata_df["baboon_id"].unique()
        
        self.metadata_df = pd.concat([self.metadata_df, metadata_df]).sort_values(by = 'collection_date')
        
        self.data_df = pd.concat([self.data_df, data_df])
        dataless_baboons = []
        futures = []
        cpus = max(1, min(multiprocessing.cpu_count() - 2, len(baboons_to_add)))
        with ProcessPoolExecutor(cpus) as executor:
            for baboon_id in baboons_to_add:

                if baboon_id not in self.baboons.keys() and len(data_df.loc[np.intersect1d(metadata_df[metadata_df['baboon_id']==baboon_id].index, data_df.index)])< dataless_threshold:
                    dataless_baboons.append(baboon_id)
                else:
                    fut = executor.submit(BaboonModel, baboon_id, self.data_df, self.metadata_df, fit=True, alpha = self.alpha_, gamma = self.gamma_)
                    futures.append(fut)
        for fut in futures:
            baboon = fut.result()
            # print("baboon_id: ", baboon.baboon_id)
            self.baboons[baboon.baboon_id] = baboon

        futures = []
        cpus = max(1, min(multiprocessing.cpu_count() - 2, len(dataless_baboons)))
        mean_beta = np.mean([b.beta_ for b in self.baboons.values()], axis=0) # calculate the mean of beta for baboon with notenough data
        with ProcessPoolExecutor(multiprocessing.cpu_count() - 2) as executor:
            for baboon_id in dataless_baboons:
                fut = executor.submit(BaboonModel, baboon_id, self.data_df, self.metadata_df)
                futures.append(fut)
        for fut in futures:
            baboon = fut.result()
            baboon.beta_ = mean_beta
            self.baboons[baboon.baboon_id] = baboon


    def predict(self,  baboons_to_predict, iterative = False):
        predictions = pd.DataFrame(columns=self.data_df.columns)        
        cpus = max(1, min(multiprocessing.cpu_count() - 2, len(baboons_to_predict)))
        futures = []
        with ProcessPoolExecutor(cpus) as executor:
            for baboon_id in baboons_to_predict:
                # print("predicting: ",baboon_id)
                if baboon_id not in self.baboons.keys():
                    raise ValueError(f"baboon with id {baboon_id} is not in the model")
                baboon = self.baboons[baboon_id]
                # print(len(baboon.data_I), len(baboon.metadata_I))
                fut = executor.submit(baboon.predict, self.alpha_, self.gamma_, iterative)
                futures.append(fut)
        
        for fut in futures:
            predictions = pd.concat([predictions, fut.result()])
        return predictions



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
    data_clean.set_index('sample', inplace=True)

    # to drop = data_idx - chosen samples
    # metadata_idx.drop (to_drop)


    to_drop = data_df[~data_df['sample'].isin(data_clean.index)]["sample"].values

    metadata_clean = metadata_df[~metadata_df["sample"].isin(to_drop)]    
    metadata_clean.set_index('sample', inplace=True)

    metadata_clean['collection_date'] = pd.to_datetime(metadata_clean['collection_date'])
    return metadata_clean, data_clean
    

