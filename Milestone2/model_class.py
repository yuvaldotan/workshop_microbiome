class DO_model:
    def __init__(self):
        self.baboon_id = None
        self.alpha_ = None
        self.data = None
        self.mean_data = None #df of mean values
        self.delta_t = None # vector of time differences
        self.metadata = None
    
    def fit(self, lambda_):
        # calculate optimised alpha for a given lambda

        def objective(alpha, lambda_):
            # calculate the objective function

            '''for the ith row in self.data (from row 3)
            1. calculate mean of previous i-2 rows = d_mean
            2. calculate time difference between ith row and i-1th row = d_time
            calculate the prediction for the ith row using the formula'''


            '''
            calculate difference between prediction and actual value using bray-curtis dissimilarity and return the sum/mean -TBD'''

            delta_t = self.data['time'].diff()[-1]
            D_mean = self.data.mean(axis=0) #check axis
            D_time = self.data['time'].diff()

        # optimise alpha using scipy.optimize.minimize



    def predict(self, X_test):
        pass