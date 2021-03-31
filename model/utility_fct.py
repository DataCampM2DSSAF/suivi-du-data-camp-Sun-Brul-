import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import KFold

def k_fold(n, value_est):
    kf = KFold(n_splits=5)

def expend_feature_test(df):
    """
    Return a dataframe with expension of sequence for test set prediction 
    
    Args:
        df (Dataframe): same format as train
        
    Returns:
        sub_df: a dataframe with: number of rows = seq_scored 
        columns name = [id, base, base_structure_type, base_predicted_loop_type]
    """
    if (df.shape[0] != df.id.nunique()):
        print('repetition in RNA sequnence, clean dataframe first')
        return # dose the same as retunr None, which exit the function 
    
    else:
        col_names = ['id','base', 'base_structure_type', 'base_predicted_loop_type']

        #dataframe creation using list of lists
        # loop for each unique sequence
        sub_data = []
        for row_i in df.index:
            #loop for the legth of sequnece score (trian length is different from test)
            serie_i = df.loc[row_i] #panda series object
            seq_length  = serie_i['seq_length']
            for seq_i in range (seq_length):
                seq_data = [serie_i['id'] + '_' + str(seq_i), serie_i['sequence'][seq_i],
                            serie_i['structure'][seq_i], serie_i['predicted_loop_type'][seq_i]]
                sub_data.append(seq_data)

    sub_df = pd.DataFrame(sub_data, columns =col_names,  dtype = float) 
    return sub_df


def fianle_transform_without_SN (dataframe, replace_type):
    # filter with SN_filter criteria 
    #use expend to change feature 
    data_filter_seq = expend_feature(dataframe)
    
    #make feature to integer
    for r_types in replace_type:  
        data_filter_seq = data_filter_seq.replace(r_types)
 
    return data_filter_seq


def expend_feature (df):
    """
    Return a dataframe with expension of sequence
    
    Args:
        df (Dataframe): same format as train    
    Returns:
        sub_df: a dataframe with:
        number of rows = seq_scored
        columns name:[id, base, base_structure_type, base_predicted_loop_type, reactivity_error,
        deg_error_Mg_pH10,deg_error_pH10, deg_error_Mg_50C, deg_error_50C, reactivity, 
        deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
    """
    if (df.shape[0] != df.id.nunique()):
        print('repetition in RNA sequnence, clean dataframe first')
        return # dose the same as retunr None, which exit the function 
    
    else:
        col_names = ['id','base', 'base_structure_type', 'base_predicted_loop_type', 'reactivity_error',
            'deg_error_Mg_pH10', 'deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C', 'reactivity', 
            'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

        #dataframe creation using list of lists
        # loop for each unique sequence
        sub_data = []
        for row_i in df.index:
            #loop for the legth of sequnece score (trian length is different from test)
            serie_i = df.loc[row_i] #panda series object
            seq_length  = serie_i['seq_scored']
            for seq_i in range (seq_length):
                seq_data = [serie_i['id'], serie_i['sequence'][seq_i],
                            serie_i['structure'][seq_i], serie_i['predicted_loop_type'][seq_i],
                            serie_i['reactivity_error'][seq_i], serie_i['deg_error_Mg_pH10'][seq_i],
                            serie_i['deg_error_pH10'][seq_i], serie_i['deg_error_Mg_50C'][seq_i],
                            serie_i['deg_error_50C'][seq_i], serie_i['reactivity'][seq_i],
                            serie_i['deg_Mg_pH10'][seq_i], serie_i['deg_pH10'][seq_i],
                            serie_i['deg_Mg_50C'][seq_i], serie_i['deg_50C'][seq_i]]
                sub_data.append(seq_data)

        sub_df = pd.DataFrame(sub_data, columns =col_names,  dtype = float) 
        return sub_df


def count(x,colonne) :
    return (colonne==x).sum()


def add_features(seq):
    # Coumpt the numerous of the group
    gr=1
    seq['count_gr']=gr
    for i in range (1,seq.shape[0]):
        if ((seq.id.iloc[i]!=seq.id.iloc[i-1]) or (seq.base_predicted_loop_type.iloc[i]!=seq.base_predicted_loop_type.iloc[i-1])):
            gr=gr+1
        seq.count_gr[i]=gr
    #Count the number of letters in the group 
    seq['N']=seq['count_gr'].apply(lambda x: count(x,seq['count_gr']))
    return seq
    
    
def fianle_transform (dataframe):
    # filter with SN_filter criteria 
    data_filter = dataframe[dataframe["SN_filter"] == 1]
    
    #use expend to change feature 
    data_filter_seq = expend_feature(data_filter)
    
    #add feature 
    data_filter_seq = add_features(data_filter_seq)
    
    #make feature onehot encoding
    one_hot_base = pd.get_dummies(data_filter_seq['base'],drop_first=True) #drop "A"
    one_hot_loop = pd.get_dummies(data_filter_seq['base_predicted_loop_type'],drop_first=True) # drop "B"
    one_hot_struct = pd.get_dummies(data_filter_seq['base_structure_type'],drop_first=True)# drop "("
    
    #final dataset
    data_filter_seq = data_filter_seq.drop(["base", "base_predicted_loop_type","base_structure_type"], axis = 1)
    df = pd.concat([data_filter_seq,one_hot_base, one_hot_loop ,one_hot_struct], axis = 1)
    return df


def MSE(vec1,vec2):
    Res = 0 
    n=len(vec1)
    for i in range (0,n):  #looping through each element of the list
        diff_squared = (vec1[i]-vec2[i])**2   
        Res = Res + diff_squared  #taking a sum of all the differences
    return Res/n #dividing summation by total values to obtain average


def mcrmsc_yannick(y_true, y_pred):
        y_true = y_true.values
        rmse = []
        for i in range (5):
            rmse.append(mean_squared_error(y_true[:,i], y_pred[:,i], squared = False))      
        mcrmsc = np.mean(rmse)
        return (mcrmsc, rmse)
    
def ajout_N_predicted_loop(data):
    for i in letters_loop_type:
        col = np.where(data[i]==1, data['N'], 0) 
        data ["N_"+i] = col
    col_B=np.where(data.N_E+data.N_H+data.N_I+data.N_M+data.N_S+data.N_X==0,data.N,0)
    data["N_B"]=col_B
    return data


def MCRMSE(y_true, y_pred):
    """
    Return loss between true and prediction, with  mean column wise root mean squared error
    from sklearn.metrics import mean_squared_error
    Args:
        y_true: matrix 
        y_pred: matrix
       
    Returns:
        output: double 
    """
    y_true = y_true.values
    n,Nt = y_pred.shape
    
    Res = 0
    
    for i in range(0,Nt):
        Res = Res + mean_squared_error(y_true[:,i], y_pred[:,i], squared = False)
    return Res/ Nt



class linear_mcrmse: 
    """       
    Parameters
    ----------
    X : `dataframe`, shape=(n_samples,n_features)
        features
    y : `dataframe`, shape=(n_samples, n_y)
        double
    lamb : `float`, 
        value of the regularization parameter
    beta : `numpy.array`, shape=(n_features,n_y)
        weight matrix
    """
    
    def __init__(self,X,y,lamb, n_ite = 10000, precision = 10^-4, beta = None):
        self.X_ = np.asanyarray(X)
        self.y_ = np.asanyarray(y)
        self.lamb_ = lamb
        self.n_samples_, self.n_features_ = X.shape
        self.n_y_ = y.shape[1]
        self.beta_ = np.random.random((self.n_features_, self.n_y_))
        if (beta == None):
            self.beta_ = np.zeros((self.n_features_, self.n_y_))
        else:
            self.beta_ = beta
        self.n_ite_ = n_ite
        self.precision_ = precision

    
    def loss(self):
        # compute mcrmsc loss 
        y_pred =  np.dot(self.X_, self.beta_)
        rmse = []
        for i in range (self.n_y_):
            rmse.append(mean_squared_error(y_pred[:,i],self.y_[:,i], squared = False))
        mcrmsc = np.mean(rmse)
        return (mcrmsc, rmse)
    
    
    def grad_loss(self):
        # the gradiant for mcrmsc gradiant
        rmse = self.loss()[1]
        grad = np.zeros((self.n_features_, self.n_y_))
        y_pred =  np.dot(self.X_, self.beta_)
        for j in range(self.n_y_):
            # loop over columns
            for i in range(self.n_features_):
                #loop over line
                grad_temp = 0
                for x in range(self.n_y_):
                    #loop over column 
                    grad_temp += self.X_[i, x]*(y_pred[i, x] - self.y_[i, x])/ rmse[x] / self.n_features_
                grad[i, j] = grad_temp/ self.n_y_
        return (grad)
    
    def fit (self):
        self.cost_ = [[100],]
        y_pred =  np.dot(self.X_, self.beta_)
        cost = self.loss()
        self.cost_.append(cost)
        for _ in range(self.n_ite_):
            cost = self.loss()[0]
            if (cost > self.cost_[-2][0]):
                break 
            else: 
                gradient_vector = self.grad_loss()
                self.beta_ -= (self.lamb_)/self.n_features_ * gradient_vector
                cost = self.loss()
                self.cost_.append(cost)
        return self
    
    def predict(self, X_test):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        X_test : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        return np.dot(X_test, self.beta_)