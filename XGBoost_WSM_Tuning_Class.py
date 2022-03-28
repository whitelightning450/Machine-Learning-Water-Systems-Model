#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#my python module
import XGB_Model
from pathlib import Path
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[ ]:



#This is the final dataset to make predictions on
p = Path('Training_Simulations')

Sims = {i.stem[0:8] : pd.read_excel(i, skiprows = 5) for i in p.glob('**/*.xlsx')}


# In[ ]:


'''
A list of all of the model input features and the target.
Data takes in Goldsim simulations and splits them into training/testing.
Each simulation is the same for the training period (2000-2020)
Testing data varies based on climate scenario (Ave_Clim, Dro_Clim, Sur_Clim)

The data processing also adds a few features (the previous time steps target values and time variables)
'''
feat = [
    #Time
    'DOY' , 'Month',
    #Streamflow , 
     'SLCDPU_Surface_Supplies','BCC_Streamflow',  'LCC_Streamflow','Dell_Streamflow',
    'Lambs_Streamflow', 'CC_Streamflow',
    #Demands 
    'SLCDPU_Prod_Demands','SLCDPU_DC_Water_Use_Initial','SLCDPU_GW_Initial', 
    #Reservoir Levels
   'Mtn_Dell_Percent_Full_Initial','LittleDell_Percent_Full_Initial']

'''
select your target you want to train a model for:
(SLCDPU_GW, Mtn_Dell_Percent_Full, LittleDell_Percent_Full, SLCDPU_DC_Water_Use)

'''

targ = ['Mtn_Dell_Percent_Full'] 


#load Data processing module
DataProcc = XGB_Model.XGB_Tuning()

#Load in data and process accordingly
DataProcc.ProcessData(Sims, 'Obs_Dry', feat, targ, 2021, False, allData = True)

'''
This step check to collinearity among features.
In the instance feature collinearity exceeds the threshold (col_threashold),
the lesser correlated feature to the target will be removed.
These remaining features go to the next step, Recursive Feature Elimination

'''

DataProcc.CollinearityRemoval(col_threshold= .9)


# In[ ]:


'''
This step uses recursive feature elimination to identify the optimial features for the XGBoost
algorithm and the specific water system target. The function searches from 2 features to the maximum 
collinearity check features to determint the optimial targets features.
'''
#Run RFE feature selection to identify good features
DataProcc.FeatureSelection()

#from thorough analysis and testing, use these identified features
DataProcc.Feature_Optimization()


# In[ ]:



#Identify optimal water system component Parameters
#Any range can be used, however the following are optimized.

if targ[0] =='Mtn_Dell_Percent_Full':  #Excellent good.
    parameters = {
                  'nthread':[-1], #my identified features, prev
                  'objective':['reg:squarederror'],
                  'learning_rate': [1], #0.1, 0.3
                  'max_depth': [3], #4,5
                  'min_child_weight': [1], #6,4
                  'subsample': [0.9], #0.7,0.9
                  'colsample_bytree': [0.8], #0.8
                  "reg_lambda":[1.5], #1,1
                  'reg_alpha': [1.5],  #0,0
                  'n_estimators': [750], #375,350
                   'n_jobs':[-1]
                                       }

if targ[0] =='LittleDell_Percent_Full':  #Very good!, note, below params are for feature subsets
    parameters = {
                  'nthread':[-1], #when use hyperthread, xgboost may become slower
                  'objective':['reg:squarederror'],
                  'learning_rate': [ .4], #0.5 , 0.01, 0.4
                  'max_depth': [3], #3, 5 , 3
                  'min_child_weight': [4], #4 , 4, 4
                  'subsample': [0.8], #0.8, 0.5 , 0.8
                  'colsample_bytree': [0.8], #0.8, 0.8, 0.8
                  "reg_lambda":[0], #0,1 ,0
                  'reg_alpha': [0], #0,0, 0
                  'n_estimators': [20000], #1900, 500, 20000
                   'n_jobs':[-1]
                                       }
if targ[0] =='SLCDPU_DC_Water_Use':  #Good!
    parameters = {
                  'nthread':[-1], #when use hyperthread, xgboost may become slower
          'objective':['reg:squarederror'],
          'learning_rate': [.3], # 0.3 is good too
          'max_depth': [3], #3
          'min_child_weight': [8], #8
          'subsample': [0.8], #,0.8 is good
          'colsample_bytree': [0.8], #0.8 is good.
          "reg_lambda":[0], #0
          'reg_alpha': [0], #0
          'n_estimators': [20000],#10000, 20000 is good too
           'n_jobs':[-1]
                                       }

if targ[0] =='SLCDPU_GW':
    parameters = {
                  'nthread':[-1], #when use hyperthread, xgboost may become slower
                  'objective':['reg:squarederror'],
                  'learning_rate': [0.3], #0.3
                  'max_depth': [3], #3
                  'min_child_weight': [6], #6
                  'subsample': [0.6], #0.6
                  'colsample_bytree': [0.6], #0.6
                  "reg_lambda":[0], #0
                  'reg_alpha': [0], #0
                  'n_estimators': [500], #500
                   'n_jobs':[-1]
                                       }




#using the respective parameters, identifiy the optimal hyper-parameters the respective XGBoost model
DataProcc.GridSearch(parameters) 


# In[ ]:


#set directory to save model
M_save_filepath = "Models/V2/XGBoost_"+targ[0]+".dat"

#need to separate the train vs predict
DataProcc.Train(M_save_filepath) 


# In[ ]:





# In[ ]:


'''
Test the model and make a prediction on the unseen target for the respective water year
'''


# In[ ]:


#Make a prediction to evaluate each model
#XGBoost
XGBboost = XGB_Model.XGB_model(targ[0])
XGBboost.XGB_Predict(DataProcc.test_feat[DataProcc.Final_Features], DataProcc.test_targs)   
#Plot the prediction results
XGBboost.PredictionPerformancePlot()


# In[ ]:





# In[ ]:




