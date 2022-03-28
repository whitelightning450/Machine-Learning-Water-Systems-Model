#Script developed by Ryan C. Johnson, University of Alabama for the
#Salt Lake City Climate Vulnerability Project.
#Date: 3/4/2022
# coding: utf-8


import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from xgboost import cv
import time
import pickle
import joblib
from pickle import dump
import numpy as np
import copy
from collinearity import SelectNonCollinear
from sklearn.feature_selection import f_regression 
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import RFE
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from xgboost import cv
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

from numpy import mean
from numpy import std
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from collections import defaultdict
import jenkspy
from matplotlib.dates import MonthLocator, DateFormatter


class XGB_model():
    
    
    def __init__(self, Target, cwd):
        self = self
        self.Target = Target
        self.Prediction = self.Target+'_Pred'
        self.Prediction_Rolling = self.Prediction+'_Rolling'
        self.T_initial = self.Target+'_Initial'
        self.cwd = cwd
        
    def fit(self,param, X,y, M_save_filepath):
        self.param=param
        self.num_round=param['num_boost_round']
        start_time = time.time()      
        print('Model Training')
        y = y[self.Target]
        feature_names = list(X.columns)
        dtrain = xgb.DMatrix(np.array(X), label=np.array(y),feature_names=feature_names)
        model = xgb.Booster(self.param, [dtrain])
        model = xgb.train(self.param,dtrain,num_boost_round=self.num_round, xgb_model=model)

        c_time = round(time.time() - start_time,2)
        print('Calibration time', round(c_time), 's')
        print('Saving Model')
        #adjust this to match changing models
        pickle.dump(model, open(self.cwd + M_save_filepath, "wb"))   

        self.model_=model
        
        
        
    def predict(self,X, model):
        self.model_=model
        dtest=xgb.DMatrix(X)
        return self.model_.predict(dtest) 
    
    
    
    
    def XGB_Predict(self, test_feat, test_targ):
        
        #Make predictions with the model
        model = pickle.load(open(self.cwd+"/Model_History/V2/XGBoost_"+self.Target+".dat", "rb"))
        start_time = time.time()  
        #since the previous timestep is being used, we need to predict this value
        predict = []
        featcol = test_feat.columns
        for i in range(0,(len(test_feat)-1),1):
            t_feat = np.array(test_feat.iloc[i])
            t_feat = t_feat.reshape(1,len(t_feat))
            t_feat = pd.DataFrame(t_feat, columns = featcol)
            p = self.predict(t_feat, model)
            if self.T_initial in featcol:
                test_feat[self.T_initial].iloc[(i+1)] = p
            predict.append(p[0])
        #need to manually add one more prediction
        predict.append(predict[-1])

        #add physical limitations to predictions
        if self.Target =='SLCDPU_GW':
            predict = np.array(predict)
            predict[predict > 89.49] = 89.49


        #Use this line for PCA
        #predict = Targ_scaler.inverse_transform(predict.reshape(len(predict),1))
        c_time = round(time.time() - start_time,8)
        print('prediction time', round(c_time), 's')

        #Analyze model performance
        #use this line for PCA
        #Targ_scaler.inverse_transform(test_targ)
        Analysis = pd.DataFrame(test_targ, columns = [self.Target])
        Analysis[self.Prediction] = predict
        Analysis[self.Prediction_Rolling] = Analysis[self.Prediction].rolling(5).mean()

        Analysis[self.Prediction_Rolling] = Analysis[self.Prediction_Rolling].interpolate(method='linear',
                                                       limit_direction='backward', 
                                                       limit=5)

        RMSEpred = mean_squared_error(Analysis[self.Target],Analysis[self.Prediction], squared=False)
        RMSErolling = mean_squared_error(Analysis[self.Target],Analysis[self.Prediction_Rolling], squared=False)

     #   print('RMSE for predictions: ', RMSEpred, 'af/d. RMSE for rolling prediction mean: ', RMSErolling, 'af/d')

        self.Analysis = Analysis

    
    
       
    #Make a plot of predictions
    def PredictionPerformancePlot(self):

        #predicted and observed
        labelsize = 14

        # better control over ax
        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches(9,8)
        maxGW = max(max(self.Analysis[self.Target]), max(self.Analysis[self.Prediction]))*1.2

        self.Analysis.plot( y = self.Target, ax=ax[0], color = 'blue', label = self.Target)
        self.Analysis.plot(y = self.Prediction , ax=ax[0], color = 'orange', label = self.Prediction)
        self.Analysis.plot(y = self.Prediction_Rolling , ax=ax[0], color = 'green', label = self.Prediction_Rolling)

        ax[0].set_xlabel('Time ', size = labelsize)
        ax[0].set_ylabel(self.Target, size = labelsize)
        #plt.xlim(0,370)
        ax[0].set_ylim(0,maxGW*1.4)
        ax[0].legend(loc="upper left",title = 'Prediction/Target')

        self.Analysis.plot.scatter(x = self.Target, y = self.Prediction_Rolling , ax=ax[1], color = 'green', label = self.Prediction_Rolling)
        self.Analysis.plot.scatter(x = self.Target, y = self.Prediction , ax=ax[1], color = 'orange', label = self.Prediction)

        ax[1].plot((0,maxGW),(0,maxGW), linestyle = '--', color  = 'red')



        #plt.title('Production Simulations', size = labelsize+2)
        #fig.savefig(O_path + 'Figures/MLP/MLP_Prod.png', dpi = 300)

        plt.show()
        RMSEpred = mean_squared_error(self.Analysis[self.Target],self.Analysis[self.Prediction], squared=False)
        RMSErolling = mean_squared_error(self.Analysis[self.Target],self.Analysis[self.Prediction_Rolling], squared=False)

        print('RMSE for predictions: ', RMSEpred, '. RMSE for rolling prediction mean: ', RMSErolling)





    
    


#Developing the XGBoost_Tuning package
class XGB_Tuning():
    
    
    
    def __init__(self, cwd):
        self = self
        self.cwd = cwd
        
        
        
    def ProcessData(self, df, sim, feat, targ, test_yr, scaling, allData):
        print('Processing data to tune XGBoost model for ', targ[0])
        print('This may take a few moments depending on computational power and data size')
        self.targ = targ[0]
        data = copy.deepcopy(df)
        #get month, day, year, from df
        dflen = len(data[sim])
        months = []
        days = []
        years = []
        data[sim]['DOY'] = 0
        for t in range(0,dflen,1):
            y = data[sim]['Time'][t].year
            m = data[sim]['Time'][t].month
            d = data[sim]['Time'][t].day
            months.append(m)
            days.append(d)
            years.append(y)

            data[sim]['DOY'].iloc[t] = data[sim]['Time'].iloc[t].day_of_year

        years = list( dict.fromkeys(years) )  
        #remove yr 2000  and 2022 as it is not a complete year
        #test by removing 2008, 2015, and 2017 too as these are the test years
        years = years[1:-1]
        data[sim]['Month'] = months
        data[sim]['Day'] = days
        data[sim].index = data[sim]['Time']



        #input each year's initial reservoir conditions./ previous timestep conditions.
        data[sim]['Mtn_Dell_Percent_Full_Initial'] = 0
        data[sim]['LittleDell_Percent_Full_Initial'] = 0
        data[sim]['SLCDPU_GW_Initial'] = 0
        data[sim]['SLCDPU_DC_Water_Use_Initial'] = 0
        timelen = len(data[sim])
        for t in range(0,timelen, 1):
            data[sim]['Mtn_Dell_Percent_Full_Initial'].iloc[t] = data[sim]['Mtn_Dell_Percent_Full'].iloc[(t-1)]
            data[sim]['LittleDell_Percent_Full_Initial'].iloc[t] = data[sim]['LittleDell_Percent_Full'].iloc[(t-1)]
            data[sim]['SLCDPU_GW_Initial'].iloc[t] = data[sim]['SLCDPU_GW'].iloc[(t-1)]
            data[sim]['SLCDPU_DC_Water_Use_Initial'].iloc[t] = data[sim]['SLCDPU_DC_Water_Use'].iloc[(t-1)]


        #make an aggregated streamflow metric
        data[sim]['SLCDPU_Surface_Supplies'] = data[sim]['BCC_Streamflow']+data[sim]['LCC_Streamflow']+data[sim]['CC_Streamflow']+data[sim]['Dell_Streamflow']+data[sim]['Lambs_Streamflow']


        features = data[sim][feat]
        targets = data[sim][targ]

        f_col = list(features.columns)
        t_col = list(targets.columns)

        if scaling ==True:
            del data[sim]['Time']
            Feat_scaler = MinMaxScaler()
            Targ_scaler = MinMaxScaler()
            Feat_scaler.fit(features)
            Targ_scaler.fit(targets)
            features = Feat_scaler.transform(features)
            targets = Targ_scaler.transform(targets)
            f = pd.DataFrame(features, columns = f_col)
            t = pd.DataFrame(targets, columns = t_col)
            f.index = data[sim].index
            t.index = data[sim].index

        else:
            f = features
            t = targets
            #looks like adding more data can help train models, extending period to include      march and april
        train_feat = f.loc['2000-10-1':str(test_yr)+'-3-31']
        train_targ = t.loc['2000-10-1':str(test_yr)+'-3-31']

        test_feat = f.loc[str(test_yr)+'-4-1':str(test_yr)+'-10-31']
        test_targs =t.loc[str(test_yr)+'-4-1':str(test_yr)+'-10-31']



        if allData == True:
            #need to remove years 2008,2015,2017 as these are testing streamflow conditions.
            testyrs = [2008,2015,2017]
            trainyrs = list(np.arange(2001, 2021, 1))

            for t in testyrs:
                trainyrs.remove(t)
                train_feat.drop(train_feat.loc[str(t)+'-4-1':str(t)+'-10-31'].index, inplace=True)
                train_targ.drop(train_targ.loc[str(t)+'-4-1':str(t)+'-10-31'].index, inplace=True)

        if allData ==False:
        #need to remove years 2008,2015,2017 as these are testing streamflow conditions.
            testyrs = [2008,2015,2017]
            trainyrs = list(np.arange(2001, 2021, 1))

            for t in testyrs:
                trainyrs.remove(t)
                train_feat.drop(train_feat.loc[str(t-1)+'-11-1':str(t)+'-10-31'].index, inplace=True)
                train_targ.drop(train_targ.loc[str(t-1)+'-11-1':str(t)+'-10-31'].index, inplace=True)

            # Model is focused on April to October water use, remove dates out of this timeframe
            for t in trainyrs:    
                train_feat.drop(train_feat.loc[str(t-1)+'-12-1':str(t)+'-1-31'].index, inplace=True)
                train_targ.drop(train_targ.loc[str(t-1)+'-12-1':str(t)+'-1-31'].index, inplace=True)

            #Drop WY2000
            train_feat.drop(train_feat.loc['2000-1-1':'2001-3-30'].index, inplace=True)
            train_targ.drop(test_targ.loc['2000-1-1':'2001-3-30'].index, inplace=True)

        #Shuffle training data to help model training
        if scaling ==True:
            return train_feat, train_targ, test_feat, test_targs, Targ_scaler

        else:
            self.train_feat, self.train_targ, self.test_feat,self.test_targs = train_feat, train_targ, test_feat, test_targs
            
            
            
            
            
            
            
    def CollinearityRemoval(self, col_threshold):
        print('Calculating collinearity matrix and removing features > ', str(col_threshold))
        start_time = time.time()  
        #look at correlations among features
        features = self.train_feat.columns
        X = np.array(self.train_feat)
        y = np.array(self.train_targ)

        selector = SelectNonCollinear(correlation_threshold=col_threshold,scoring=f_regression) 
        selector.fit(X,y)
        mask = selector.get_support()
        Col_Check_feat = pd.DataFrame(X[:,mask],columns = np.array(features)[mask]) 
        Col_Check_features = Col_Check_feat.columns
        sns.heatmap(Col_Check_feat.corr().abs(),annot=True)

        self.Col_Check_feat, self.Col_Check_features =Col_Check_feat, Col_Check_features
        c_time = round(time.time() - start_time,8)
        print('Feature development time', round(c_time), 's')
        
        
        
        
        
        # get a list of models to evaluate
    def get_models(self):
        models = dict()
        for i in range(2, len(self.X.columns)):
            rfe = RFE(estimator=XGBRegressor(), n_features_to_select=i)
            model = XGBRegressor()
            models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
        self.models = models

        
        
        
    # evaluate a given model using cross-validation
    def evaluate_model(self, model):
        #pipeline = Pipeline(steps=[('s',rfe),('m',model)])
        pipeline = model
        # evaluate model
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=1)
        n_scores = cross_val_score(pipeline, self.X, self.y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
        self.scores = n_scores

        
        
        
        
        

    def FeatureSelection(self):
        start_time = time.time()  
        # define dataset
        X = self.Col_Check_feat
        self.X =X
        y = self.train_targ# get the models to evaluate
        self.y = y
        self.get_models()

        # evaluate the models and store results
        results, names = list(), list()
        print('Using RFE to determine optimial features, scoring is:')
        for name, model in self.models.items():
            self.evaluate_model(model)
            results.append(self.scores)
            names.append(name)
            print('>%s %.3f (%.3f)' % (name, mean(self.scores), std(self.scores)))

        score_cols = ['n_feat' , 'mean_MAE', 'std_MAE']
        Feat_Eval = pd.DataFrame(columns = score_cols)

        for i in range(0,len(results)):
            feats = i+2
            meanMAE = mean(results[i])
            stdMAE = std(results[i])
            s = [feats, abs(meanMAE), stdMAE]
            Feat_Eval.loc[len(Feat_Eval)] = s  
            #mean and std MAE both are applicable. std works well when feweer features are used
        Feat_Eval=Feat_Eval.sort_values(by=['std_MAE', 'n_feat'])
        Feat_Eval = Feat_Eval.reset_index()
        print(Feat_Eval)
        n_feat = int(Feat_Eval['n_feat'][0])

        # create pipeline
        rfe = RFE(estimator=XGBRegressor(), n_features_to_select=n_feat)
        rfe = rfe.fit(X, y)
        # summarize the selection of the attributes
        print(rfe.support_)
        print(rfe.ranking_)


        RFE_Feat = pd.DataFrame(self.Col_Check_features, columns = ['Features'])
        RFE_Feat['Selected']= rfe.support_
        RFE_Feat = RFE_Feat[RFE_Feat['Selected']==True]
        RFE_Feat = RFE_Feat['Features']
        RFE_Features = self.Col_Check_feat[RFE_Feat]
        print('The Recursive Feature Elimination identified features are: ')
        print(list(RFE_Feat))

        self.Final_FeaturesDF, self.Final_Features = RFE_Features, list(RFE_Feat)
        c_time = round(time.time() - start_time,8)
        print('Feature selection time: ', round(c_time), 's')


        
        
        
        
        
        
   #These are the top features for XBoost
 #RFE feature selection is a good starting point, but these features optimize predictive performance
    def Feature_Optimization(self):
        print(' ')
       
        print('Features optimization identifies the following features best fit for the XGB-WSM')
        if self.targ =='LittleDell_Percent_Full':
            self.Final_Features = ['Month', 'Dell_Streamflow', 'Mtn_Dell_Percent_Full_Initial', 'LittleDell_Percent_Full_Initial']
            self.Final_FeaturesDF = self.Col_Check_feat[self.Final_Features]

        if self.targ =='Mtn_Dell_Percent_Full':
            self.Final_Features= ['SLCDPU_Surface_Supplies', 'Dell_Streamflow', 'Lambs_Streamflow',
                                      'SLCDPU_GW_Initial', 'Mtn_Dell_Percent_Full_Initial']
            self.Final_FeaturesDF = self.Col_Check_feat[self.Final_Features]

        if self.targ=='SLCDPU_GW':
            self.Final_FeaturesDF = self.Col_Check_feat[self.Final_Features]

        if self.targ =='SLCDPU_DC_Water_Use':
            self.Final_Features = ['BCC_Streamflow', 'SLCDPU_Prod_Demands', 'SLCDPU_DC_Water_Use_Initial', 
                                   'Mtn_Dell_Percent_Full_Initial', 'LittleDell_Percent_Full_Initial']
            self.Final_FeaturesDF = self.Col_Check_feat[self.Final_Features]

        #save features list 
        pickle.dump(self.Final_Features, open(self.cwd + "/Model_History/V2/"+self.targ+"_features.pkl", "wb")) 

        print('The final features for ', self.targ, 'are: ')
        print(self.Final_FeaturesDF.columns)

        
        
        #gridsearch hyper parameter function
    def GridSearch(self, parameters):
        start_time = time.time() 
        print('Performing a Grid Search to identify the optimial model hyper-parameters')
        xgb1 = XGBRegressor()


        xgb_grid = GridSearchCV(xgb1,
                                parameters,
                                cv = 3,
                                n_jobs = -1,
                                verbose=3)
        xgb_grid.fit(self.Final_FeaturesDF, self.train_targ[self.targ])

        print('The best hyperparameter three-fold cross validation score is: ')
        print(xgb_grid.best_score_)

        print(' ')
        print('The optimal hyper-parameters are: ')
        print(xgb_grid.best_params_)
        
        print(' ')
        c_time = round(time.time() - start_time,8)
        print('Hyper-parameter Optimization time', round(c_time), 's')

        self.xgb_grid = xgb_grid
        
        
        
        
        
#Model Training Function
    def Train(self, M_save_filepath):   
        
        #get the optimial hyperparams
        params = {"objective":"reg:squarederror",
                  'booster' :  "gbtree" , 
                  'eta': self.xgb_grid.best_params_['learning_rate'],
                  "max_depth":self.xgb_grid.best_params_['max_depth'],
                  "subsample":self.xgb_grid.best_params_['subsample'],
                  "colsample_bytree":self.xgb_grid.best_params_['colsample_bytree'],
                  "reg_lambda":self.xgb_grid.best_params_['reg_lambda'],
                  'reg_alpha':self.xgb_grid.best_params_['reg_alpha'],
                  "min_child_weight":self.xgb_grid.best_params_['min_child_weight'],
                  'num_boost_round':self.xgb_grid.best_params_['n_estimators'],
                  'verbosity':0,
                'nthread':-1
                 }


        #Train the model
        model = XGB_model(self.targ, self.cwd) 
        model.fit(params,self.Final_FeaturesDF, self.train_targ, M_save_filepath)
        xgb.plot_importance(model.model_, max_num_features=20)
        
        
        
        
        
        
        
        
        
        
        
#XGB Prediction Engine
class XGB_Prediction():
    
    def __init__(self, MDell_Thresh, LDell_Thresh, units):
        self = self
        #set reservoir level thresholds as global vars
        self.MDell_Thresh = MDell_Thresh
        self.LDell_Thresh = LDell_Thresh
        self.units = units
    
    #Data Processing needed to make a prediction
    def ProcessData(self, Sim, scenario, test_yr):
        #define global variables
        self.scenario = scenario
        self.Sim = Sim
        self.test_yr = test_yr
        
        print('Processing data into features/targets for ', self.scenario, ' scenario')
        #Input optimial features from XGBoost_WSM_Tuning.
        LittleDell_Percent_Full = pickle.load(open("Models/V2/LittleDell_Percent_Full_features.pkl", "rb"))
        Mtn_Dell_Percent_Full = pickle.load(open("Models/V2/Mtn_Dell_Percent_Full_features.pkl", "rb")) 
        SLCDPU_GW = pickle.load(open("Models/V2/SLCDPU_GW_features.pkl", "rb"))
        SLCDPU_DC_Water_Use = pickle.load(open("Models/V2/SLCDPU_DC_Water_Use_features.pkl", "rb"))



        feat = {
                'LittleDell_Percent_Full':LittleDell_Percent_Full,
                'Mtn_Dell_Percent_Full':Mtn_Dell_Percent_Full,
                'SLCDPU_GW': SLCDPU_GW,
                'SLCDPU_DC_Water_Use': SLCDPU_DC_Water_Use   
                }

        #make a DF with some additional features (from GS)
        data = copy.deepcopy(self.Sim)
        dflen = len(data[self.scenario])
        months = []
        days = []
        years = []
        data[self.scenario]['DOY'] = 0
        for t in range(0,dflen,1):
            y = data[self.scenario]['Time'][t].year
            m = data[self.scenario]['Time'][t].month
            d = data[self.scenario]['Time'][t].day
            months.append(m)
            days.append(d)
            years.append(y)

            data[self.scenario]['DOY'].iloc[t] = data[self.scenario]['Time'].iloc[t].day_of_year

        years = list( dict.fromkeys(years) )  
        #remove yr 2000  and 2022 as it is not a complete year
        years = years[1:-1]
        data[self.scenario]['Month'] = months
        data[self.scenario]['Day'] = days
        data[self.scenario].index = data[self.scenario]['Time']



        #input each year's initial reservoir conditions./ previous timestep conditions.
        data[self.scenario]['Mtn_Dell_Percent_Full_Initial'] = 0
        data[self.scenario]['LittleDell_Percent_Full_Initial'] = 0
        data[self.scenario]['SLCDPU_GW_Initial'] = 0
        data[self.scenario]['SLCDPU_DC_Water_Use_Initial'] = 0
        timelen = len(data[self.scenario])
        for t in range(0,timelen, 1):
            data[self.scenario]['Mtn_Dell_Percent_Full_Initial'].iloc[t] = data[self.scenario]['Mtn_Dell_Percent_Full'].iloc[(t-1)]
            data[self.scenario]['LittleDell_Percent_Full_Initial'].iloc[t] = data[self.scenario]['LittleDell_Percent_Full'].iloc[(t-1)]
            data[self.scenario]['SLCDPU_GW_Initial'].iloc[t] = data[self.scenario]['SLCDPU_GW'].iloc[(t-1)]
            data[self.scenario]['SLCDPU_DC_Water_Use_Initial'].iloc[t] = data[self.scenario]['SLCDPU_DC_Water_Use'].iloc[(t-1)]


        #make an aggregated streamflow metric
        data[self.scenario]['SLCDPU_Surface_Supplies'] = data[self.scenario]['BCC_Streamflow']+data[self.scenario]['LCC_Streamflow']+data[self.scenario]['CC_Streamflow']+data[self.scenario]['Dell_Streamflow']+data[self.scenario]['Lambs_Streamflow']

        #Make dictionary of acutal features
        features = { 'LittleDell_Percent_Full':data[self.scenario][feat['LittleDell_Percent_Full']],
                'Mtn_Dell_Percent_Full':data[self.scenario][feat['Mtn_Dell_Percent_Full']],
                'SLCDPU_GW': data[self.scenario][feat['SLCDPU_GW']],
                'SLCDPU_DC_Water_Use': data[self.scenario][feat['SLCDPU_DC_Water_Use']]  
                   }  
        #set up Targets
        targ = ['SLCDPU_GW', 'Mtn_Dell_Percent_Full', 'LittleDell_Percent_Full','SLCDPU_DC_Water_Use']
        targets = data[self.scenario][targ]

        for i in features:
            features[i] = features[i].loc[str(self.test_yr)+'-4-1':str(self.test_yr)+'-10-30']

        Hist_targs = targets.loc[:str(self.test_yr)+'-3-31'].copy()
        targets = targets.loc[str(self.test_yr)+'-4-1':str(self.test_yr)+'-10-30']

        self.features, self.targets, self.Hist_targs =features, targets, Hist_targs

        
        
     #This uses the XGB model to make predictions for each water system component at a daily time step.
    def WSM_Predict(self):   
        #Set up the target labels
        #Mountain Dell
        self.MDell = 'Mtn_Dell_Percent_Full'
        self.MDell_Pred = self.MDell+'_Pred'
        self.MDell_Pred_Rol = self.MDell_Pred+'_Rolling'
        self.MDell_Initial = self.MDell+'_Initial'

        #Little Dell
        self.LDell = 'LittleDell_Percent_Full'
        self.LDell_Pred = self.LDell+'_Pred'
        self.LDell_Pred_Rol = self.LDell_Pred+'_Rolling'
        self.LDell_Initial = self.LDell+'_Initial'

        #GW
        self.GW = 'SLCDPU_GW'
        self.GW_Pred = self.GW+'_Pred'
        self.GW_Pred_Rol = self.GW_Pred+'_Rolling'
        self.GW_Initial = self.GW+'_Initial'

        #GW
        self.DC = 'SLCDPU_DC_Water_Use'
        self.DC_Pred = self.DC+'_Pred'
        self.DC_Pred_Rol = self.DC_Pred+'_Rolling'
        self.DC_Initial = self.DC+'_Initial'

        #Grab features/targets for the respective target    
        MDell_feat = copy.deepcopy(self.features[self.MDell])
        MDell_targ = copy.deepcopy(self.targets[self.MDell])

        LDell_feat = copy.deepcopy(self.features[self.LDell])
        LDell_targ = copy.deepcopy(self.targets[self.LDell])

        GW_feat = copy.deepcopy(self.features[self.GW])
        GW_targ = copy.deepcopy(self.targets[self.GW])

        DC_feat = copy.deepcopy(self.features[self.DC])
        DC_targ = copy.deepcopy(self.targets[self.DC])

        #Make predictions with the model, load model from XGBoost_WSM_Tuning
        MDell_model = pickle.load(open("Models/V1/XGBoost_"+self.MDell+".dat", "rb"))
        LDell_model = pickle.load(open("Models/V2/XGBoost_"+self.LDell+".dat", "rb"))
        GW_model = pickle.load(open("Models/V2/XGBoost_"+self.GW+".dat", "rb"))
        DC_model = pickle.load(open("Models/V2/XGBoost_"+self.DC+".dat", "rb"))




        start_time = time.time()  
        #since the previous timestep is being used, we need to predict this value
        #Mtn dell
        MDell_predict = []
        MDell_col = MDell_feat.columns

        #lil Dell
        LDell_predict = []
        LDell_col = LDell_feat.columns

        #GW
        GW_predict = []
        GW_col = GW_feat.columns

        #GW
        DC_predict = []
        DC_col = DC_feat.columns

        #Make Predictions by row, update DF intitials to make new row prediction based on the current
        for i in range(0,(len(LDell_feat)-1),1):
            #MOuntain Dell
            MDell_t_feat = np.array(MDell_feat.iloc[i])
            MDell_t_feat = MDell_t_feat.reshape(1,len(MDell_t_feat))
            MDell_t_feat = pd.DataFrame(MDell_t_feat, columns = MDell_col)
            M = XGB_model.predict(MDell_model, MDell_t_feat, MDell_model)

            #Little Dell
            LDell_t_feat = np.array(LDell_feat.iloc[i])
            LDell_t_feat = LDell_t_feat.reshape(1,len(LDell_t_feat))
            LDell_t_feat = pd.DataFrame(LDell_t_feat, columns = LDell_col)
            L = XGB_model.predict(LDell_model, LDell_t_feat, LDell_model)

            #GW
            GW_t_feat = np.array(GW_feat.iloc[i])
            GW_t_feat = GW_t_feat.reshape(1,len(GW_t_feat))
            GW_t_feat = pd.DataFrame(GW_t_feat, columns = GW_col)
            G = XGB_model.predict(GW_model, GW_t_feat, GW_model)
            #add physical limitations to predictions
            G = np.array(G)


            #DC
            DC_t_feat = np.array(DC_feat.iloc[i])
            DC_t_feat = DC_t_feat.reshape(1,len(DC_t_feat))
            DC_t_feat = pd.DataFrame(DC_t_feat, columns = DC_col)
            D = XGB_model.predict(DC_model, DC_t_feat, DC_model)






            #This updates each DF with the predictions
            #Mountain Dell Features
            if self.LDell_Initial in MDell_col:
                MDell_feat[self.LDell_Initial].iloc[(i+1)] = L

            if self.MDell_Initial in MDell_col:
                MDell_feat[self.MDell_Initial].iloc[(i+1)] = M

            if self.GW_Initial in MDell_col:
                MDell_feat[self.GW_Initial].iloc[(i+1)] = G

            if self.DC_Initial in MDell_col:
                MDell_feat[self.DC_Initial].iloc[(i+1)] = D

            #Little Dell Features
            if self.LDell_Initial in LDell_col:
                LDell_feat[self.LDell_Initial].iloc[(i+1)] = L

            if self.MDell_Initial in LDell_col:
                LDell_feat[self.MDell_Initial].iloc[(i+1)] = M

            if self.GW_Initial in LDell_col:
                LDell_feat[self.GW_Initial].iloc[(i+1)] = G

            if self.DC_Initial in LDell_col:
                LDell_feat[self.DC_Initial].iloc[(i+1)] = D

            #Gw Features
            if self.LDell_Initial in GW_col:
                GW_feat[self.LDell_Initial].iloc[(i+1)] = L

            if self.MDell_Initial in GW_col:
                GW_feat[self.MDell_Initial].iloc[(i+1)] = M

            if self.GW_Initial in GW_col:
                GW_feat[self.GW_Initial].iloc[(i+1)] = G

            if self.DC_Initial in GW_col:
                GW_feat[self.DC_Initial].iloc[(i+1)] = D

            #DC Features
            if self.LDell_Initial in DC_col:
                DC_feat[self.LDell_Initial].iloc[(i+1)] = L

            if self.MDell_Initial in DC_col:
                DC_feat[self.MDell_Initial].iloc[(i+1)] = M

            if self.GW_Initial in DC_col:
                DC_feat[self.GW_Initial].iloc[(i+1)] = G

            if self.DC_Initial in DC_col:
                DC_feat[self.DC_Initial].iloc[(i+1)] = D



          #Append predictions      
            MDell_predict.append(M[0])
            LDell_predict.append(L[0])
            GW_predict.append(G[0])
            DC_predict.append(D[0])


        #need to manually add one more prediction
        MDell_predict.append(MDell_predict[-1])
        LDell_predict.append(LDell_predict[-1])
        GW_predict.append(GW_predict[-1])
        DC_predict.append(DC_predict[-1])


        #Use this line for PCA
        c_time = round(time.time() - start_time,8)
        print('prediction time', round(c_time), 's')

        #Analyze model performance
        #Add Little Dell
        Analysis = pd.DataFrame(LDell_predict,index =self.features[self.LDell].index, columns = [self.LDell_Pred])
        #non-zero values cannot occur
        Analysis[self.LDell_Pred][Analysis[self.LDell_Pred]<0] = 0


        #Add Mountain Dell
        Analysis[self.MDell_Pred] = MDell_predict
        #non-zero values cannot occur
        Analysis[self.MDell_Pred][Analysis[self.MDell_Pred]<0] = 0


        #Add GW
        Analysis[self.GW_Pred] = np.float32(GW_predict)
        #non-zero values cannot occur
        Analysis[self.GW_Pred][Analysis[self.GW_Pred]<0] = 0


        #Add DC
        Analysis[self.DC_Pred] = np.float32(DC_predict)
        #non-zero values cannot occur
        Analysis[self.DC_Pred][Analysis[self.DC_Pred]<0] = 0


        print('Predictions Complete') 
        #input physical limitations to components. the 0.000810714 is a conversion from m3 to af
        Analysis[self.GW_Pred].loc[Analysis[self.GW_Pred]<0] =0
        Analysis[self.GW_Pred].loc[Analysis[self.GW_Pred]>11.038412*8.10714] = 11.038412*8.10714
        Analysis[self.GW_Pred].loc['2021-7-10':'2021-8-30'][Analysis[self.GW_Pred]<11.038416*8.10714]=11.038416*8.10714

        Analysis[self.DC_Pred].loc[Analysis[self.DC_Pred]<0.05*8.10714] =0.05*8.10714
        Analysis[self.MDell_Pred].loc[Analysis[self.MDell_Pred]<25] =25
        Analysis[self.LDell_Pred].loc[Analysis[self.LDell_Pred]<10] =10


        #calculate 5day rolling means
        Analysis[self.DC_Pred_Rol] = Analysis[self.DC_Pred].rolling(5).mean()

        Analysis[self.DC_Pred_Rol] = Analysis[self.DC_Pred_Rol].interpolate(method='linear',
                                                       limit_direction='backward', 
                                                       limit=5)
        Analysis[self.GW_Pred_Rol] = Analysis[self.GW_Pred].rolling(5).mean()

        Analysis[self.GW_Pred_Rol] = Analysis[self.GW_Pred_Rol].interpolate(method='linear',
                                                       limit_direction='backward', 
                                                       limit=5)
        Analysis[self.MDell_Pred_Rol] = Analysis[self.MDell_Pred].rolling(5).mean()

        Analysis[self.MDell_Pred_Rol] = Analysis[self.MDell_Pred_Rol].interpolate(method='linear',
                                                       limit_direction='backward', 
                                                       limit=5)
        Analysis[self.LDell_Pred_Rol] = Analysis[self.LDell_Pred].rolling(5).mean()

        Analysis[self.LDell_Pred_Rol] = Analysis[self.LDell_Pred_Rol].interpolate(method='linear',
                                                       limit_direction='backward', 
                                                       limit=5)


        self.Analysis = Analysis
        self.HistoricalAnalysis()
        self.RRV_Assessment()
        print('Plotting results for visual analysis:')
        self.WSM_Pred_RRV_Plot()
 


 #A function to calculate the daily mean values for each water system component 
    def DailyMean(self,component, month, yrs, days, monthnumber, inputyr):

        Daylist = defaultdict(list)
        DayFrame= defaultdict(list)
        timecol = ['Year', 'Month' , 'Day']

        for i in days:
            Daylist[month+ str(i)]= []
            DayFrame[month + str(i)] = pd.DataFrame(yrs, columns=['Year'])


        for i in yrs:
            for j in days:
                Daylist[month+str(j)].append(self.Histyrs.loc[str(i)+'-'+ monthnumber +'-'+str(j)][component])
                DayFrame[month+str(j)]['Day']=j
                DayFrame[month+str(j)]['Month'] = int(monthnumber)

        for i in DayFrame:
            DayFrame[i][component] = Daylist[i]

        histcomponent = 'Hist_Mean_' + component

        for i in DayFrame:
            DayFrame[i][histcomponent]= np.mean(DayFrame[i][component])
            del DayFrame[i][component]

            ##put into year of choice
            DayFrame[i]['Year']=inputyr
            #create the date for input into figure DF
            DayFrame[i].insert(loc=0, column='Date', value=pd.to_datetime(DayFrame[i][['Year', 'Month', 'Day']]))
            DayFrame[i] = DayFrame[i].drop(columns = timecol)
            DayFrame[i]=DayFrame[i].set_index('Date')
            DayFrame[i]=DayFrame[i].iloc[0]
            DayFrame[i] = pd.DataFrame(DayFrame[i]).T


        return DayFrame      
        
        
 #Perform a historical analysis of each WSC to compare performance of current scenario       
    def HistoricalAnalysis(self):
        
        print('Calculating historical water system component means to create baseline for comparison with prediction')
        
        targets = ['SLCDPU_GW', 'Mtn_Dell_Percent_Full', 'LittleDell_Percent_Full','SLCDPU_DC_Water_Use']


        pbar = ProgressBar()
        for component in pbar(targets):

            histcomponent = 'Hist_Mean_' + component
            predcomponent = component+'_Pred'


            #Use historical data, prior to WY2021
            Histyrs=self.Hist_targs.copy()
            Histyrs = Histyrs[:"2020-10-31"]


            #Select time of importance 2021, 2022
            self.Analysis = self.Analysis[self.Analysis.index.year.isin([self.test_yr])].copy()

            #remove months that are not if interst in historical dataset
            self.Histyrs = Histyrs[~Histyrs.index.month.isin([1,2,3,11,12])]

           
            '''
            Using the historical daily DC water usage, Find the mean daily DC usage and add it to the 
            Main DF to compare 2021 and 2022 water usage.
            '''
            yrs = np.arange(2001,2021,1)
            Aprdays = np.arange(1,31,1)
            Maydays = np.arange(1,32,1)
            Jundays = np.arange(1,31,1)
            Juldays = np.arange(1,32,1)
            Augdays = np.arange(1,32,1)
            Sepdays = np.arange(1,31,1)
            Octdays = np.arange(1,32,1)

            #Set up DF for mean daily DC water usage for WY 2021
            Apr = self.DailyMean(component,'Apr', yrs, Aprdays, '04', self.test_yr)
            May = self.DailyMean(component,'May', yrs, Maydays, '05', self.test_yr)
            Jun = self.DailyMean(component,'Jun', yrs, Jundays, '06', self.test_yr)
            Jul = self.DailyMean(component,'Jul', yrs, Juldays, '07', self.test_yr)
            Aug = self.DailyMean(component,'Aug', yrs, Augdays, '08', self.test_yr)
            Sep = self.DailyMean(component,'Sep', yrs, Sepdays, '09', self.test_yr)
            Oct = self.DailyMean(component,'Oct', yrs, Octdays, '10', self.test_yr)

            DC_Mean = pd.DataFrame()
            for i in Apr:
                DC_Mean = DC_Mean.append(Apr[i])
            for i in May:
                DC_Mean = DC_Mean.append(May[i])
            for i in Jun:
                DC_Mean = DC_Mean.append(Jun[i])
            for i in Jul:
                DC_Mean = DC_Mean.append(Jul[i])
            for i in Aug:
                DC_Mean = DC_Mean.append(Aug[i])
            for i in Sep:
                DC_Mean = DC_Mean.append(Sep[i])
            for i in Oct:
                DC_Mean = DC_Mean.append(Oct[i])

            #create an empty column for mean delivery
            self.Analysis[histcomponent] = 0

            #Update the Output2021 with historical period daily DC usage
            self.Analysis.update(DC_Mean)

            predcomponent_diff = predcomponent+'_diff'

            res = ['Mtn_Dell_Percent_Full', 'LittleDell_Percent_Full', 'Mtn_Dell_Percent_Full_Pred',
                   'LittleDell_Percent_Full','LittleDell_Percent_Full_Pred']

            #we want to mark the reservoirs at a concern if they go below a certain level
            if component in res:
                if component == 'Mtn_Dell_Percent_Full':
                    #Dead pool for mtn dell is ~25, mark as vulnerable when it gets to 35%
                    self.Analysis[predcomponent_diff] = self.MDell_Thresh-self.Analysis[predcomponent]

                if component == 'LittleDell_Percent_Full':
                    #Dead pool for lil dell is ~5%, mark as vulnerable when it gets to 15%
                    self.Analysis[predcomponent_diff] = self.LDell_Thresh-self.Analysis[predcomponent]
            else:  
                self.Analysis[predcomponent_diff] = self.Analysis[predcomponent]-self.Analysis[histcomponent]

        self.Prediction_Comparative_Analysis()


    #Create historical RRV Analysis to define historical RRV thresholds to compare predictions with
    def Prediction_Comparative_Analysis(self):
        print('Processing predictions and historical means for comparative performance analysis.')

        #Find the historical daily values for the water system. 
        #This creates a baseline to gage reliability, resilience, vulnerability
        self.years =  [2001,2002,2003,2004, 2005, 2006, 2007,2008, 2009, 2010,
                  2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,2020]


        #Determine the maximum historical system severity
        df = self.Analysis.copy()
        #Daily2021  = ForecastDataPrep(Analysis, Hist_targs, 2021)


        #Get the historical mean DC deliverity values for one year
        Hist_Mean_MDell = list(df['Hist_Mean_Mtn_Dell_Percent_Full'].copy())
        Hist_Mean_LDell = list(df['Hist_Mean_LittleDell_Percent_Full'].copy())
        Hist_Mean_GW = list(df['Hist_Mean_SLCDPU_GW'].copy())
        Hist_Mean_DC = list(df['Hist_Mean_SLCDPU_DC_Water_Use'].copy())


        #Get the reference perid simulation results
        SimDF = self.Sim[self.scenario].copy()
        SimDF.index = SimDF['Time']
        del SimDF['Time']
        #Convert GS output from AF to M3


        #Select the first 20 years
        Hist = pd.DataFrame(columns = SimDF.columns)
        for y in self.years :
            Hist = Hist.append(SimDF.loc[str(y)+'-4-01':str(y)+'-10-30'])


        #Make the data to input into long term TS
        yearlen = len(Hist_Mean_MDell)
        Hist_Mean_MDell = Hist_Mean_MDell*20
        Hist_Mean_LDell = Hist_Mean_LDell*20
        Hist_Mean_GW = Hist_Mean_GW*20
        Hist_Mean_DC = Hist_Mean_DC*20


        #Hist_Mean_DC = Oct_Dec_Hist_Mean_DC+Hist_Mean_DC
        Hist['Hist_Mean_Mtn_Dell_Percent_Full'] = Hist_Mean_MDell
        Hist['Hist_Mean_LittleDell_Percent_Full'] = Hist_Mean_LDell
        Hist['Hist_Mean_SLCDPU_GW'] = Hist_Mean_GW
        Hist['Hist_Mean_SLCDPU_DC_Water_Use'] = Hist_Mean_DC

        #Find above/below specific reservoir levels
        Hist['Mtn_Dell_Percent_Full_diff'] = self.MDell_Thresh-Hist['Mtn_Dell_Percent_Full']
        Hist['LittleDell_Percent_Full_diff'] = self.LDell_Thresh-Hist['LittleDell_Percent_Full']

        #Find above/below historical DC/GW and
        Hist['SLCDPU_GW_diff'] = Hist['SLCDPU_GW']-Hist['Hist_Mean_SLCDPU_GW']
        Hist['SLCDPU_DC_Water_Use_diff'] = Hist['SLCDPU_DC_Water_Use']-Hist['Hist_Mean_SLCDPU_DC_Water_Use']


        for i in np.arange(0,len(Hist),1):
            if Hist['Mtn_Dell_Percent_Full_diff'].iloc[i] <1:
                Hist['Mtn_Dell_Percent_Full_diff'].iloc[i] = 0

            if Hist['LittleDell_Percent_Full_diff'].iloc[i] <1:
                Hist['LittleDell_Percent_Full_diff'].iloc[i] = 0

            if Hist['SLCDPU_GW_diff'].iloc[i] <1:
                Hist['SLCDPU_GW_diff'].iloc[i] = 0

            if Hist['SLCDPU_DC_Water_Use_diff'].iloc[i] <1:
                Hist['SLCDPU_DC_Water_Use_diff'].iloc[i] = 0




        Historical_Max_Daily_MDell = max(Hist['Mtn_Dell_Percent_Full_diff'])
        Historical_Max_Daily_LDell = max(Hist['LittleDell_Percent_Full_diff'])
        Historical_Max_Daily_GW = max(Hist['SLCDPU_GW_diff'])
        Historical_Max_Daily_DC = max(Hist['SLCDPU_DC_Water_Use_diff'])


        self.Hist, self.Historical_Max_Daily_MDell, self.Historical_Max_Daily_LDell = Hist, Historical_Max_Daily_MDell, Historical_Max_Daily_LDell
        self.Historical_Max_Daily_GW, self.Historical_Max_Daily_DC =  Historical_Max_Daily_GW, Historical_Max_Daily_DC

    



    def RRV_Assessment(self):
        print('Initiating water system component RRV analysis.')
        #Make a dictionary to store each targets RRV information
        Target_RRV = [ 'SLCDPU_GW', 'SLCDPU_DC_Water_Use',
               'Mtn_Dell_Percent_Full', 'LittleDell_Percent_Full']
        Target_RRV= dict.fromkeys(Target_RRV)


        self.RRV_DF =pd.DataFrame(columns =['Model', 'Climate', 'Target', 'Reliability',
            'Resilience', 'Vulnerability', 'MaxSeverity', 'Maximum_Severity'])


        #Find the historical RRV, the jenks breaks will use this
        self.TargetRRV(self.Hist,'Hist', 'Mtn_Dell_Percent_Full',
                                          self.Historical_Max_Daily_MDell, self.years)
        self.TargetRRV(self.Hist,'Hist', 'LittleDell_Percent_Full', 
                                          self.Historical_Max_Daily_LDell, self.years)
        self.TargetRRV(self.Hist,'Hist', 'SLCDPU_GW', 
                                          self.Historical_Max_Daily_GW, self.years)
        self.TargetRRV(self.Hist,'Hist', 'SLCDPU_DC_Water_Use',
                                          self.Historical_Max_Daily_DC, self.years)


        #XGB-WSM
        self.TargetRRV(self.Analysis, 'XGB_WSM','Mtn_Dell_Percent_Full_Pred',  self.Historical_Max_Daily_MDell, [self.test_yr])

        self.TargetRRV(self.Analysis, 'XGB_WSM','LittleDell_Percent_Full_Pred',  self.Historical_Max_Daily_LDell, [self.test_yr])

        self.TargetRRV(self.Analysis, 'XGB_WSM', 'SLCDPU_GW_Pred',  self.Historical_Max_Daily_GW, [self.test_yr])

        self.TargetRRV(self.Analysis, 'XGB_WSM', 'SLCDPU_DC_Water_Use_Pred',self.Historical_Max_Daily_DC, [self.test_yr])

        print('Setting up an RRV dataframe and calculating each water system component RRV')
        print('Finalizing analysis and placing into Jenks classification categories.')
        for target in Target_RRV:
            Target_RRV[target] = self.RRVanalysis(target)

        #Make Target_RRV a global variable
        self.Target_RRV =Target_RRV



    def TargetRRV(self, DF, Sim, Target,  Max, years):

        preds = ['SLCDPU_GW_Pred', 'SLCDPU_DC_Water_Use_Pred',
           'Mtn_Dell_Percent_Full_Pred', 'LittleDell_Percent_Full_Pred']

        RRV_Data_D =pd.DataFrame(columns = ['SLCDPU_Prod_Demands', 'SLCDPU_Population', 'BCC_Streamflow',
           'LCC_Streamflow', 'CC_Streamflow', 'Dell_Streamflow',
           'Lambs_Streamflow', 'SLCDPU_GW', 'SLCDPU_DC_Water_Use',
           'Mtn_Dell_Percent_Full', 'LittleDell_Percent_Full',
           'Hist_Mean_Mtn_Dell_Percent_Full', 'Hist_Mean_LittleDell_Percent_Full',
           'Hist_Mean_SLCDPU_GW', 'Hist_Mean_SLCDPU_DC_Water_Use', 'Mtn_Dell_Percent_Full_diff',
           'LittleDell_Percent_Full_diff', 'SLCDPU_GW_diff', 'SLCDPU_DC_Water_Use_diff', 'Clim',
           Target+'_Zt', Target+'_Wt',Target+'_WSCI_s', Target+'_Sev',
           Target+'_Vul'])


        Extra_Targ = Target+'_diff'
       
        for y in years:
            DF2 = DF.loc[str(y)+'-04-01':str(y)+'-10-31'].copy()
            self.RRV(DF2,Extra_Targ, Target,  Max,y)

            if Target in preds:
                Target = Target[:-5]
            RRVass = list([Sim,self.scenario, Target, self.Rel, self.Res, self.Vul, self.Max_Severity, self.MaxSevNorm])
            self.RRV_DF.loc[len(self.RRV_DF)] = RRVass



    #we need to calculate the RRV metrics
    def RRV(self, DF2, Extra_target, target, maxseverity, yr):
        
        preds = ['SLCDPU_GW_Pred', 'SLCDPU_DC_Water_Use_Pred',
           'Mtn_Dell_Percent_Full_Pred', 'LittleDell_Percent_Full_Pred']

        if target in preds:
            hist_target = 'Hist_Mean_'+ target[:-5]
        else:
            hist_target = 'Hist_Mean_'+ target
        df = DF2.copy()
        df['Clim'] = self.scenario


        #period of interest is from April to October

        df = df.loc[str(yr)+'-04-01':str(yr)+'-10-31']


        #length of study period
        T = len(df)
        #make sure ExtraDC is never less than 0
        for i in np.arange(1,T,1):
            if df[Extra_target].iloc[i] < 0:
                    df[Extra_target].iloc[i] = 0

        '''
        Reliability
        Reliability = sum of timesteps Zt/T
        Zt is 0 if the target exceeds (U) the historical average and 1 if it does not (S)
        '''
        Zt = target+'_Zt'
        df[Zt] = 1
        for i in np.arange(0,T,1):
            if df[Extra_target].iloc[i] > 1:
                    df[Zt].iloc[i] = 0
            if df[Extra_target].iloc[i] < 1:
                    df[Extra_target].iloc[i] = 0

        Rel = df[Zt].sum()/T

        '''
        Resilience
        Resilience = sum of timesteps Wt/(T-sum(Zt))
        Wt is 1 if Xt is U and Xt+1 is S
        '''
        Wt = target+'_Wt'
        df[Wt]=0
        for i in np.arange(1,T,1):
            if df[Zt].iloc[i-1] == 0 and df[Zt].iloc[i] == 1:
                df[Wt].iloc[i] = 1
        #To get in days do 1/Res        
        Res = 1/((1+df[Wt].iloc[0:T-1].sum())/(T+1-df[Zt].sum()))

        '''
        Vulnerability
        We use Exposure and severity to determine Vulnerability
        Exposure DCwater requests > hist ave, WRI_s) is an index from 0-1, WRI_s =1- WR_s/WR_h
        Severity is the amount of ExtraDC water, and then normalized based on the
        largest value to provide values from 0-1
        '''
        #Exposure

        WSCI_s = target+ '_WSCI_s'
        df[WSCI_s] = df[target]/(df[hist_target]+1)
        for i in np.arange(0,T,1):
            if df[WSCI_s].iloc[i] > 1:
                df[WSCI_s].iloc[i] = 1
        #This is average exposure
        Exp = df[WSCI_s].sum()/T

        #Severity
        Max_Severity = df[Extra_target].max()
        #if MaxSeverity == 0:
         #   MaxSeverity = 1
        #This is the maximum found for all simulations 
        MaxSeverity = maxseverity
        Severity = target+'_Sev'
        df[Severity] = df[Extra_target]/MaxSeverity
        #This is average severity
        Sev = df[Severity].sum()/(T+1-df[Zt].sum())
        MaxSevSI = df[Severity].max()*MaxSeverity
        MaxSevNorm = df[Severity].max()

        Vulnerability = target + '_Vul'
        df[Vulnerability] = (0.5*df[WSCI_s])+(0.5*df[Severity])
        #Vulerability = Exposure +Severity
        Vul = (0.5*Exp) + (0.5*Sev)

        self.Rel, self.Res, self.Vul, self.df, self.Max_Severity, self.MaxSevSI, self.MaxSevNorm = Rel, Res, Vul, df, Max_Severity, MaxSevSI, MaxSevNorm






    def RRVanalysis(self, Target):

        #Get the historical RRV for each target
        Breaks_Data = self.RRV_DF.loc[(self.RRV_DF['Model'] == 'Hist') & (self.RRV_DF['Target'] == Target)]

        Cat_Data = self.RRV_DF.loc[(self.RRV_DF['Target'] == Target)]

        #Find the natural breaks in the RRV
        #The eval data set has values greater than the historical and are identified as Nan in the 
        #eval dataframe. These values will be marked as extreme
        VBreaks = jenkspy.jenks_breaks(Breaks_Data['Vulnerability'], nb_class=3)
        VBreaks[0] = 0.0

        Cat_Data['Jenks_Vul'] = pd.cut(Cat_Data['Vulnerability'],
                                bins=VBreaks,
                                labels=['low', 'medium', 'high'],
                                            include_lowest=True)
        self.VBreaks = [ np.round(v,2) for v in VBreaks ]
     #   print(Target, ' Vulnerability Breaks')
      #  print('Low, Medium, High: ', VBreaks)

        SBreaks = jenkspy.jenks_breaks(Breaks_Data['Maximum_Severity'], nb_class=3)
        SBreaks[0] = 0.0

        Cat_Data['Jenks_Sev'] = pd.cut(Cat_Data['Maximum_Severity'],
                                bins=SBreaks,
                                labels=[ 'low', 'medium', 'high'],
                                            include_lowest=True)   

        self.SBreaks = [np.round(s,2) for s in SBreaks]
       # print(Target, ' Severity Breaks')
       # print('Low, Medium, High: ' ,SBreaks)

        return Cat_Data






    def WSM_Pred_RRV_Plot(self):
        print('Using the ', self.MDell_Thresh,'% & ', self.LDell_Thresh, '% capacities for Mountain & Little Dell Reservoirs')
        print('and the historical daily mean municipal groundwater withdrawal and Deer Creek Reservoir use:')
        print('\033[0;32;48m Green \033[0;0m shading suggests satisfactory conditions.')
        print('\033[0;31;48m Red \033[0;0m shading suggests unsatisfactory conditions.')
        print( ' ')
        print('Total volume of Groundwater withdrawal is ', round(sum(self.Analysis[self.GW_Pred])), 'acre-feet')
        print('Total volume of Deer Creek water requests is ', round(sum(self.Analysis[self.DC_Pred])), 'acre-feet')
        
        
        #Set up the target labels
        #Mountain Dell
        MDell_Hist = 'Hist_Mean_'+ self.MDell
    
        #Little Dell
        LDell_Hist = 'Hist_Mean_'+ self.LDell

        #GW
        GW_Hist = 'Hist_Mean_'+ self.GW

        #GW
        DC_Hist = 'Hist_Mean_'+ self.DC
        
        af_to_MGD = 271328
        
        if self.units == 'MGD':
            self.Analysis[self.GW_Pred] = self.Analysis[self.GW_Pred]*af_to_MGD
            self.Analysis[self.DC_Pred] = self.Analysis[self.DC_Pred]*af_to_MGD
            self.Analysis[GW_Hist] = self.Analysis[GW_Hist]*af_to_MGD
            self.Analysis[DC_Hist] = self.Analysis[DC_Hist]*af_to_MGD
            

   
        #Define max values
        max_LDell = max(max(self.Analysis[self.LDell_Pred]),  max(self.Analysis[LDell_Hist]))*1.4
        max_MDell = max(max(self.Analysis[self.MDell_Pred]), max(self.Analysis[MDell_Hist]))*1.4
        max_GW = max(max(self.Analysis[self.GW_Pred]), max(self.Analysis[GW_Hist]))*1.4
        max_DC = max(max(self.Analysis[self.DC_Pred]), max(self.Analysis[DC_Hist]))*1.4


        All_RRV = pd.DataFrame()
        for targs in self.Target_RRV:
            targ = pd.DataFrame(self.Target_RRV[targs][-6:])
            All_RRV = All_RRV.append(targ)

        #Sort DF to make plots more comprehendable
        All_RRV.sort_values(['Climate', 'Target'], ascending=[True, True], inplace=True)
        All_RRV = All_RRV.reset_index()
        self.All_RRV = All_RRV

        components ={'Mtn_Dell_Percent_Full':pd.DataFrame(All_RRV.loc[(All_RRV['Model'] == 'XGB_WSM') &  (All_RRV['Target']=='Mtn_Dell_Percent_Full')].copy()),
                     'LittleDell_Percent_Full':pd.DataFrame(All_RRV.loc[(All_RRV['Model'] == 'XGB_WSM') &  (All_RRV['Target']=='LittleDell_Percent_Full')].copy()),
                     'SLCDPU_GW' : pd.DataFrame(All_RRV.loc[(All_RRV['Model'] == 'XGB_WSM') &  (All_RRV['Target']=='SLCDPU_GW')].copy()),
                     'SLCDPU_DC_Water_Use': pd.DataFrame(All_RRV.loc[(All_RRV['Model'] == 'XGB_WSM') &  (All_RRV['Target']=='SLCDPU_DC_Water_Use')].copy())
                    }

        delcols = ['index', 'Climate', 'Target', 'Resilience', 'MaxSeverity', 'Jenks_Vul', 'Jenks_Sev' ]
        for comp in components:
            components[comp] = components[comp].drop(delcols, axis = 1)
            components[comp] = components[comp].set_index('Model')
            components[comp] = components[comp].T

        self.components = components
       
        # better control over ax
        fig, ax = plt.subplots(4, 2)
        fig.set_size_inches(12,12)
        plt.subplots_adjust(wspace = 0.25, hspace = 0.3)
        labelsize = 12
        width = 0.7
        colors = [ 'blue']
        self.Analysis['MDell_Thresh'] =self.MDell_Thresh
        self.Analysis['LDell_Thresh'] =self.LDell_Thresh

        
        
        
        #PLot Mountain Dell
        self.Analysis.plot(y = self.MDell_Pred , ax=ax[0,0], color = 'blue', label = 'Predicted')
        self.Analysis.plot(y = MDell_Hist , ax=ax[0,0], color = 'black', label = 'Historical Mean Reservoir Level')
        ax[0,0].axhline(y = self.MDell_Thresh, color = 'red', label = 'Unsatifactory Conditions Threshold')
        ax[0,0].fill_between(self.Analysis.index.values, self.Analysis[self.MDell_Pred], self.Analysis['MDell_Thresh'], where=self.Analysis[self.MDell_Pred] >= self.Analysis['MDell_Thresh'],
                facecolor='green', alpha=0.2, interpolate=True)
        ax[0,0].fill_between(self.Analysis.index.values, self.Analysis[self.MDell_Pred], self.Analysis['MDell_Thresh'], where=self.Analysis[self.MDell_Pred] < self.Analysis['MDell_Thresh'],
                facecolor='red', alpha=0.2, interpolate=True)

        
        ax[0,0].set_xlabel(' ', size = labelsize)
        ax[0,0].set_ylabel('Mountain Dell Reservoir \n Level (%)', size = labelsize)
        ax[0,0].set_ylim(0,100)
        ax[0,0].legend(bbox_to_anchor=(1,1.5), loc="upper center", ncol = 2, fontsize = 14)
        ax[0,0].xaxis.set_major_locator(MonthLocator())
        ax[0,0].xaxis.set_major_formatter(DateFormatter('%b'))
        ax[0,0].tick_params(axis='both', which='major', labelsize=8)



        #Mountain Dell 
        components[self.MDell].plot.bar(width=width, color=colors, legend=False, ax = ax[0,1])
        ax[0,1].set_ylim(0,1)
        ax[0,1].axes.xaxis.set_ticklabels([])


         #PLot Little Dell
        self.Analysis.plot(y = self.LDell_Pred , ax=ax[1,0], color = 'blue', label = 'Predicted')
        self.Analysis.plot(y = LDell_Hist , ax=ax[1,0], color = 'black', label = 'Historical Mean Reservoir Level')
        ax[1,0].fill_between(self.Analysis.index.values, self.Analysis[self.LDell_Pred], self.Analysis['LDell_Thresh'], where=self.Analysis[self.LDell_Pred] >= self.Analysis['LDell_Thresh'],
                facecolor='green', alpha=0.2, interpolate=True)
        ax[1,0].fill_between(self.Analysis.index.values, self.Analysis[self.LDell_Pred], self.Analysis['LDell_Thresh'], where=self.Analysis[self.LDell_Pred] < self.Analysis['LDell_Thresh'],
                facecolor='red', alpha=0.2, interpolate=True)

        ax[1,0].axhline(y = self.LDell_Thresh, color = 'red', label = 'Unsatifactory Conditions Threshold')
        ax[1,0].set_xlabel('  ', size = labelsize)
        ax[1,0].set_ylabel('Little Dell Reservoir \n Level (%)', size = labelsize)
        ax[1,0].set_ylim(0,100)
        ax[1,0].legend().set_visible(False)
        ax[1,0].xaxis.set_major_locator(MonthLocator())
        ax[1,0].xaxis.set_major_formatter(DateFormatter('%b'))
        ax[1,0].tick_params(axis='both', which='major', labelsize=8)


        #Little Dell 
        components[self.LDell].plot.bar(width=width, color=colors, legend=False, ax = ax[1,1])
        ax[1,1].set_ylim(0,1)
        ax[1,1].axes.xaxis.set_ticklabels([])


        #PLot GW
        self.Analysis.plot(y = self.GW_Pred , ax=ax[2,0], color = 'blue', label = 'Predicted')
        self.Analysis.plot(y = GW_Hist , ax=ax[2,0], color = 'red', label = 'Historical')
        ax[2,0].fill_between(self.Analysis.index.values, self.Analysis[self.GW_Pred], self.Analysis[GW_Hist], where=self.Analysis[self.GW_Pred] >= self.Analysis[GW_Hist],
                facecolor='red', alpha=0.2, interpolate=True)
        ax[2,0].fill_between(self.Analysis.index.values, self.Analysis[self.GW_Pred], self.Analysis[GW_Hist], where=self.Analysis[self.GW_Pred] < self.Analysis[GW_Hist],
                facecolor='green', alpha=0.2, interpolate=True)


        ax[2,0].set_xlabel(' ', size = labelsize)
        ax[2,0].set_ylabel('Groundwater Withdrawal \n ('+ self.units+')', size = labelsize)
        ax[2,0].set_ylim(0,max_GW)
        ax[2,0].legend().set_visible(False)
        ax[2,0].xaxis.set_major_locator(MonthLocator())
        ax[2,0].xaxis.set_major_formatter(DateFormatter('%b'))
        ax[2,0].tick_params(axis='both', which='major', labelsize=8)


            #GW
        components[self.GW].plot.bar(width=width, color=colors, legend=False, ax = ax[2,1])
        ax[2,1].set_ylim(0,1)
        ax[2,1].axes.xaxis.set_ticklabels([])


        #PLot DC
        self.Analysis.plot(y = self.DC_Pred , ax=ax[3,0], color = 'blue', label = 'Predicted')
        self.Analysis.plot(y = DC_Hist , ax=ax[3,0], color = 'red', label = 'Historical')
        ax[3,0].fill_between(self.Analysis.index.values, self.Analysis[self.DC_Pred], self.Analysis[DC_Hist], where=self.Analysis[self.DC_Pred] >= self.Analysis[DC_Hist],
                facecolor='red', alpha=0.2, interpolate=True)
        ax[3,0].fill_between(self.Analysis.index.values, self.Analysis[self.DC_Pred], self.Analysis[DC_Hist], where=self.Analysis[self.DC_Pred] < self.Analysis[DC_Hist],
                facecolor='green', alpha=0.2, interpolate=True)

        ax[3,0].set_xlabel('Time ', size = labelsize)
        ax[3,0].set_ylabel('Deer Creek Reservoir \n ('+ self.units+')', size = labelsize)
        ax[3,0].set_ylim(0,max_DC)
        ax[3,0].legend().set_visible(False)
        ax[3,0].xaxis.set_major_locator(MonthLocator())
        ax[3,0].xaxis.set_major_formatter(DateFormatter('%b'))
        ax[3,0].tick_params(axis='both', which='major', labelsize=8)

        #DC 
        components[self.DC].plot.bar(width=width, color=colors, legend=False, ax = ax[3,1])
        ax[3,1].set_ylim(0,1)
        ax[3,1].set_xticklabels(["Reliability", "Vulnerability", "Max Severity"], rotation=45)





        #plt.title('Production Simulations', size = labelsize+2)
        fig.savefig('Figures/'+self.scenario+ '_Analysis.pdf')

        plt.show()













