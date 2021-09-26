#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:10:45 2021

@author: adityakumar.pal
"""


#relevant imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
import seaborn as sns
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.preprocessing import LabelEncoder
import shap 
import sage
import numpy as np
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
import lime
import lime.lime_tabular
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import sklearn.inspection.partial_dependence as partial_dependence
from sklearn.impute import SimpleImputer


#define variables

dir = os.getcwd()
filename='online_shoppers_intention.csv'


def read_file(file):
    '''
    

    Parameters
    ----------
    file : filename 

    Returns
    -------
    df : data frame of raw data

    '''
    
    try:
        df=pd.read_csv(os.path.join(dir,file))
        print("File read from path", os.path.join(dir,file))
        print("File read with shape", df.shape)
        return df
    except:
        print("file not found")
    
data = read_file(filename)

def missing_treatment(df):
    '''
    

    Parameters
    ----------
    df : raw data 

    Returns
    -------
    df : raw data with missing value treatment

    '''
    
    print(df.isnull().sum())
    if any(df.isnull()):
        print("Data fed has missing values, Will impute with mean")
        df = df.fillna(df.mean())
        return df
    print("no nulls")
    return df

data_treated = missing_treatment(data)

def label_encoder(df, response_var):
    '''
    

    Parameters
    ----------
    df : raw treated data
    response_var : string input for response variable name

    Returns
    -------
    numeric label encoded data

    '''
    
    
    le=LabelEncoder()
    df[response_var] = le.fit_transform(df[response_var])
    print("Distribution post encoding is ", df[response_var].value_counts())
    return df
    
data_encoded = label_encoder(data_treated,'Revenue')

def modelling(df, response_var):
    '''
    

    Parameters
    ----------
    df : encoded raw data
    response_var : string input for response variable name

    Returns
    -------
    model : the random forest model function
    train_X : train data
    test_X : test data
    train_y : train labels
    test_y : test labels
    df : raw data with dummy variables

    '''


    df = pd.get_dummies(df)
    x = df
    x = x.drop([response_var], axis = 1)
    y = df[response_var]
    
    train_X, test_X, train_y, test_y = train_test_split(x, y,test_size=0.1, random_state=1)
    
    model = RandomForestClassifier(n_estimators=100,
                                  random_state=0).fit(train_X, train_y)


    return model, train_X, test_X, train_y, test_y, df

model, train_X, test_X, train_y, test_y, data_model = modelling(data_encoded, "Revenue")

def feature_importance(model,df):
    '''

    Parameters
    ----------
    model :the model function which has the default random forest model
    
    df : data frame to feature names for importance plots

    Returns
    -------
    importance :random forest importance scores

    '''
    importance = model.feature_importances_
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
        
    # plot feature importance
    plt.figure(figsize=(16,8))
    plt.bar(df.iloc[:,:-1].columns, [i*100 for i in importance],color='green')
    plt.title('Column wise Feature Importance score in %')
    plt.xlabel('Feature Name')
    plt.ylabel('Importance score in %')
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()
    
    return importance

importance = feature_importance(model, data_model)

def permutation_importance(model,test_x, test_y):
    '''
    

    Parameters
    ----------
    model : the model function which has the default random forest model
    
    test_x : test data to generate permutation scores
    test_y : the test data labels

    Returns
    -------
    permutation scores

    '''
    
    perm = PermutationImportance(model, random_state=1).fit(test_x, test_y)
    eli5.show_weights(perm, feature_names = test_x.columns.tolist())
    
permutation = permutation_importance(model,test_X,test_y)

def shap_explainer(model, test_x):
    '''
    Parameters
    ----------
    model : the model function which has the default random forest model
    test_x : test data to geenrate shap scores

    Returns
    -------
    SHAP summary plots

    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_x)
    
    shap.summary_plot(shap_values[1], test_x)

shap_explainer(model, test_X)

def customer_analysis(model, customer):
    '''
    Function to compute SHAP force plots
    
    Parameters
    ----------
    model : the model function which has the default random forest model
    customer: a single data point or observation from the raw data
        
    Returns
    -------
    Shap force plots
    '''
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(customer)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], customer)

customers = test_X.iloc[1,:].astype(float)

force = customer_analysis(model, customers)


def SAGE_computation(data):
    '''
    Function to compute sage plots and scores
    
    Parameters
    ----------
    data : the raw treated data which has encoded labels
        
    Returns
    -------
    SAGE plots

    '''
    
    feature_names = data.columns.tolist()[:-1]
    categorical_columns = ['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend']
    categorical_inds = [feature_names.index(col) for col in categorical_columns]
    
    train, test = train_test_split(data.values, test_size=int(0.1 * len(data.values)), random_state=0)
    train, val = train_test_split(train, test_size=int(0.1 * len(data.values)), random_state=0)

    
    Y_train = train[:, -1].copy().astype(int)
    Y_val = val[:, -1].copy().astype(int)
    Y_test = test[:, -1].copy().astype(int)
    train = train[:, :-1].copy()
    val = val[:, :-1].copy()
    test = test[:, :-1].copy()
    
    
    model = CatBoostClassifier(iterations=50,
                           learning_rate=0.3,
                           depth=3)

    model = model.fit(train, Y_train, categorical_inds, eval_set=(val, Y_val),
                  verbose=False)

    p = np.array([np.sum(Y_train == i) for i in np.unique(Y_train)]) / len(Y_train)
    base_ce = log_loss(Y_test.astype(int), p[np.newaxis].repeat(len(test), 0))
    ce = log_loss(Y_test.astype(int), model.predict_proba(test))
    
    print('Base rate cross entropy = {:.3f}'.format(base_ce))
    print('Model cross entropy = {:.3f}'.format(ce))

    
    imputer = sage.MarginalImputer(model, train[:512])
    estimator = sage.PermutationEstimator(imputer, 'cross entropy')

    sage_values = estimator(test, Y_test)

    sage_values.plot(feature_names, title='Feature Importance (Marginal Sampling)')


sage = SAGE_computation(data_encoded)

def lime_computation(model, train):
    '''
    

    Parameters
    ----------
    model : random forest model object
    train : train data

    Returns
    -------
    explainer : LIME explainer
    predict_fn_rf : the model predict function

    '''

    predict_fn_rf = lambda x: model.predict_proba(x).astype(float)
    X = train.values
    explainer = lime.lime_tabular.LimeTabularExplainer(X,feature_names =       train.columns,class_names=['1','0'],kernel_width=5)

    return explainer,predict_fn_rf

test_X = test_X.reset_index(drop=True)
choosen_instance = test_X.iloc[0].values
explainer ,predict_fn_rf= lime_computation(model, train_X)
exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)
exp.show_in_notebook(show_all=False)


choosen_instance = test_X.iloc[21].values
explainer ,predict_fn_rf= lime_computation(model, train_X)
exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)
exp.show_in_notebook(show_all=False)

def partial_plots(train, feature_list):
    clf = GradientBoostingClassifier()
    my_imputer = SimpleImputer()
    imputed_x = my_imputer.fit_transform(train)

    clf.fit(imputed_x,y)
    plots = plot_partial_dependence(clf, features=feature_list, X=imputed_x, 
                                        feature_names=feature_list, grid_resolution=4)

partial_plots(train_X,['ExitRates', 'PageValues'])