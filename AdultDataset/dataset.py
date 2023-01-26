import pandas as pd
import numpy as np
import scipy

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from copy import deepcopy

from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt


TARGET_NAME = 'income'
continuous_feature_names = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_feature_names = ['education', 'educational-num', 'gender', 'marital-status',
       'native-country', 'occupation', 'race', 'relationship', 'workclass']


def get_random_dict(feature_name, int_to_str=False):

    df = pd.read_csv('data/adult.csv')
    del df['fnlwgt']
    df = df.replace({'<=50K': 0, '>50K': 1})

    temp_dict = {}
    for i, name in enumerate(df[feature_name].value_counts().keys()): 
        if int_to_str:   
            temp_dict[i] = name
        else:
            temp_dict[name] = i

    return temp_dict


def get_dataset():
    """
    Assumes target class is binary 0 1, and that 1 is the semi-factual class
    """

    df = pd.read_csv('data/adult.csv')
    del df['fnlwgt']
    df = df.replace({'<=50K': 0, '>50K': 1})
    # df = df[df['capital-gain'] < 1000]
    # df = df[df['capital-loss'] < 1000]

    for f in continuous_feature_names:
        df[f] = df[f].astype('float')

    df.education = df.education.replace({
                             'Preschool':1,
                             '1st-4th':2, 
                             '5th-6th':3,
                             '7th-8th':4,
                             '9th':5,
                             '10th':6, 
                             '11th':7,
                             '12th':8,
                             'HS-grad':9,
                             'Some-college':10, 
                             'Assoc-voc':11,
                             'Assoc-acdm':12,
                             'Bachelors':13,
                             'Masters':14, 
                             'Doctorate':15,
                             'Prof-school':16
                            })

    # education num -- already ints


    df.gender = df.gender.replace({'Female': 1,
                             'Male':2,
                            })

    df['marital-status'] = df['marital-status'].replace({'Never-married': 1,
                             'Separated':2,
                             'Widowed':3, 
                             'Married-spouse-absent':4,
                             'Divorced':5,
                             'Married-AF-spouse':6, 
                             'Married-civ-spouse':7,
                            })

    t = get_random_dict('native-country', int_to_str=False)
    df['native-country'] = df['native-country'].replace(t)   

    df.occupation = df.occupation.replace({
                             'Priv-house-serv':1,
                             'Other-service':2, 
                             'Handlers-cleaners':3,
                             '?':4,
                             'Farming-fishing':5,
                             'Machine-op-inspct':6, 
                             'Adm-clerical':7,
                             'Transport-moving':8,
                             'Craft-repair':9, 
                             'Sales':10,
                             'Tech-support':11,
                             'Protective-serv':12,
                             'Armed-Forces':13, 
                             'Prof-specialty':14,
                             'Exec-managerial':15
                            })

    df['race'] = df['race'].replace({'Amer-Indian-Eskimo': 1,
                             'Black':2,
                             'Other':3, 
                             'White':4,
                             'Asian-Pac-Islander':5,
                            })

    t = get_random_dict('relationship', int_to_str=False)
    df['relationship'] = df['relationship'].replace(t)   

    df.workclass = df.workclass.replace({
                             'Never-worked':1,
                             '?':2, 
                             'Without-pay':3,
                             'Private':4,
                             'State-gov':5,
                             'Self-emp-not-inc':6, 
                             'Local-gov':7,
                             'Federal-gov':8,
                             'Self-emp-inc':9, 
                            })

    return df


def make_human_readable(df):

    df.education = df.education.replace({
                             1:'Preschool',
                             2:'1st-4th', 
                             3:'5th-6th',
                             4:'7th-8th',
                             5:'9th',
                             6:'10th', 
                             7:'11th',
                             8:'12th',
                             9:'HS-grad',
                             10:'Some-college', 
                             11:'Assoc-voc',
                             12:'Assoc-acdm',
                             13:'Bachelors',
                             14:'Masters', 
                             15:'Doctorate',
                             16:'Prof-school'
                            })

    # education num -- already ints


    df.gender = df.gender.replace({1:'Female',
                             2:'Male',
                            })


    df['marital-status'] = df['marital-status'].replace({1:'Never-married',
                             2:'Separated',
                             3:'Widowed', 
                             4:'Married-spouse-absent',
                             5:'Divorced',
                             6:'Married-AF-spouse', 
                             7:'Married-civ-spouse',
                            })


    t = get_random_dict('native-country', int_to_str=False)
    df['native-country'] = df['native-country'].replace(t)   


    df.occupation = df.occupation.replace({
                             1:'Priv-house-serv',
                             2:'Other-service', 
                             3:'Handlers-cleaners',
                             4:'?',
                             5:'Farming-fishing',
                             6:'Machine-op-inspct', 
                             7:'Adm-clerical',
                             8:'Transport-moving',
                             9:'Craft-repair', 
                             10:'Sales',
                             11:'Tech-support',
                             12:'Protective-serv',
                             13:'Armed-Forces', 
                             14:'Prof-specialty',
                             15:'Exec-managerial'
                            })


    df['race'] = df['race'].replace({1:'Amer-Indian-Eskimo',
                             2:'Black',
                             3:'Other', 
                             4:'White',
                             5:'Asian-Pac-Islander',
                            })

    t = get_random_dict('relationship', int_to_str=True)
    df['relationship'] = df['relationship'].replace(t)   

    df.workclass = df.workclass.replace({
                             1:'Never-worked',
                             2:'?', 
                             3:'Without-pay',
                             4:'Private',
                             5:'State-gov',
                             6:'Self-emp-not-inc', 
                             7:'Local-gov',
                             8:'Federal-gov',
                             9:'Self-emp-inc', 
                            })
    return df


def actionability_constraints():
    
    #### increasing means "increasing" probability of loan
    #### based on common sense actionable directions


    meta_action_data =  {
     'age': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'workclass': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'education': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'educational-num': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'marital-status': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'occupation': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'relationship': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'race': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'gender': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'capital-gain': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'capital-loss': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': True,
      'can_decrease': False},

     'hours-per-week': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'native-country': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

    }
    
    return meta_action_data

