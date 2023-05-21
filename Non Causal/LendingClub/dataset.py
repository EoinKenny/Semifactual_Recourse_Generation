import pandas as pd
import numpy as np
import scipy
import pdb

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

from copy import deepcopy

from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt


TARGET_NAME = 'loan_status'
continuous_feature_names = ['loan_amnt', 'pub_rec_bankruptcies', 'annual_inc', 'dti']
categorical_feature_names = ['emp_length', 'term', 'grade', 'home_ownership', 'purpose']


def get_dataset(seed):
    """
    Assumes target class is binary 0 1, and that 1 is the semi-factual class
    """

    df = pd.read_csv('data/reduced_df.csv')

    for cat in categorical_feature_names:
        df[cat] = df[cat].astype('int')
    for cat in continuous_feature_names:
        df[cat] = df[cat].astype('float')

    return df


def make_human_readable(df):


    df.home_ownership = df.home_ownership.replace({
                             1:'RENT',
                             2:'ANY', 
                             3:'OTHER',
                             4:'MORTGAGE',
                             5:'OWN', 
                             6:'NONE',
                            })

    df.grade = df.grade.replace({
                             1:'G',
                             2:'F', 
                             3:'E',
                             4:'D',
                             5:'C', 
                             6:'B',
                             7:'A',
                            })

    df.emp_length = df.emp_length.replace({
                             1:'< 1 year',
                             2:'1 year', 
                             3:'2 years',
                             4:'3 years',
                             5:'4 years', 
                             6:'5 years',
                             7:'6 years',
                             8:'7 years', 
                             9:'8 years',
                             10:'9 years',
                             11:'10+ years', 

                            })


    df.purpose = df.purpose.replace({
                             1:'small_business',
                             2:'house', 
                             3:'renewable_energy',
                             4:'moving', 
                             5:'debt_consolidation',
                             6:'other',
                             7:'medical', 
                             8:'educational',
                             9:'major_purchase',
                             10:'vacation', 
                             11:'home_improvement',
                             12:'credit_card',
                             13:'car', 
                             14:'wedding',

                            })

    df.term = df.term.replace({
                             1:' 60 months',
                             2:' 36 months', 
                            })

    return df


def actionability_constraints():
    
    #### increasing means "increasing" probability of loan
    #### based on common sense actionable directions


    meta_action_data =  {
     'home_ownership': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'annual_inc': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'emp_length': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'grade': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'dti': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': True,
      'can_decrease': False},

     'pub_rec_bankruptcies': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': True,
      'can_decrease': False},

     'purpose': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'loan_amnt': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': True,
      'can_decrease': False},

     'term': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},
    }
    
    return meta_action_data



