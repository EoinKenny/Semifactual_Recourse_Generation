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


TARGET_NAME = 'cancer'
continuous_feature_names = []  
categorical_feature_names = ['agegrp', 'density', 'race', 'Hispanic', 'bmi', 'agefirst', 'nrelbc',
       'brstproc', 'lastmamm', 'surgmeno', 'hrt']



def get_dataset():
    """
    Assumes target class is binary 0 1, and that 1 is the semi-factual class
    """

    df = pd.read_csv('data/cancer_risk.csv')
    df.agegrp = df.agegrp.replace({
                             '80-84':1,
                             '75-79':2, 
                             '70-74':3,
                             '65-69':4,
                             '60-64':5,
                             '55-59':6, 
                             '50-54':7,
                             '45-49':8
                            })


    df.density = df.density.replace({'Extremely dense': 1,
                             'Heterogeneously dense':2,
                             'Scattered fibroglandular densities':3, 
                             'Almost entirely fat':4,
                            })


    df.race = df.race.replace({'white': 1,
                             'black':2,
                             'Asian/Pacific Islander':3, 
                             'Native American':4,
                            })


    df.nrelbc = df.nrelbc.replace({'2 relatives with cancer': 1,
                             '1 relatives with cancer':2,
                             '0 relatives with cancer':3, 
                            })

    df.surgmeno = df.surgmeno.replace({'Surgical menopause': 1,
                             'natural Surgical menopause':2
                                  })

    df.lastmamm = df.lastmamm.replace({'false positive Result of last mammogram': 1,
                             'negative Result of last mammogram':2
                                  })

    df.bmi = df.bmi.replace({'35 bmi': 1,
                             '30-34.99 bmi':2,
                             '10-24.99 bmi':3, 
                             '25-29.99 bmi':4,
                            })

    df['agefirst'] = df['agefirst'].replace({
                                     'Nulliparous first birth': 1,
                                     'Age 30 or greater first birth':2,
                                     'Age < 30 first birth':3, 
                                  })

    df.brstproc = df.brstproc.replace({'yes Previous breast procedure': 1,
                                     'no Previous breast procedure':2
                                  })

    df.Hispanic = df.Hispanic.replace({'not hispanic': 1,
                                     'hispanic':2
                                  })

    df.hrt = df.hrt.replace({'yes hrt': 1,
                             'no hrt':2
                                  })

    return df


def make_human_readable(df):

    df.agegrp = df.agegrp.replace({
                             1:'80-84',
                             2:'75-79', 
                             3:'70-74',
                             4:'65-69',
                             5:'60-64',
                             6:'55-59', 
                             7:'50-54',
                             8:'45-49'
                            })


    df.density = df.density.replace({
                             1:'Extremely dense',
                             2:'Heterogeneously dense',
                             3:'Scattered fibroglandular densities', 
                             4:'Almost entirely fat',
                            })


    df.race = df.race.replace({
                             1:'white',
                             2:'black',
                             3:'Asian/Pacific Islander', 
                             4:'Native American',
                            })


    df.nrelbc = df.nrelbc.replace({
                             1:'2 relatives with cancer',
                             2:'1 relatives with cancer',
                             3:'0 relatives with cancer', 
                            })

    df.surgmeno = df.surgmeno.replace({1:'Surgical menopause',
                             2:'natural Surgical menopause'
                                  })

    df.lastmamm = df.lastmamm.replace({1:'false positive Result of last mammogram',
                             2:'negative Result of last mammogram'
                                  })

    df.bmi = df.bmi.replace({1:'35 bmi',
                             2:'30-34.99 bmi',
                             3:'10-24.99 bmi', 
                             4:'25-29.99 bmi',
                            })

    df['agefirst'] = df['agefirst'].replace({
                                     1:'Nulliparous first birth',
                                     2:'Age 30 or greater first birth',
                                     3:'Age < 30 first birth', 
                                  })

    df.brstproc = df.brstproc.replace({1:'yes Previous breast procedure',
                                     2:'no Previous breast procedure'
                                  })

    df.Hispanic = df.Hispanic.replace({1:'not hispanic',
                                     2:'hispanic'
                                  })

    df.hrt = df.hrt.replace({1:'yes hrt',
                             2:'no hrt'
                                  })


    return df


def actionability_constraints():
    
    #### increasing means "increasing" probability of loan
    #### based on common sense actionable directions


    meta_action_data =  {
     'bmi': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'brstproc': {'actionable': False,
      'min': 0,
      'max': 1, 
      'can_increase': False,
      'can_decrease': False},

     'hrt': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'agegrp': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'density': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'race': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'Hispanic': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'agefirst': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'nrelbc': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'lastmamm': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'surgmeno': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},


    }
    
    return meta_action_data


