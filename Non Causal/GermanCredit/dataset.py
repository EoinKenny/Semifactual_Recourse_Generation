import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from copy import deepcopy
from collections import Counter


TARGET_NAME = 'credit_risk'
continuous_feature_names = ['duration', 'amount', 'age']
categorical_feature_names = ['status', 'credit_history', 'purpose', 'savings',
       'employment_duration', 'installment_rate', 'personal_status_sex',
       'other_debtors', 'present_residence', 'property',
       'other_installment_plans', 'housing', 'number_credits', 'job',
       'people_liable', 'telephone', 'foreign_worker']

# continuous_feature_names = ['duration', 'amount']
# categorical_feature_names = ['savings', 'number_credits']



"""
$`laufkont = status`
                                               
 1 : no checking account                       
 2 : ... < 0 DM                                
 3 : 0<= ... < 200 DM                          
 4 : ... >= 200 DM / salary for at least 1 year

$`laufzeit = duration`
     
$`moral = credit_history`
                                                
 0 : delay in paying off in the past            
 1 : critical account/other credits elsewhere   
 2 : no credits taken/all credits paid back duly
 3 : existing credits paid back duly till now   
 4 : all credits at this bank paid back duly    

$`verw = purpose`
                        
 0 : others             
 1 : car (new)          
 2 : car (used)         
 3 : furniture/equipment
 4 : radio/television   
 5 : domestic appliances
 6 : repairs            
 7 : education          
 8 : vacation           
 9 : retraining         
 10 : business          

$`hoehe = amount`
     

$`sparkont = savings`
                               
 1 : unknown/no savings account
 2 : ... <  100 DM             
 3 : 100 <= ... <  500 DM      
 4 : 500 <= ... < 1000 DM      
 5 : ... >= 1000 DM            

$`beszeit = employment_duration`
                     
 1 : unemployed      
 2 : < 1 yr          
 3 : 1 <= ... < 4 yrs
 4 : 4 <= ... < 7 yrs
 5 : >= 7 yrs        

$`rate = installment_rate`
                   
 1 : >= 35         
 2 : 25 <= ... < 35
 3 : 20 <= ... < 25
 4 : < 20          

$`famges = personal_status_sex`
                                         
 1 : male : divorced/separated           
 2 : female : non-single or male : single
 3 : male : married/widowed              
 4 : female : single                     

$`buerge = other_debtors`
                 
 1 : none        
 2 : co-applicant
 3 : guarantor   

$`wohnzeit = present_residence`
                     
 1 : < 1 yr          
 2 : 1 <= ... < 4 yrs
 3 : 4 <= ... < 7 yrs
 4 : >= 7 yrs        

$`verm = property`
                                              
 1 : unknown / no property                    
 2 : car or other                             
 3 : building soc. savings agr./life insurance
 4 : real estate                              

$`alter = age`
     

$`weitkred = other_installment_plans`
           
 1 : bank  
 2 : stores
 3 : none  

$`wohn = housing`
             
 1 : for free
 2 : rent    
 3 : own     

$`bishkred = number_credits`
         
 1 : 1   
 2 : 2-3 
 3 : 4-5 
 4 : >= 6

$`beruf = job`
                                               
 1 : unemployed/unskilled - non-resident       
 2 : unskilled - resident                      
 3 : skilled employee/official                 
 4 : manager/self-empl./highly qualif. employee

$`pers = people_liable`
              
 1 : 3 or more
 2 : 0 to 2   

$`telef = telephone`
                              
 1 : no                       
 2 : yes (under customer name)

$`gastarb = foreign_worker`
        
 1 : yes
 2 : no 

$`kredit = credit_risk`
         
 0 : bad 
 1 : good

 """


# def make_human_readable(df):

#     """
#     Convert integers into text, but not the label
#     """
    

#     df.savings = df.savings.replace({1:'unknown/no savings account',
#                                    2:'... <  100 DM',
#                                    3:'100 <= ... <  500 DM',
#                                    4:'500 <= ... < 1000 DM',
#                                    5: '... >= 1000 DM'})



#     df.number_credits = df.number_credits.replace({1:'1',
#                                    2:'2-3',
#                                    3:'4-5',
#                                    4:'>= 6'})
    
#     return df



def make_human_readable(df):

    """
    Convert integers into text, but not the label
    """
    
    df.status = df.status.replace({1: 'no checking account',
                                     2: '< 0 DM',
                                     3: '0 <= ... <= 200 DM', 
                                     4: '>= 200 DM / salary for at least 1 year'
                                  })

    df.credit_history = df.credit_history.replace({
                               0:'Delay in paying off in the past',
                               1:'Critical account/other credits elsewhere',
                               2:'No credits taken/all credits paid back duly',
                               3:'Existing credits paid back duly till now', 
                               4:'All credits at this bank paid back duly'})

    df.purpose = df.purpose.replace({0:'Others',
                                     1:'Car (new)',
                                     2:'Car (Used)',
                                     3:'Furniture/equipment',
                                     4:'Radio/television',
                                     5:'Domestic Applicances',
                                     6:'Repairs',
                                     7:'Education',
                                     8:'Vacation',
                                     9:'Retraining',
                                     10:'Business'
                                    })

    df.savings = df.savings.replace({1:'unknown/no savings account',
                                   2:'... <  100 DM',
                                   3:'100 <= ... <  500 DM',
                                   4:'500 <= ... < 1000 DM',
                                   5: '... >= 1000 DM'})

    df.employment_duration = df.employment_duration.replace({1:'unemployed',
                                   2:'< 1 yr',
                                   3:'1 <= ... < 4 yrs',
                                   4:'4 <= ... < 7 yrs',
                                   5: '>= 7 yrs '})

    df.installment_rate = df.installment_rate.replace({1:'>= 35',
                                   2:'25 <= ... < 35',
                                   3:'20 <= ... < 25',
                                   4:'< 20'})

    df.personal_status_sex = df.personal_status_sex.replace({1:'divorced/separated',
                                   2:'non-single or male : single',
                                   3:'married/widowed',
                                   4:'single'})

    df.other_debtors = df.other_debtors.replace({1:'none',
                                   2:'co-applicant',
                                   3:'guarantor',
                                })

    df.present_residence = df.present_residence.replace({1:'< 1 yr',
                                   2:'1 <= ... < 4 yrs',
                                   3:'4 <= ... < 7 yrs',
                                   4:'>= 7 yrs'})

    df.property = df.property.replace({1:'unknown / no property',
                                   2:'car or other',
                                   3:'building soc. savings agr./life insurance',
                                   4:'real estate'})

    df.other_installment_plans = df.other_installment_plans.replace({1:'bank',
                                   2:'stores',
                                   3:'none',
                                })

    df.housing = df.housing.replace({1:'for free',
                                   2:'rent',
                                   3:'own',
                                })

    df.number_credits = df.number_credits.replace({1:'1',
                                   2:'2-3',
                                   3:'4-5',
                                   4:'>= 6'})

    df.job = df.job.replace({1:'unemployed/unskilled - non-resident',
                                   2:'unskilled - resident',
                                   3:'skilled employee/official',
                                   4:'manager/self-empl./highly qualif. employee'})

    df.people_liable = df.people_liable.replace({
                                   1:'3 or more',
                                   2:'0 to 2'})

    df.telephone = df.telephone.replace({1:'no',
                                   2:'yes (under customer name)'})

    df.foreign_worker = df.foreign_worker.replace({1:'yes',
                                   2:'no'})

#     df.credit_risk = df.credit_risk.replace({0:'bad', 1:'good'})
    
    return df



def get_dataset(seed):

    """
    Read in dataset in the form of integers
    """

    df = pd.read_csv('data/SouthGermanCredit/SouthGermanCredit.asc', sep=" ")
    
#     df = df[['savings', 'duration', 'amount', 'number_credits', 'credit_risk']]

    return df



# # Just for user study
# def actionability_constraints():
    
#     #### increasing means "increasing" probability of loan
#     #### based on common sense actionable directions
    
#     meta_action_data =  {
#      'duration': {'actionable': True,
#       'min': 0,
#       'max': 1,
#       'can_increase': False,
#       'can_decrease': True},

#      'savings': {'actionable': False,
#       'min': 0,
#       'max': 1,
#       'can_increase': False,
#       'can_decrease': False},
        
#      'number_credits': {'actionable': True,  # flip
#       'min': 0,
#       'max': 1,
#       'can_increase': False,
#       'can_decrease': True},

#      'amount': {'actionable': False,
#       'min': 0,
#       'max': 1,
#       'can_increase': False,
#       'can_decrease': False},

#     }
    
#     return meta_action_data




def actionability_constraints():
    
    #### increasing means "increasing" probability of loan
    #### based on common sense actionable directions
    
    meta_action_data =  {
     'duration': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': True,
      'can_decrease': False},

     'amount': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': True,
      'can_decrease': False},

     'age': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'status': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'credit_history': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'purpose': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'savings': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'employment_duration': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'installment_rate': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': True,
      'can_decrease': False},

     'personal_status_sex': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'other_debtors': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': True,
      'can_decrease': False},

     'present_residence': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'property': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'other_installment_plans': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'housing': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'number_credits': {'actionable': True,  # flip
      'min': 0,
      'max': 1,
      'can_increase': True,
      'can_decrease': False},

     'job': {'actionable': True,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'people_liable': {'actionable': True, 
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': True},

     'telephone': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},

     'foreign_worker': {'actionable': False,
      'min': 0,
      'max': 1,
      'can_increase': False,
      'can_decrease': False},
    }
    
    return meta_action_data


