
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
import re
from scipy.stats import zscore
# from fancyimpute import IterativeImputer
import sys 

# First classification algorithm
def decision_tree_algorithm():
    df = pd.read_csv('data.csv', sep=';')
    pick = information_gain_DT(df)
    # Need to pick the right attribute.
    
def attributes(col,df,value_list):
    number_of_values = len(value_list)
    student_dict = {}
    
    for value in value_list:
        stress_levels = stress_level_evaluator(value,col,df,number_of_values)
        student_dict[value] = stress_levels
    return student_dict

def entropy(dictionary):
    for values in dictionary:
        entropyscore = entropyscore - values * np.log2(values)
    return entropyscore
    
# defines the information gained by each different step in the decision tree
# this is done using the entropy. Could also be done using Gini impurity.
def information_gain_DT(df):

    for col in df.columns:
    
        for value in df[col]:
                
            if col == df.columns[1]:            
                value_list = ['AI','Econometrics','Computational Science','Quantitative Risk Management','Business Analytics','Computer Science','Finance and Technology','Bioinformatics','Exhange Programme','Neuroscience','PhD','Life Sciences','Other']
                studies_dict = attributes(col,df,value_list)
                entropyscore = entropy(studies_dict)

            if col == df.columns[2]:
                value_list = [0,1]
                machine_dict = attributes(col,df,value_list)

            if col == df.columns[3]:
                value_list = [0,1]
                information_dict = attributes(col,df,value_list)

            if col == df.columns[4]:
                value_list = [0,1]
                statistics_dict = attributes(col,df,value_list)

            if col == df.columns[5]:
                value_list = [0,1]
                database_dict = attributes(col,df,value_list)
            # Gender
            if col == df.columns[6]:
                value_list = [0,1]
                gender_dict = attributes(col,df,value_list)
            # Use chatgpt or not
            if col == df.columns[7]:
                value_list = [0,1]
                chatgpt_dict = attributes(col,df,value_list)

            # Birthday but seems irrelevant.
            if col == df.columns[8]:
                pass
            if col == df.columns[9]:
                value_list = ['0-50', '51-100', '101-200', '201-300', '301-400', '401-500', '501-1000', '1001+']
                peoplepred_dict = attributes(col,df,value_list)
                
            # standing up
            if col == df.columns[10]:
                value_list = [0,1]
                standingup_dict = attributes(col,df,value_list)
            
            if col == df.columns[11]:
                print(col)
            # Sports hours per week
            if col == df.columns[12]:
                value_list = ['0','0-2','2-4','4-6','6-10','10-15','15+']
                sports_dict = attributes(col,df,value_list)
            # Does not have any values
            if col == df.columns[13]:
                pass
            if col == df.columns[14]:
                pass
            if col == df.columns[15]:
                pass
                # Moet dan nog iets met deze dict values gebeuren om een evaluation te maken.
                                    
    attribute = 0
    return attribute

# predict stress level 
# should return a list of number of occurences of each of the different stress levels.
def stress_level_evaluator(value,col,df,number_of_values):
    # need to first create 10 different totals of the stress 
    
    stress_levels = np.zeros((1,number_of_values))
    row_numbers = df[df[col] == value].index.tolist()
    stress = 'What is your stress level (0-100)?'
    for row_number in row_numbers:
        value = df.loc[row_number, stress]

        # Placeholder code
        try:
            value = int(value)
        except ValueError:
            value = 1
    
        if value >= 0 and value < 10:
            stress_levels[0,0] = stress_levels[0,0] + 1
        if value >= 10 and value < 20:
            stress_levels[0,1] = stress_levels[0,1] + 1
        if value >= 20 and value < 30:
           stress_levels[0,2] = stress_levels[0,2] + 1
        if value >= 30 and value < 40:
           stress_levels[0,3] = stress_levels[0,3] + 1
        if value >= 40 and value < 50:
            stress_levels[0,4] = stress_levels[0,4] + 1
        if value >= 50 and value < 60:
           stress_levels[0,5] = stress_levels[0,5] + 1
        if value >= 60 and value < 70:
          stress_levels[0,6] = stress_levels[0,6] + 1
        if value >= 70 and value < 80:
            stress_levels[0,7] = stress_levels[0,7] + 1
        if value >= 80 and value < 90:
           stress_levels[0,8] = stress_levels[0,8] + 1
        if value >= 90:
            stress_levels[0,9] = stress_levels[0,9] + 1
        
    return stress_levels



# The gain ratio is a method to reduce the 
def gain_ratio_DT():
    pass

# Test decision tree algorithm

stress_level = decision_tree_algorithm()


# # Iterate through the column
                # for i in df[col]:
                    
                #     if i in string_counts:
                #         string_counts[i] += 1
                #     else:
                #         string_counts[i] = 1
                # # Sort the dictionary based on the number of occurrences (values)
                # sorted_counts = sorted(string_counts.items(), key=lambda x: x[1], reverse=True)

                # for key, count in string_counts.items():
                #     print(f"{key}: {count}")