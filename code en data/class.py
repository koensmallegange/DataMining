
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
    df = pd.read_csv('clean.csv', sep=';')
    pick = information_gain_DT(df)
    # Need to pick the right attribute.
    
def attributes(col,df,value_list):
    number_of_values = len(value_list)
    student_dict = {}
    
    for value in value_list:
        stress_levels = stress_level_evaluator(value,col,df,number_of_values)
        student_dict[value] = stress_levels
    return student_dict

# predict stress level 
# should return a list of number of occurences of each of the different stress levels.
def stress_level_evaluator(value,col,df,number_of_values):
    # need to first create 10 different totals of the stress 
    stress_levels = np.zeros((1,10))
    row_numbers = df[df[col] == str(value)].index.tolist()
    print('row')
    print(row_numbers)
    
    stress = 'What is your stress level (0-100)?'
    total = 0
    for row_number in row_numbers:
        print(value)
        print('jajajtoure')
        value = df.loc[row_number, stress]
        # Placeholder code
        try:
            value = int(value)
        except ValueError:
            value = 1

        total = total + 1
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

    stress_level = stress_levels[0]

    # Laplace replacement and probability calculation.
    number_of_zeros = np.count_nonzero(stress_level == 0)
    i = 0
    # HIER NIET ZEKER OF DE LEGE LIJSTEN OOK ALLEMAAL VALUES MOETEN KRIJGEN
    # EN OOK NIET OF DE NIET-NUL VALUES OOK EEN EXTRA WAARDE MOETEN KRIJGEN MAAR NEEM AAN VAN WEL
    # VRAAG DIT AAN TA.
    for item in stress_level:
        chance_per_stress_level = (item + number_of_zeros)/(total + number_of_zeros * 10)
        stress_level[i] = chance_per_stress_level
        i = i + 1
        
    return stress_level

def Laplace_estimator():
    pass


def weighted_average(list_for_weight,entropydict):

    new_dict = {}
    i = 0
    totalsum = sum(list_for_weight)
    weighted_entropy = 0

    for values in entropydict.values():
        weighted_entropy = weighted_entropy + (list_for_weight[i]/totalsum) * values
        i = i + 1
    return weighted_entropy

def entropy(dictionary):
    
    entropydict = {}
    list_for_weight = []
    i = 0

    for values in dictionary.values():
    
        entropyscore = 0
        
        for single_value in values:
            entropyscore = entropyscore - (single_value * np.log2(single_value))

        list_for_weight.append(sum(values))
        entropydict[list(dictionary.keys())[i]] = entropyscore
        i = i + 1
    weighted_entropy = weighted_average(list_for_weight,entropydict)

    return weighted_entropy
    
# defines the information gained by each different step in the decision tree
# this is done using the entropy. Could also be done using Gini impurity.
def information_gain_DT(df):


    lowest_entropy = 10
    # dit hele proces moet elke keer na het weghalen van een waarde binnen df.columns gekopieerd worden.
    print(df.columns)
    for col in df.columns:
                   
        if col == df.columns[1]:            
            value_list = ['AI','Econometrics','Computational Science','Quantitative Risk Management','Business Analytics','Computer Science','Finance and Technology','Bioinformatics','Exhange Programme','Neuroscience','PhD','Life Sciences','Other']
            studies_dict = attributes(col,df,value_list)
            # chances_list = get_chance(studies_dict)
            weighted_entropy = entropy(studies_dict)
            if weighted_entropy <= lowest_entropy:
                lowest_entropy = weighted_entropy
                attribute = col

        if col == df.columns[2]:
            value_list = [0,1]
            machine_dict = attributes(col,df,value_list)
            weighted_entropy = entropy(machine_dict)
            if weighted_entropy <= lowest_entropy:
                lowest_entropy = weighted_entropy
                attribute = col

        if col == df.columns[3]:
            value_list = [0,1]
            information_dict = attributes(col,df,value_list)
            weighted_entropy = entropy(information_dict)
            if weighted_entropy <= lowest_entropy:
                lowest_entropy = weighted_entropy
                attribute = col


        if col == df.columns[4]:
            value_list = [0,1]
            statistics_dict = attributes(col,df,value_list)
            weighted_entropy = entropy(statistics_dict)
            if weighted_entropy <= lowest_entropy:
                lowest_entropy = weighted_entropy
                attribute = col


        if col == df.columns[5]:
            value_list = [0,1]
            database_dict = attributes(col,df,value_list)
            weighted_entropy = entropy(database_dict)
            if weighted_entropy <= lowest_entropy:
                lowest_entropy = weighted_entropy
                attribute = col

        # Gender
        if col == df.columns[6]:
            value_list = [0,1]
            gender_dict = attributes(col,df,value_list)
            weighted_entropy = entropy(gender_dict)
            if weighted_entropy <= lowest_entropy:
                lowest_entropy = weighted_entropy
                attribute = col

        # Use chatgpt or not
        if col == df.columns[7]:
            value_list = [0,1]
            chatgpt_dict = attributes(col,df,value_list)
            weighted_entropy = entropy(chatgpt_dict)
            if weighted_entropy <= lowest_entropy:
                lowest_entropy = weighted_entropy
                attribute = col


        # Birthday but seems irrelevant.
        if col == df.columns[8]:
            pass
        if col == df.columns[9]:
            value_list = ['0-50', '51-100', '101-200', '201-300', '301-400', '401-500', '501-1000', '1001+']
            peoplepred_dict = attributes(col,df,value_list)
            weighted_entropy = entropy(peoplepred_dict)
            if weighted_entropy <= lowest_entropy:
                lowest_entropy = weighted_entropy
                attribute = col

            
        # standing up
        if col == df.columns[10]:
            value_list = [0,1]
            standingup_dict = attributes(col,df,value_list)
            weighted_entropy = entropy(standingup_dict)
            if weighted_entropy <= lowest_entropy:
                lowest_entropy = weighted_entropy
                attribute = col

        # stress level is the one we are trying to predict so can be left out.
        if col == df.columns[11]:
            pass
            
        # Sports hours per week
        if col == df.columns[12]:
            value_list = ['0','0-2','2-4','4-6','6-10','10-15','15+']
            sports_dict = attributes(col,df,value_list)
            weighted_entropy = entropy(sports_dict)
            if weighted_entropy <= lowest_entropy:
                lowest_entropy = weighted_entropy
                attribute = col
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


def tree_structure(df):

    for col in df.columns:
        
    
        for value in df[col]:
                
            if col == df.columns[1]:            
                value_list = ['AI','Econometrics','Computational Science','Quantitative Risk Management','Business Analytics','Computer Science','Finance and Technology','Bioinformatics','Exhange Programme','Neuroscience','PhD','Life Sciences','Other']
                studies_dict = attributes(col,df,value_list)
                # chances_list = get_chance(studies_dict)
                weighted_entropy = entropy(studies_dict)

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
            # stress level is the one we are trying to predict so can be left out.
            if col == df.columns[11]:
                pass
                
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



# unnecessary now 
# def get_chance(dictionary):
#     # for this you need both the total number of occurences of any stress level and the number
#     # need to actually create another dict
#     for values in dictionary:
#         for single_values in values:
#             # This needs to be equated before, because the number of total number needs to be adjusted
#             for single_value in single_values:
#                 if single_value == 0:
#                     single_value = Laplace_estimator()


# def information_gain_DT(df):


#     lowest_entropy = 10
#     # dit hele proces moet elke keer na het weghalen van een waarde binnen df.columns gekopieerd worden.
    
#     for col in df.columns:
    
#         for value in df[col]:
#             print('value')
#             print(value)
                
#             if col == df.columns[1]:            
#                 value_list = ['AI','Econometrics','Computational Science','Quantitative Risk Management','Business Analytics','Computer Science','Finance and Technology','Bioinformatics','Exhange Programme','Neuroscience','PhD','Life Sciences','Other']
#                 studies_dict = attributes(col,df,value_list)
#                 # chances_list = get_chance(studies_dict)
#                 weighted_entropy = entropy(studies_dict)

#             if col == df.columns[2]:
#                 value_list = [0,1]
#                 machine_dict = attributes(col,df,value_list)

#             if col == df.columns[3]:
#                 value_list = [0,1]
#                 information_dict = attributes(col,df,value_list)

#             if col == df.columns[4]:
#                 value_list = [0,1]
#                 statistics_dict = attributes(col,df,value_list)

#             if col == df.columns[5]:
#                 value_list = [0,1]
#                 database_dict = attributes(col,df,value_list)
#             # Gender
#             if col == df.columns[6]:
#                 value_list = [0,1]
#                 gender_dict = attributes(col,df,value_list)
#             # Use chatgpt or not
#             if col == df.columns[7]:
#                 value_list = [0,1]
#                 chatgpt_dict = attributes(col,df,value_list)

#             # Birthday but seems irrelevant.
#             if col == df.columns[8]:
#                 pass
#             if col == df.columns[9]:
#                 value_list = ['0-50', '51-100', '101-200', '201-300', '301-400', '401-500', '501-1000', '1001+']
#                 peoplepred_dict = attributes(col,df,value_list)
                
#             # standing up
#             if col == df.columns[10]:
#                 value_list = [0,1]
#                 standingup_dict = attributes(col,df,value_list)
#             # stress level is the one we are trying to predict so can be left out.
#             if col == df.columns[11]:
#                 pass
                
#             # Sports hours per week
#             if col == df.columns[12]:
#                 value_list = ['0','0-2','2-4','4-6','6-10','10-15','15+']
#                 sports_dict = attributes(col,df,value_list)
#             # Does not have any values
#             if col == df.columns[13]:
#                 pass
#             if col == df.columns[14]:
#                 pass
#             if col == df.columns[15]:
#                 pass
#                 # Moet dan nog iets met deze dict values gebeuren om een evaluation te maken.
                                    
#     attribute = 0
#     return attribute