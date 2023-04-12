# Data Mining Techniques
# Koen Smallegange 
# april 2023
# ---------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from functions import bin_programmes
import sys

df = pd.read_csv("data.csv", delimiter = ";", error_bad_lines='false')
df.rename(
    columns=({ 'Tijdstempel': 'time', 'What is your gender?': 'gender', 'What programme are you in?': 'programme',
               'Have you taken a course on machine learning?': 'ML course',
               'Have you taken a course on information retrieval?': 'IR course', 'When is your birthday (date)?': 'birthday',
               'How many students do you estimate there are in the room?':'student guess',
               'Did you stand up to come to your previous answer?': 'stand up', 'What is your stress level (0-100)?':'stress level',
               'Have you taken a course on statistics?': 'statistics course', 'Have you taken a course on databases?' : 'databases course',
               'I have used ChatGPT to help me with some of my study assignments ': 'used chatGPT',
               'How many hours per week do you do sports (in whole hours)?': 'sports hours',
               'Give a random number': 'random number', 'Time you went to bed Yesterday':'bed time',
               'What makes a good day for you (1)?': 'good day 1', 'What makes a good day for you (2)?': 'good day 2'}),
    inplace=True,
)

dfc = bin_programmes(df)
prol = []

for index, row in dfc.iterrows():
    cpgt = dfc.at[index, 'used chatGPT']
    
    if 'yes' in cpgt:
        program = dfc.at[index, 'programme']
        print(cpgt, program, index)
        prol.append(program)


print(prol)









