# Data Mining Techniques
# Koen Smallegange 
# april 2023
# 
# This script controls all code
# 
# ---------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from clean import clean_frame

# define path to data
path = 'data.csv'

# get a clean dataframe and save as csv
df = clean_frame(path)

# use new clean csv for new dataframe
dfc = pd.read_csv('clean.csv')

# ---------- TRANSFORMATIONS------------
# 

# average stress level per programme
dfc['StressLevelLower'] = dfc['What is your stress level (0-100)?'].str.split('-', expand=True)[0].astype(float)
dfc['StressLevelUpper'] = dfc['What is your stress level (0-100)?'].str.split('-', expand=True)[1].astype(float)
dfc['AverageStressLevel'] = (dfc['StressLevelLower'] + dfc['StressLevelUpper']) / 2
avg_stress_by_program = dfc.groupby('What programme are you in?')['AverageStressLevel'].mean()

avg_stress_by_program(kind='bar', alpha = 0.75)
plt.ylabel('Average Stress Level')
plt.title('Average Stress Level by Program')
plt.subplots_adjust(bottom = 0.35)
plt.show()


# estimation of students in the room per program
df['Estimate_Lower'] = df['How many students do you estimate there are in the room?'].str.split('-').str[0].astype(float)
df['Estimate_Upper'] = df['How many students do you estimate there are in the room?'].str.split('-').str[1].str.split().str[0].astype(float)
df['Estimate_Middle'] = (df['Estimate_Lower'] + df['Estimate_Upper']) / 2

# Calculate average estimate and standard deviation per study program
average_estimate = df.groupby('What programme are you in?').mean()
std_estimate = df.groupby('What programme are you in?').std()

# Plot average estimate and standard deviation per study program
plt.figure(figsize=(10, 6))
average_estimate['Estimate_Middle'].plot(kind='bar', color='lightblue', label='Middle of Estimate Range')
plt.errorbar(average_estimate.index, average_estimate['Estimate_Middle'], yerr=std_estimate['Estimate_Middle'], fmt='o', color='black', label='Std Deviation')
plt.title('Average Estimate of Number of Students in the Room with Standard Deviation per Study Program')
plt.ylabel('Average Estimate')
plt.legend()
plt.xticks(rotation=45)
plt.subplots_adjust(bottom = 0.35)
plt.show()






       