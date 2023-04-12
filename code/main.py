# Data Mining Techniques
# Reinout Mensing, Margot Boekema, Koen Smallegange 
# april 2023
# ---------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



##TASKS
# 1. DATA PREPERATION -----------------------------------------
##1A. EXPLORATORY DATA ANALYSIS.

df = pd.read_csv("data.csv", delimiter = ";", error_bad_lines='false')
df.rename(
    columns=({ 'Tijdstempel': 'time', 'What is your gender?': 'gender', 'What programme are you in?': 'programme',
               'Have you taken a course on machine learning?': 'ML course',
               'Have you taken a course on information retrieval?': 'IR course', 'When is your birthday (date)?': 'birthday',
               'How many students do you estimate there are in the room?':'student guess',
               'Did you stand up to come to your previous answer?': 'stand up', 'What is your stress level (0-100)?':'stress level',
               'Have you taken a course on statistics?': 'statistics course', 'Have you taken a course on databases?' : 'databases course',
               'I have used ChatGPT to help me with some of my study assignments' : 'used chatGPT',
               'How many hours per week do you do sports (in whole hours)?': 'sports hours',
               'Give a random number': 'random number', 'Time you went to bed Yesterday':'bed time',
               'What makes a good day for you (1)?': 'good day 1', 'What makes a good day for you (2)?': 'good day 2'}),
    inplace=True,
)


print(f"Number of records: {len(df)}")
print(f"Number of attributes: {len(df.columns)}")

# Get the names of the attributes
print("Attribute names:")
print(list(df.columns))

# Get the data types of the attributes
print("Attribute data types:")
print(df.dtypes)

print("Summary statistics for numerical attributes:")
print(df.describe())

# Get the number of missing values for each attribute
print("Number of missing values for each attribute:")
print(df.isnull().sum())

# Get the percentage of missing values for each attribute
print("Percentage of missing values for each attribute:")
print(df.isnull().sum() / len(df) * 100)

# Plot a histogram of a numerical attribute
sns.histplot(df['student guess'], bins=20)
plt.title("Distribution of estimated number of students in the room")
plt.xlabel("Number of students")
plt.ylabel("Count")
plt.show()

# Plot a scatter plot of two numerical attributes
sns.scatterplot(x='stress level', y='bed time', data=df)
plt.title("Relationship between stress level (0-100) and hours of sleep")
plt.xlabel("stress level")
plt.ylabel("hours of sleep")
plt.show()

# Plot a bar chart of a categorical attribute
sns.countplot(x='statistics course', data=df)
plt.title("Frequency of statistics course")
plt.xlabel("statistics course")
plt.ylabel("Count")
plt.show()

# Plot a box plot of a numerical attribute grouped by a categorical attribute
sns.boxplot(x='statistics course', y='student guess', data=df)
plt.title("Distribution of statistics course by student guess")
plt.xlabel("statistics course")
plt.ylabel("student guess")
plt.show()

