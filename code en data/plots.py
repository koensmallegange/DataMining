# Data Mining Techniques
# Margot Boekema
# april 2023
#
# This script is for plotting
#
# ---------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import numpy as np
import re
from scipy.stats import zscore
# from fancyimpute import IterativeImputer
from prettytable import PrettyTable
import sys 

dfc = pd.read_csv('clean.csv')
df = pd.read_csv('data.csv', delimiter=";")
#on_bad_lines='skip' if it gives an error


# set plotting style 
plt.style.use('seaborn') 
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = 18
plt.rcParams['font.size'] = 18

def replace_non_numeric(df, column_name, replace_value=np.nan):
    """
    This function removes all letters and symbols from a column of a pandas DataFrame, then replaces all
    non-numeric values with a given value.

    Arguments:
    df -- the pandas DataFrame
    column_name -- the name of the column to remove letters and symbols from
    replace_value -- the value to replace non-numeric values with (default is NaN)

    Returns:
    The adjusted pandas DataFrame.
    """
    # Remove all letters and symbols from the column using regular expressions
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^0-9\.]+', '', str(x)))

    # Convert the column to a numeric data type, replacing non-numeric values with NaN
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

    # Replace all NaN values with the replace_value argument
    df[column_name] = df[column_name].fillna(replace_value)

    return df


def remove_words(text, words_to_remove):
        """
        This function removes words from a text based on a list of words to remove.
        """
        words_to_remove = set(words_to_remove)
        # Split the text into a list of words
        words = text.split()

        # Remove the words from the list
        words = [word for word in words if word.lower() not in words_to_remove]

        # Join the words back into a text
        text = " ".join(words)

        return text


def most_frequent_words(col1, col2, n):
    """
    Returns a list of the n most frequent words in two columns of text data combined.

    Arguments:
    col1 -- a list or array of text data
    col2 -- a list or array of text data
    n -- the number of most frequent words to return

    Returns:
    A list of tuples containing the most frequent words and their counts.
    """
    #columns to strings
    col1_strings = [str(row) for row in col1]
    col2_strings = [str(row) for row in col2]
    # split the words in both columns
    col1_words = [word.lower() for row in col1_strings for word in row.split()]
    col2_words = [word.lower() for row in col2_strings for word in row.split()]
    # combine the two columns of words
    combined_words = col1_words + col2_words
    word_counts = Counter(combined_words)
    most_common = word_counts.most_common(n)

    return most_common


def replace_outliers(data, column_name, threshold):
    """
    This function replaces all values in a column of a pandas DataFrame that are above a certain threshold with the threshold value.

    Arguments:
    df -- the pandas DataFrame
    column_name -- the name of the column to replace values in
    threshold -- the threshold value

    Returns:
    The adjusted pandas DataFrame.
    """
    # Loop through the values in the specified column and replace values above the threshold with the threshold value
    for i, value in enumerate(data[column_name]):
        if value > threshold:
            data[column_name][i] = threshold

    # Return the adjusted DataFrame
    return data

def scatterplot(x,y):

    """
    Fancy scatter plot of 2 values

    """
    plt.scatter(x, y, s=100, c='red', alpha=0.5, edgecolors='black', linewidths=1)

    # Add a title and axis labels
    plt.title('Scatter Plot')
    plt.xlabel('x')
    plt.ylabel('y')

    # Add a grid and adjust the plot limits
    plt.grid(True)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])

    # Add a colorbar legend
    cbar = plt.colorbar()
    cbar.set_label('Colorbar label')

    # Show the plot
    plt.show()

def get_value_counts(df, column_name):
    """
    Returns a dictionary with the count of each unique value in the specified column of a pandas dataframe.
    """
    value_counts_dict = df[column_name].value_counts().to_dict()
    return value_counts_dict


# ---------------------------------------------------------------------------------------------------------------------------


#Calling out functions
print(dfc.dtypes)
#'Have you taken a course on machine learning?', 'Have you taken a course on information retrieval?',
#            'Have you taken a course on statistics?','Have you taken a course on databases?'
replace_non_numeric(df, 'Have you taken a course on machine learning?', replace_value=np.nan)
print(df['Have you taken a course on machine learning?'])
remove_word = ['good', 'nice']
text = " ".join(dfc['What makes a good day for you (1)?'].astype(str)) + " " + " ".join(dfc['What makes a good day for you (2)?'].astype(str))
text = remove_words(text, remove_word)
most_frequent = most_frequent_words(dfc['What makes a good day for you (1)?'], dfc['What makes a good day for you (2)?'], 5)
replace_non_numeric(df,"How many hours per week do you do sports (in whole hours)? ")
replace_non_numeric(df,"What is your stress level (0-100)?")
replace_outliers(df, "How many hours per week do you do sports (in whole hours)? ", 25)
replace_outliers(df, "What is your stress level (0-100)?", 100)

#Tables
#Table most frequent words in good day questions
print('Most frequent words: ')
print(most_frequent)
table = PrettyTable()
table.field_names = ['Word', 'Frequency']
for row in most_frequent:
    table.add_row(row)
print(table)

words = [row[0] for row in most_frequent]
frequencies = [row[1] for row in most_frequent]

# Create a bar chart
plt.bar(words, frequencies, alpha = 0.75)
plt.ylabel('Frequency')
plt.title('Most Frequent Words in Good Day Questions')
plt.xticks(rotation=90)
plt.show()



#Plotting
# Wordcloud
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(text)

plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

# #Scatterplot
plt.figure()
sns.scatterplot(x= df["How many hours per week do you do sports (in whole hours)? "], y= df["What is your stress level (0-100)?"], alpha = 0.75)
plt.title("Scatter plot of weekly hours of sports vs. stress level?")
plt.xlabel("Sports hours")
plt.ylabel("Stress level")
plt.show()

#scatterplot(df['How many hours per week do you do sports (in whole hours)? '], df['What is your stress level (0-100)?'])
#scatterplot(dfc['Time you went to bed Yesterday'], df['What is your stress level (0-100)?'])

# #Bar chart of programme counts
programme_counts = get_value_counts(dfc, 'What programme are you in?')
print(programme_counts)
sns.barplot(y=list(programme_counts.values()), x=list(programme_counts.keys()), alpha = 0.75)
plt.title('Number of students in each study programme')
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom = 0.35)
plt.ylabel('Count')
plt.show()

# #Heatmap of course correlations
corr = dfc[['Have you taken a course on machine learning?', 'Have you taken a course on information retrieval?',
            'Have you taken a course on statistics?','Have you taken a course on databases?']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', xticklabels=['Machine Learning', 'Information Retrievel', 'Statistics', 'Databases'], yticklabels=['Machine Learning', 'Information Retrievel', 'Statistics', 'Databases'], alpha = 0.85)
plt.title('Correlation between students taking different courses')
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom = 0.25)
plt.subplots_adjust(left = 0.2)
plt.show()

# Counts the number of students who have used ChatGPT
chatgpt_counts = dfc['I have used ChatGPT to help me with some of my study assignments'].value_counts()
plt.pie(chatgpt_counts, labels=chatgpt_counts.index)
plt.title('Proportion of Students who have used ChatGPT with a study assignment')
plt.show()

# Heatmap of correlations between stress levels (Dit kan interessant zijn, moeten alleen een paar data trasnformaties voor gebeuren)
# corr = dfc[['What is your gender?', 'What is your stress level (0-100)?',
#            'How many hours per week do you do sports (in whole hours)?','Time you went to bed Yesterday']].corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm', xticklabels=['Gender', 'Stress level', 'Hours of sports', 'Time to bed'], yticklabels=['Gender', 'Stress level', 'Hours of sports', 'Time to bed'])
# plt.title('Correlation Between stress indicators')
# plt.xlabel('Course')
# plt.ylabel('Course')
# plt.show()


# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# plotting some connections 

# plot average stress per program

# Extract the lower and upper bounds of stress levels as separate columns
dfc['StressLevelLower'] = dfc['What is your stress level (0-100)?'].str.split('-', expand=True)[0].astype(float)
dfc['StressLevelUpper'] = dfc['What is your stress level (0-100)?'].str.split('-', expand=True)[1].astype(float)

# Calculate the average stress level per program
dfc['AverageStressLevel'] = (dfc['StressLevelLower'] + dfc['StressLevelUpper']) / 2
avg_stress_by_program = dfc.groupby('What programme are you in?')['AverageStressLevel'].mean()

# Plot the average stress level per program
avg_stress_by_program.plot(kind='bar', alpha = 0.75)
plt.ylabel('Average Stress Level')
plt.title('Average Stress Level by Program')
plt.subplots_adjust(bottom = 0.35)
plt.show()


# Extract the lower and upper bounds of stress levels as separate columns
dfc['StressLevelLower'] = dfc['What is your stress level (0-100)?'].str.split('-', expand=True)[0].astype(float)
dfc['StressLevelUpper'] = dfc['What is your stress level (0-100)?'].str.split('-', expand=True)[1].astype(float)

# Calculate the average stress level per program
dfc['AverageStressLevel'] = (dfc['StressLevelLower'] + dfc['StressLevelUpper']) / 2
avg_stress_by_program = dfc.groupby('What programme are you in?')['AverageStressLevel'].mean()

# Plot the average stress level per program
avg_stress_by_program.plot(kind='bar', alpha = 0.75)
plt.ylabel('Average Stress Level')
plt.title('Average Stress Level by Program')
plt.subplots_adjust(bottom = 0.35)
plt.show()

# plot stdev of students estimate per program

# Create the DataFrame
df = dfc

# Extract lower and upper bounds of estimate range
df['Estimate_Lower'] = df['How many students do you estimate there are in the room?'].str.split('-').str[0].astype(float)
df['Estimate_Upper'] = df['How many students do you estimate there are in the room?'].str.split('-').str[1].str.split().str[0].astype(float)

# Calculate middle of estimate range
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

