
# Door: Reinout Mensing
# 
# Script takes a dirty csv and puts out a clean csv
# 
# ----------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import zscore
import re
#from fancyimpute import IterativeImputer

def calculate_zscores(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Calculate the Z-scores for each numerical column
    zscores_df = df.select_dtypes(include=['number']).apply(zscore)

    return zscores_df

def replace_outliers_with_mean(df, z_threshold=3):
    # Calculate the Z-scores for each numerical column
    zscores_df = df.select_dtypes(include=['number']).apply(zscore)
    
    # Replace outliers with the mean of the non-outliers in each column
    for column in zscores_df.columns:
        non_outliers = df.loc[(zscores_df[column].abs() <= z_threshold), column]
        mean = non_outliers.mean()
        df[column] = df[column].where(zscores_df[column].abs() <= z_threshold, mean)
    
    return df

def replace_outliers_with_median(df, z_threshold=3):
    # Calculate the Z-scores for each numerical column
    zscores_df = df.select_dtypes(include=['number']).apply(zscore)

    # Replace outliers with the median of the non-outliers in each column
    for column in zscores_df.columns:
        non_outliers = df.loc[(zscores_df[column].abs() <= z_threshold), column]
        median = non_outliers.median()
        df[column] = df[column].where(zscores_df[column].abs() <= z_threshold, median)

    return df

def multiple_imputation(df, max_iter=10):
    # The IterativeImputer class requires the data to be in a numeric format
    # Make a copy of the original DataFrame and convert non-numeric columns to numeric using pd.to_numeric
    numeric_df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            numeric_df[col] = pd.to_numeric(df[col], errors='coerce')

    # Perform multiple imputation
    imputer = IterativeImputer(max_iter=max_iter)
    imputed_data = imputer.fit_transform(numeric_df)

    # Create a new DataFrame with the imputed values and the original column names and index
    imputed_df = pd.DataFrame(imputed_data, columns=df.columns, index=df.index)

    # Round the imputed values (optional)
    imputed_df = imputed_df.round()

    return imputed_df

def read(path):
    # Read the dataset
    df = pd.read_csv(path, sep=';')

    # Remove leading and trailing whitespace from column names
    df.columns = df.columns.str.strip()

    # Convert 'yes', 'no', 'unknown' to 1, 0, and np.nan
    yes_no_mapping = {'yes': 1, 'no': 0}
    male_female_mapping = {'male': 1, 'female':0}
    information_retrieval = {'1': 1, '0':0}
    database_retrieval = {'ja': 1, 'nee': 0}
    df['Have you taken a course on machine learning?'] = df['Have you taken a course on machine learning?'].map(yes_no_mapping)
    df['Have you taken a course on information retrieval?'] = df['Have you taken a course on information retrieval?'].replace(information_retrieval)
    df['Have you taken a course on information retrieval?'] = df['Have you taken a course on information retrieval?'].where(df['Have you taken a course on information retrieval?'].isin([0, 1]), None)
    df['Have you taken a course on statistics?'] = df['Have you taken a course on statistics?'].map(lambda x: 1 if x == 'mu' else 0)
    df['Have you taken a course on databases?'] = df['Have you taken a course on databases?'].map(database_retrieval)
    df['When is your birthday (date)?'] = pd.to_datetime(df['When is your birthday (date)?'], errors='coerce')
    df['Did you stand up to come to your previous answer    ?'] = df['Did you stand up to come to your previous answer    ?'].map(yes_no_mapping)
    df['I have used ChatGPT to help me with some of my study assignments'] = df['I have used ChatGPT to help me with some of my study assignments'].map(yes_no_mapping)
    df['What is your gender?'] = df['What is your gender?'].map(male_female_mapping)

    return df

def process_time_bed():
    pass

def process_stress_levels(df):
    # Define the bins and labels
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']

    df['What is your stress level (0-100)?'] = pd.to_numeric(
        df['What is your stress level (0-100)?'], errors='coerce')
    
    # Assign the bins
    df['What is your stress level (0-100)?'] = pd.cut(df['What is your stress level (0-100)?'], bins=bins, labels=labels)
    
    # Remove rows with NaN values in the 'Stress Level Range' column
    df = df.dropna(subset=['What is your stress level (0-100)?'])
    
    return df

def convert_to_24h(time_str):
        time_str = time_str.strip().lower()
        time_str = re.sub(r"[^0-9a-z:]", "", time_str)  # Remove non-alphanumeric and non-colon characters
        time_str = re.sub(r":", "", time_str)
        
        if len(time_str) == 4 and time_str.isdigit():
            hours = int(time_str[:2])
            minutes = int(time_str[2:])
            if 0 <= hours < 24 and 0 <= minutes < 60:
                return f"{hours:02d}:{minutes:02d}"
        elif len(time_str) <= 2 and time_str.isdigit():
            hours = int(time_str)
            if 0 <= hours < 24:
                return f"{hours:02d}:00"
        
        match = re.match(r"(\d{1,2}):(\d{2})([ap]m)?", time_str)
        if match:
            hours, minutes, am_pm = match.groups()
            hours = int(hours)
            minutes = int(minutes)
            
            if am_pm:
                if am_pm == "pm" and hours != 12:
                    hours += 12
                elif am_pm == "am" and hours == 12:
                    hours = 0
            
            if 0 <= hours < 24 and 0 <= minutes < 60:
                return f"{hours:02d}:{minutes:02d}"
        
        return None

def bin_time(time_str):
    if time_str is None:
        return None
    hours, minutes = map(int, time_str.split(':'))
    
    if 0 <= minutes < 30:
        minutes_range = "00-29"
    else:
        minutes_range = "30-59"
    
    return f"{hours:02d}:{minutes_range}"

def process_bed_times(df):
    df = df.copy()
    df['Time you went to bed Yesterday'] = df['Time you went to bed Yesterday'].apply(lambda x: convert_to_24h(str(x)))
    df['Time you went to bed Yesterday'] = df['Time you went to bed Yesterday'].apply(bin_time)
    return df

def process_sports(df):
    bins = [0,0.01, 2, 4, 6 , 10, 15, 20]
    labels = ['0','0-2','2-4','4-6','6-10','10-15','15+']
    df['How many hours per week do you do sports (in whole hours)?'] = pd.to_numeric(
        df['How many hours per week do you do sports (in whole hours)?'], errors='coerce')
        # Replace the values with the corresponding range
    df['How many hours per week do you do sports (in whole hours)?'] = pd.cut(
        df['How many hours per week do you do sports (in whole hours)?'], bins=bins, labels=labels)
    return df


def process_room_estimates(df):
    # Convert non-numeric values to NaN
    df['How many students do you estimate there are in the room?'] = pd.to_numeric(
        df['How many students do you estimate there are in the room?'], errors='coerce')

    # Remove unreasonable values (below zero)
    df = df[df['How many students do you estimate there are in the room?'] >= 0]

    # Define the bins for the ranges
    bins = [-1, 50, 100, 200, 300, 400, 500, 1000, float('inf')]
    labels = ['0-50', '51-100', '101-200', '201-300', '301-400', '401-500', '501-1000', '1001+']

    # Replace the values with the corresponding range
    df['How many students do you estimate there are in the room?'] = pd.cut(
        df['How many students do you estimate there are in the room?'], bins=bins, labels=labels)

    return df

def bin_programmes(df):
    '''
    Classifies all program names and changes them in the df
    '''
  
    for index, row in df.iterrows():
    
        program = df.at[index, 'What programme are you in?']

        if 'AI' in program or 'ntelligence' in program:
            df.at[index, 'What programme are you in?'] = "Artificial Intelligence"
        
        elif 'conometrics' in program: 
            df.at[index, 'What programme are you in?'] = 'Econometrics'
        
        elif 'omputational' in program or 'CLS' in program:
            df.at[index, 'What programme are you in?'] = "Computational Science"
        
        elif 'antitative' in program or 'QRM' in program: 
            df.at[index, 'What programme are you in?'] = "Quantitative Risk Management"
        
        elif 'nalytics' in program or 'BA' in program:
            df.at[index, 'What programme are you in?'] = "Business Analytics"

        elif 'omputer' in program or 'CS' in program:
            df.at[index, 'What programme are you in?'] = "Computer Science"

        elif 'fin' in program: 
            df.at[index, 'What programme are you in?'] = "Finance and Technology"
        
        elif 'io' in program: 
            df.at[index, 'What programme are you in?'] = "Bioinformatics"
        
        elif 'ex' in program:
            df.at[index, 'What programme are you in?'] = "Exhange Programme"
        
        elif 'euro' in program: 
            df.at[index, 'What programme are you in?'] = "Neuroscience"
        
        elif 'phd' in program or 'PhD' in program: 
            df.at[index, 'What programme are you in?'] = "PhD"
        
        elif 'life' in program:
            df.at[index, 'What programme are you in?'] = "Life Sciences"
        
        else: 
            df.at[index, 'What programme are you in?'] = "Other"

    return df


def clean_frame(path):
    df = read(path)
    df = process_stress_levels(df)
    df = process_room_estimates(df)
    df = process_sports(df)
    df = process_bed_times(df)
    df = bin_programmes(df)

    df.to_csv('clean.csv', index=False)
    # Calculate Z-scores and replace outliers (using the median-based approach)
    zscores_df = df.select_dtypes(include=['number']).apply(zscore)
    z_threshold = 3
    for column in zscores_df.columns:
        non_outliers = df.loc[(zscores_df[column].abs() <= z_threshold), column]
        median = non_outliers.median()
        df[column] = df[column].where(zscores_df[column].abs() <= z_threshold, median)

    return df

