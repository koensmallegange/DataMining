# Data Mining Techniques
# Koen Smallegange 
# april 2023
# 
# This script controls all code
# 

# ---------------------------------------------------------------------------------------------------------
# ADMIN
# ---------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
from clean import clean_frame
from scipy.stats import zscore
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import sys
import numpy as np

# ---------------------------------------------------------------------------------------------------------
# set paramters
# ---------------------------------------------------------------------------------------------------------

z = 2.5
iterations = 25


# set plotting style 
plt.style.use('seaborn') 
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = 18
plt.rcParams['font.size'] = 18


# get data
path = 'data.csv'
df = clean_frame(path)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_zscores(df):
    '''
    calculates z scores for a dataframe
    '''
    zscores_df = df.select_dtypes(include=['number']).apply(zscore)

    return zscores_df


def drop_non_numeric(df):
    '''
    removes al non-numeric collumns from dataframe and returns new dataframe
    '''
    df_numeric = df.select_dtypes(include='number')
    return df_numeric


def replace_outliers_nan(df, z):
    '''
    replaces all outliers in numeric dataframe with NaN values
    '''
    zscores_df = df.select_dtypes(include=['number']).apply(zscore)
    z_threshold = z

    for j in range(len(df)):
        current_zscore = zscores_df.iloc[j]
        current_stress_level = df.iloc[j]['What is your stress level (0-100)?']
        current_sports_hours = df.iloc[j]['How many hours per week do you do sports (in whole hours)?']

        for column in zscores_df.columns:
            if abs(current_zscore[column]) > z_threshold:
                df.at[j, column] = np.nan

                # if column == 'What is your stress level (0-100)?':
                    # print(f"Replaced {df.at[j, column]} with NaN for row {j}, column {column}. Z-score: {current_zscore[column]}, Stress: {current_stress_level}, Sports: {current_sports_hours}")

    return df


def impute_nan(df, iters):
    '''
    imputes all NaN values using Iterative imputer. Returns imputed df
    '''
    imputer = IterativeImputer(max_iter=iters)
    imputed_df = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_df, columns=df.columns)

    return imputed_df


def replace_outliers_median(df, z):
    '''
    replaces al outliers with the median of the remaining df 
    '''
    zscores_df = df.select_dtypes(include=['number']).apply(zscore)
    z_threshold = z

    for column in zscores_df.columns:
        non_outliers = df.loc[(zscores_df[column].abs() <= z_threshold), column]
        median = non_outliers.median()
        df[column] = df[column].where(zscores_df[column].abs() <= z_threshold, median)
    
    return df


def replace_outliers_with_mean(df, z):
    '''
    replaces all outliers with mean of remaining df
    '''
    zscores_df = df.select_dtypes(include=['number']).apply(zscore)
    z_threshold = z

    for column in zscores_df.columns:
        non_outliers = df.loc[(zscores_df[column].abs() <= z_threshold), column]
        mean = non_outliers.mean()
        df[column] = df[column].where(zscores_df[column].abs() <= z_threshold, mean)
    
    return df


def impute(df, z, iters):
    '''
    controls the imputation process and saves the imputed data
    '''

    df = replace_outliers_nan(df, z)
    df = drop_non_numeric(df)
    df = impute_nan(df, iters) 

    return df
    

def plot_zscores(df): 
    '''
    this function plots the zscores of stress and sports
    '''

    zscore = calculate_zscores(df)
    
    # set figure
    fig, axs = plt.subplots(1, 2)
    fig.set_alpha(0.75)
    fig.set_size_inches(12, 7)
    fig.suptitle('Z scores of different distributions before imputing')

    axs[0].plot(zscore['What is your stress level (0-100)?'], 'o', markersize = 5)
    axs[1].plot(zscore['How many hours per week do you do sports (in whole hours)?'], 'o', markersize = 5)

    axs[0].set_title('Z scores of stress levels')
    axs[0].set_xlabel('Student')
    axs[0].set_ylabel('Z score')

    axs[1].set_title('Z scores of sporting hours')
    axs[1].set_xlabel('Student')
    axs[1].set_ylabel('Z score')

    # Show the plot
    fig.tight_layout()
    plt.show()
        

def plot_imputaiton_results(df, z, iterations):
    '''
    This function does everything from imputing to plotting 
    is is computationally havy, so I you do not require plots do not use it. 
    '''

    # get clean colloumn for plot 
    stress_before =  df['What is your stress level (0-100)?']
    sports_before =  df['How many hours per week do you do sports (in whole hours)?']


    # clean collumns with median for plot
    dfmean = clean_frame(path)
    dfmean = replace_outliers_median(dfmean, z)

    stress_median = dfmean['What is your stress level (0-100)?']
    sports_median = dfmean['How many hours per week do you do sports (in whole hours)?']


    # clean data with multiple iterations for plot
    dfimp = clean_frame(path)

    for i in range(0, iterations):
        dfimp = impute(dfimp, z, iterations)

    stress_after = dfimp['What is your stress level (0-100)?']
    sports_after = dfimp['How many hours per week do you do sports (in whole hours)?']


    # set figure
    fig, axs = plt.subplots(2, 3)
    fig.set_alpha(0.75)
    fig.set_size_inches(12, 7)
    fig.suptitle('Comparison of different imputation techniques on stress levels and sporting hours')

    # Plot the data in each subplot
    axs[0, 0].plot(range(0, len(stress_before)), stress_before, 'o', markersize = 5, label = 'stress before')
    axs[0, 1].plot(range(0, len(stress_after)), stress_after, 'o', markersize = 5, label = 'stress after imputing')
    axs[0, 2].plot(range(0, len(stress_median)), stress_median, 'o', markersize = 5, label = 'stress after replacing outliers with median')
    axs[1, 0].plot(range(0, len(sports_before)), sports_before, 'o', markersize = 5, label = 'sports before')
    axs[1, 1].plot(range(0, len(sports_after)), sports_after, 'o', markersize = 5, label = 'sports after')
    axs[1, 2].plot(range(0, len(sports_median)), sports_median, 'o', markersize = 5, label = 'sports after replacing outliers with median')

    # Add titles to each subplot
    axs[0, 0].set_title(f'Stress levels before imputing')
    axs[0, 0].set_xlabel('Student')
    axs[0, 0].set_ylabel('Stress level')

    axs[0, 1].set_title(f'Stress levels after imputing with z = {z} and {iterations} iterations')
    axs[0, 1].set_xlabel('Student')
    axs[0, 1].set_ylabel('Stress level')

    axs[0, 2].set_title(f'stress after replacing outliers with median, with z = {z}')
    axs[0, 2].set_xlabel('Student')
    axs[0, 2].set_ylabel('Stress level')

    axs[1, 0].set_title(f'Sport hours before imputing')
    axs[1, 0].set_xlabel('Student')
    axs[1, 0].set_ylabel('Hours of sports per week')

    axs[1, 1].set_title(f'Sport hourse after imputing with z = {z} and {iterations} iterations')
    axs[1, 1].set_xlabel('Student')
    axs[1, 1].set_ylabel('Hours of sports per week')

    axs[1, 2].set_title(f'sports after replacing outliers with median, with z = {z}')
    axs[1, 2].set_xlabel('Student')
    axs[1, 2].set_ylabel('Hours of sports per week')


    # Show the plot
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# make some plots
# plot_zscores(df)
# plot_imputaiton_results(df, z, iterations)


# impute datetimes 
df['Tijdstempel'] = df['Tijdstempel'].bfill()
df['When is your birthday (date)?'] = df['When is your birthday (date)?'].bfill()
df['Time you went to bed Yesterday'] = df['Time you went to bed Yesterday'].bfill()


# impute remainder of dataframe 
df_imp = df

for i in range(0, iterations):
    df_imp = impute(df_imp, z, iterations)


# get the headers for merging
columns = df_imp.columns.values
columns = list(columns)
 

#  merge imputed columns of dataframe with non-imputed columns of dataframe
dfc = pd.merge(df, df_imp, on=columns, how = 'inner')


# save the file
dfc.to_csv('imputed.csv', index=False)






