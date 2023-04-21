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

# set plotting style 
plt.style.use('seaborn') 
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = 18
plt.rcParams['font.size'] = 18

path = 'data.csv'
df = clean_frame(path)

# ---------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------------------------------------


def drop_non_numeric(df):
    df_numeric = df.select_dtypes(include='number')
    return df_numeric


def replace_outliers_nan(df, z):
    zscores_df = df.select_dtypes(include=['number']).apply(zscore)
    z_threshold = z

    for j in range(len(df)):
        current_zscore = zscores_df.iloc[j]
        current_stress_level = df.iloc[j]['What is your stress level (0-100)?']
        current_sports_hours = df.iloc[j]['How many hours per week do you do sports (in whole hours)?']

        for column in zscores_df.columns:
            if abs(current_zscore[column]) > z_threshold:
                df.at[j, column] = np.nan
                if column == 'What is your stress level (0-100)?':
                    print(f"Replaced {df.at[j, column]} with NaN for row {j}, column {column}. Z-score: {current_zscore[column]}, Stress: {current_stress_level}, Sports: {current_sports_hours}")


    # # based om z score
    # for column in zscores_df.columns:
    #     # df.loc[zscores_df[column].abs() > z_threshold, column] = np.nan
    #     # df[column] = df[column].where(zscores_df[column].abs() <= z_threshold, np.nan)
    #     df.loc[zscores_df[column].abs() > z_threshold, column] = np.nan
      
    # based on numerical limit
    # for column in df.select_dtypes(include=['float64']):
    #     df.loc[df[column] > 100, column] = np.nan

    return df


def impute_nan(df):
    imputer = IterativeImputer(max_iter=25)
    imputed_df = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_df, columns=df.columns)

    return imputed_df


def replace_outliers_median(df, z):
    zscores_df = df.select_dtypes(include=['number']).apply(zscore)
    z_threshold = z

    for column in zscores_df.columns:
        non_outliers = df.loc[(zscores_df[column].abs() <= z_threshold), column]
        median = non_outliers.median()
        df[column] = df[column].where(zscores_df[column].abs() <= z_threshold, median)
    
    return df


def replace_outliers_with_mean(df, z):
    zscores_df = df.select_dtypes(include=['number']).apply(zscore)
    z_threshold = z

    # Replace outliers with the mean of the non-outliers in each column
    for column in zscores_df.columns:
        non_outliers = df.loc[(zscores_df[column].abs() <= z_threshold), column]
        mean = non_outliers.mean()
        df[column] = df[column].where(zscores_df[column].abs() <= z_threshold, mean)
    
    return df


  
def impute(df, z):

    df = replace_outliers_nan(df, z)
    df = drop_non_numeric(df)
    df = impute_nan(df) 
    df.to_csv('clean.csv', index=False)
    return df

    


# ---------------------------------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------------------------------

# set z treshhold 
z = 2.5

# get some clean data
stress_before =  df['What is your stress level (0-100)?']
sports_before =  df['How many hours per week do you do sports (in whole hours)?']

dfmean = clean_frame(path)
dfmean = replace_outliers_median(dfmean, z)

stress_median = dfmean['What is your stress level (0-100)?']
sports_median = dfmean['How many hours per week do you do sports (in whole hours)?']

dfimp = clean_frame(path)
for i in range(0, 15):
    dfimp = impute(dfimp, z)

stress_after = dfimp['What is your stress level (0-100)?']
sports_after = dfimp['How many hours per week do you do sports (in whole hours)?']



# df_median = clean_frame(path)
# stress_median =  replace_outliers_median(df_median, z_stress)['What is your stress level (0-100)?']

# df_median = clean_frame(path)
# sports_median =  replace_outliers_median(df_median, z_sports)['How many hours per week do you do sports (in whole hours)?']

# df_imputed = clean_frame(path)
# stress_after =  impute(df_imputed, z_stress)['What is your stress level (0-100)?']

# df_imputed = clean_frame(path)
# sports_after =  impute(df_imputed, z_sports)['How many hours per week do you do sports (in whole hours)?']


# ---------------------------------------------------------------------------------------------------------
# PLOT SOME
# ---------------------------------------------------------------------------------------------------------

fig, axs = plt.subplots(2, 3)
fig.set_alpha(0.75)
fig.set_size_inches(12, 7)

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

axs[0, 1].set_title(f'Stress levels after imputing with z = {z}')
axs[0, 1].set_xlabel('Student')
axs[0, 1].set_ylabel('Stress level')

axs[0, 2].set_title(f'stress after replacing outliers with median, with z = {z}')
axs[0, 2].set_xlabel('Student')
axs[0, 2].set_ylabel('Stress level')

axs[1, 0].set_title(f'Sport hours before imputing')
axs[1, 0].set_xlabel('Student')
axs[1, 0].set_ylabel('Hours of sports per week')

axs[1, 1].set_title(f'Sport hourse after imputing with z = {z}')
axs[1, 1].set_xlabel('Student')
axs[1, 1].set_ylabel('Hours of sports per week')

axs[1, 2].set_title(f'sports after replacing outliers with median, with z = {z}')
axs[1, 2].set_xlabel('Student')
axs[1, 2].set_ylabel('Hours of sports per week')


# Show the plot
fig.tight_layout()
plt.show()
