
# Door: Reinout Mensing
# 
# Script takes a dirty csv and puts out a clean csv
# 
# ----------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

def binn_stress_levels(df):
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    
    df['What is your stress level (0-100)?'] = pd.cut(df['What is your stress level (0-100)?'], bins=bins, labels=labels)
      
    return df


def binn_sports(df):
    bins = [0,0.01, 2, 4, 6 , 10, 15, 20]
    labels = ['0','0-2','2-4','4-6','6-10','10-15','15+']

    df['How many hours per week do you do sports (in whole hours)?'] = pd.cut(
        df['How many hours per week do you do sports (in whole hours)?'], bins=bins, labels=labels)
    
    return df


def binn_room_estimates(df):
    bins = [-1, 50, 100, 200, 300, 400, 500, 1000, float('inf')]
    labels = ['0-50', '51-100', '101-200', '201-300', '301-400', '401-500', '501-1000', '1001+']

    df['How many students do you estimate there are in the room?'] = pd.cut(
        df['How many students do you estimate there are in the room?'], bins=bins, labels=labels)

    return df


def bin(df):
    df = binn_room_estimates(df)
    df = binn_sports(df)
    df = binn_stress_levels(df)

    return df

