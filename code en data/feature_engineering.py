# Data Mining Techniques
# Koen Smallegange 
# april 2023
# 
# Feature engineering
# 
# ---------------------------------------------------------------------------------------------------------

from main import df

print(df)

for index, row in df.iterrows():
    cpgt = df.at[index, 'used chatGPT']
    
    if 'yes' in cpgt:
        program = df.at[index, 'programme']
        print(cpgt, program, index)
       











