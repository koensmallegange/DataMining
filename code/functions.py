
def bin_programmes(df):
    '''
    Classifies all program names and changes them in the df
    '''


    for index, row in df.iterrows():
        program = df.at[index, 'programme']

        if 'AI' in program or 'ntelligence' in program:
            df.at[index, 'programme'] = "Artificial Intelligence"
        
        elif 'econometrics' in program: 
            df.at[index, 'programme'] = 'Econometrics'
        
        elif 'computational' in program or 'CLS' in program:
            df.at[index, 'programme'] = "Computational Science"
        
        elif 'quantitative' in program or 'QRM' in program: 
            df.at[index, 'programme'] = "Quantitative Risk Management"
        
        elif 'analytics' in program or 'BA' in program:
            df.at[index, 'programme'] = "Business Analytics"

        elif 'computer' in program or 'CS' in program:
            df.at[index, 'programme'] = "Computer Science"

        elif 'fin' in program: 
            df.at[index, 'programme'] = "Finance and Technology"
        
        elif 'bio' in program: 
            df.at[index, 'programme'] = "Bioinformatics"
        
        elif 'ex' in program:
            df.at[index, 'programme'] = "Exhange Programme"
        
        elif 'neuro' in program: 
            df.at[index, 'programme'] = "Neuroscience"
        
        elif 'phd' in program: 
            df.at[index, 'programme'] = "PhD"
        
        elif 'life' in program:
            df.at[index, 'programme'] = "Life Sciences"


        
    return df