# Data Mining Techniques
# Koen Smallegange 
# april 2023
# 
# This script is used to plot stuff
# 
# ---------------------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt
from main import dfc 
import seaborn as sbn 

# set plotting style
hfont = {'fontname':'Times'}
plt.style.use('seaborn') 

# do some plots
plt.plot(dfc["How many students do you estimate there are in the room?"], 'o')
plt.show()