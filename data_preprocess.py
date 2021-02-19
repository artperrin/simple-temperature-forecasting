import pandas as pd
import numpy as np

# enter the name of the data file (to be placed in the /data foler), without the extension (assumed to be .csv)
filename = 'weatherMonth'

# read the right data
data = pd.read_csv('data/'+filename+'.csv', sep=',')
print(data.head(15))

# here *the user's* process script


# visualizing and exporting to the right format
print(data.head(15))
data.to_csv('data/'+filename+'processed.csv', header=None, sep=',')