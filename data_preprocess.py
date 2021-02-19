import pandas as pd
import numpy as np

# enter the name of the data file (to be placed in the /data foler), without the extension (assumed to be .csv)
filename = 'weatherMonth'

# read the right data
data = pd.read_csv('data/'+filename+'.csv', sep=',')
print(data.head(15))

# here *the user's* process script
temp = data['LandAverageTemperature'].to_numpy()

temp = temp[300:]

for i in range(len(temp)-1): # if there are no values, assume that the temperature is the mean of the previous and next one
    if np.isnan(temp[i]):
        nex = 0
        for j in range(i,len(temp)-1):
            if not np.isnan(temp[j]):
                nex = temp[j]
                break
        temp[i] = (temp[i-1]+nex)/2

Temp = [round(sum(temp[i:i+6])/6,4) for i in range(0,len(temp)-6,6)]

data = pd.DataFrame(Temp)

# visualizing and exporting to the right format
print(data.head(15))
# data.to_csv('data/'+filename+'processed.csv', header=None, sep=',')
data.to_csv('data/weatherSemester_processed.csv', header=None, sep=',')