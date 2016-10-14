import numpy as np

#loading the data source file
dataFile = open('DataFiles\Source_1_50_exemplars.txt', 'r')
dataSet = np.loadtxt(dataFile)

#total number of samples available
total_number_of_samples = len(dataSet)


#Counting the occurance of different classes in the data source
classCount = np.bincount(dataSet[:, 0].astype(int))

#calculating the pi value for each class --how many times the class i are found in whole data set.
pi_c = np.divide(classCount, total_number_of_samples)


print(classCount)
print(pi_c)

