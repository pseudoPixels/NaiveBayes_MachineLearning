import numpy as np

#loading the data source file
dataFile = open('DataFiles\Source_1_50_exemplars.txt', 'r')
dataSet = np.loadtxt(dataFile)

#total number of samples available
total_number_of_samples = len(dataSet)



#find out the possible class labels in the data set. Like Class 0, 1 , 2 etc.
classLabels = np.unique(dataSet[:, 0].astype(int))



#Counting the occurance of different classes in the data source
classBins = np.bincount(dataSet[:, 0].astype(int))



#calculating the pi value for each class --how many times the class i are found in whole data set.
pi_c = np.divide(classBins, total_number_of_samples)




#finding out number of different classes... in this case 4 i.e. 0,1,2,3
#and also finding out possible attribute values for this particular attribute
#in this case it is 3 i.e 0,1,2
number_of_classes = classLabels.shape[0]
number_of_attributes = np.unique(dataSet[:, 1].astype(int)).shape[0]




#all the 4x3 theta values will be stored here. p(attribute = 1|class = 4) will be
#stored in theta[4][1] and so on...
theta = np.zeros([number_of_classes,number_of_attributes])
for row in range(total_number_of_samples):
    theta[dataSet[row][0].astype(int)][dataSet[row][1].astype(int)] +=1
#Normalizing the theta counts by individual class bins.
theta = theta / classBins.reshape(4,1)




print('Class Labels: ', classLabels)
print('Class Bins: ',classBins)
print('Class Prior or pi: ',pi_c)
print('Theta Vals: ', theta)


