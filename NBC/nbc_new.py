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
number_of_attributes = np.zeros(dataSet.shape[1]-1)
for attribute_type in range(dataSet.shape[1]-1):
    number_of_attributes[attribute_type] = np.unique(dataSet[:, attribute_type+1].astype(int)).shape[0]

#print(number_of_attributes)


#all the 4x3 theta values will be stored here. p(attribute = 1|class = 4) will be
#stored in theta[4][1] and so on...
allTheta ={}
for i in range(dataSet.shape[1]-1):
    theta = np.zeros([int(number_of_classes),int(number_of_attributes[i])])
    for row in range(total_number_of_samples):
        theta[dataSet[row][0].astype(int)][dataSet[row][i+1].astype(int)] +=1
    #Normalizing the theta counts by individual class bins.
    theta = theta / classBins.reshape(classBins.shape[0],1)
    allTheta[i] = theta


def nbcPredict(allTheta, allPi, testSet):
    classProbabilities = np.ones(classLabels.shape[0])

    for aClass in range(classLabels.shape[0]):
        thetaProduct = 1
        for i in range(testSet.shape[0]):
            thetaProduct *= allTheta[i][aClass][int(testSet[i])]
            #print(allTheta[i][aClass][int(testSet[i])])
        classProbabilities[aClass] = allPi[aClass]*thetaProduct
    return classProbabilities


print('Class Labels: ', classLabels)
#print('Class Bins: ',classBins)
#print('Class Prior or pi: ',pi_c)
#print('Theta Vals: ', allTheta)
print(np.argmax(nbcPredict(allTheta, pi_c, dataSet[3][1:])))

#print(dataSet[0][1:])


