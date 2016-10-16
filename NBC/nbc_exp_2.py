import numpy as np
import math
from scipy.misc import logsumexp


#loading the data source file
dataFile = open('DataFiles\Source_4_1000_exemplars.txt', 'r')
dataSet = np.loadtxt(dataFile)

#print(dataSet[1:,])

#total number of samples available
def get_total_number_of_samples(dataSet):
    total_number_of_samples = len(dataSet)
    return total_number_of_samples



#find out the possible class labels in the data set. Like Class 0, 1 , 2 etc.
def get_class_labels(dataSet):
    classLabels = np.unique(dataSet[:, 0].astype(int))
    return classLabels

def get_number_of_classes(dataSet):
    number_of_classes = get_class_labels(dataSet).shape[0]
    return number_of_classes

#Counting the occurance of different classes in the data source
def get_class_bins(dataSet):
    classBins = np.bincount(dataSet[:, 0].astype(int))
    return classBins



#calculating the pi value for each class --how many times the class i are found in whole data set.
def get_pi_for_mle(dataSet):
    total_number_of_samples = get_total_number_of_samples(dataSet)
    classBins = get_class_bins(dataSet)
    pi_c = np.divide(classBins, total_number_of_samples)
    return pi_c



def get_pi_for_map(dataSet, alpha):
    classBins = get_class_bins(dataSet)
    total_number_of_samples = get_total_number_of_samples(dataSet)

    classBins_with_aplpha_addition = classBins + np.ones(classBins.shape[0])*alpha
    total_number_of_samples_with_all_aplpha_addition = total_number_of_samples + alpha*total_number_of_samples
    pi_c = np.divide(classBins_with_aplpha_addition, total_number_of_samples_with_all_aplpha_addition)
    return  pi_c



#finding out number of different classes... in this case 4 i.e. 0,1,2,3
#and also finding out possible attribute values for this particular attribute
#in this case it is 3 i.e 0,1,2
def get_categories_per_attributes(dataSet):
    categories_per_attributes = np.zeros(dataSet.shape[1]-1)
    for attribute_type in range(dataSet.shape[1]-1):
        categories_per_attributes[attribute_type] = np.unique(dataSet[:, attribute_type+1].astype(int)).shape[0]
    return categories_per_attributes


#all the 4x3 theta values will be stored here. p(attribute = 1|class = 4) will be
#stored in theta[4][1] and so on...
def get_theta_for_mle(dataSet):
    allTheta ={}
    number_of_classes = get_number_of_classes(dataSet)
    categories_per_attributes = get_categories_per_attributes(dataSet)
    for i in range(dataSet.shape[1]-1):
        theta = np.zeros([int(number_of_classes),int(categories_per_attributes[i])])
        for row in range(len(dataSet)):
            theta[dataSet[row][0].astype(int)][dataSet[row][i+1].astype(int)] +=1
        #Normalizing the theta counts by individual class bins.
        theta = theta / get_class_bins(dataSet).reshape(get_class_bins(dataSet).shape[0],1)
        allTheta[i] = theta
    return allTheta



def get_theta_for_map(dataSet, classBins, categories_per_attributes, beta):
    allTheta ={}
    for i in range(dataSet.shape[1]-1):
        theta = np.zeros([int(classBins.shape[0]),int(categories_per_attributes[i])])
        for row in range(len(dataSet)):
            theta[dataSet[row][0].astype(int)][dataSet[row][i+1].astype(int)] +=1
        #Normalizing the theta counts by individual class bins.
        theta = theta / classBins.reshape(classBins.shape[0],1)
        allTheta[i] = theta
    return allTheta


def nbcPredict_with_logsumexp_trick(allTheta, allPi, testSet):
    Li = np.zeros(allPi.shape[0]);
    for aClass in range(allPi.shape[0]):
        Li[aClass] = math.log10(allPi[aClass])
        for anAttribute in range(len(allTheta.keys())):
            if(allTheta[anAttribute][aClass][int(testSet[anAttribute])] != 0):
                Li[aClass] = Li[aClass] + math.log10(allTheta[anAttribute][aClass][int(testSet[anAttribute])])
    pi = np.exp(Li - logsumexp(Li))
    return np.argmax(pi)



def nbcPredict(allTheta, allPi, testSet):
    classProbabilities = np.ones(allPi.shape[0])

    for aClass in range(allPi.shape[0]):
        thetaProduct = 1
        for i in range(testSet.shape[0]):
            thetaProduct *= allTheta[i][aClass][int(testSet[i])]
            #print(allTheta[i][aClass][int(testSet[i])])
        classProbabilities[aClass] = allPi[aClass]*thetaProduct
    return np.argmax(classProbabilities)


#print('Class Labels: ', classLabels)
#print('Class Bins: ',classBins)
#print('Class Prior or pi: ',pi_c)
#print('Theta Vals: ', allTheta)
#allTheta = get_theta_for_mle(dataSet)
#allPie = get_pi_for_mle(get_class_bins(dataSet))



def get_accuracy_via_cross_validation(dataSet):
    correct_prediction_counter = 0
    for row in range(len(dataSet)-1):
        new_dataset_with_a_row_exclusion = np.delete(dataSet, row, 0)
        if(int(dataSet[row][0]) == int(nbcPredict_with_logsumexp_trick(get_theta_for_mle(new_dataset_with_a_row_exclusion), get_pi_for_map(new_dataset_with_a_row_exclusion, .5), dataSet[row][1:]))):
            correct_prediction_counter += 1
    return correct_prediction_counter*100/len(dataSet)

#Testing prediction on first row --dataSet[0][1:]
#print('Predicted Class --> ',nbcPredict(get_theta_for_mle(dataSet), get_pi_for_mle(dataSet), dataSet[0][1:]))
#print('Predicted Class with LogSumExpTrick--> ', nbcPredict_with_logsumexp_trick(allTheta, allPie, dataSet[0][1:]))
#print('all data set', dataSet)
#dataSet = np.delete(dataSet, 0, 0)
#print('all data set', dataSet)


print('Accuracy : ', get_accuracy_via_cross_validation(dataSet), ' %')


