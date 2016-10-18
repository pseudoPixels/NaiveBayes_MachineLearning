import pandas as pd
import numpy as np
import math
import operator
from collections import Counter
from matplotlib import pyplot as plt
from scipy.misc import logsumexp

dataFrame = pd.read_csv('DataFiles\Source_1_100_exemplars.txt', sep="\t", header=None)
dataFrame.columns = ['no', 'cls', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']
#dataFrame.columns = ['no', 'cls', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6'] #FOR SAMPLE SET 3
Set1 = dataFrame.drop('no', axis=1)
trainData = Set1.values
#print(trainData)

def countNc(data, classNo):
    NcCount = np.zeros(classNo)
    for i in data:
        NcCount[i[0]] = NcCount[i[0]] + 1
    return NcCount

def get_n_values_of_features(Njc, row, feature_values):
    for f in range(0, len(row)):
        if f == 0:
            continue

        for d in range(feature_values[f - 1]):
            if (row[f] == d):
                Njc[row[0]][f - 1][d] += 1
    #print(Njc)
    return Njc


def get_pie_theta_for_prediction(data, type_of_class, feature_column_amount, feature_values):
    N = len(data)
    alphazero = type_of_class
    Nc = np.zeros(type_of_class)
    Njc = np.zeros((type_of_class, feature_column_amount, max(feature_values)))

    Nc = countNc(data, type_of_class)

    for row in data:
        Njc = get_n_values_of_features(Njc, row, feature_values)

    #New edited.
    alpha = np.array([.7 ,.1 ,.1 ,.1])
    Nc1 = Nc +  alpha - np.ones(4)
    N1 = N + np.sum(alpha) - type_of_class
    Piec = np.divide(Nc1, N1)
    #print(Piec)
    #exit()

    #Piec = [(n+1) / (N+2) for n in Nc] # PIE_C FOR MAP CALCULATION
    print(Piec)
    ##exit()
    Thetakjc = [[[(d+1) / (N+alphazero) for d in F] for F in n] for n in Njc] # THETA_JC FOR MAP CALCULATION
    #print(Thetakjc)

    #Piec = [n / N for n in Nc]  # PIE_C FOR MLE CALCULATION
    #Thetakjc = [[[d / N for d in F] for F in n] for n in Njc] # THETA_JC FOR MLE CALCULATION

    tpic = Piec
    thetak = Thetakjc

    Predicted_lic = np.zeros(len(data))

    for index,row in enumerate(data):
        Lic = np.zeros(type_of_class)
        for c in range(type_of_class):
            Lic[c] = math.log10(Piec[c])
            for j in range(1, len(row)):
                theTajc = Thetakjc[c][j-1][row[j]]
                Lic[c] += np.log10(theTajc) # FOR MAP CALCULATION
                #if theTajc != 0: # FOR MLE CALCULATION
                    #Lic[c] += math.log10(theTajc)# FOR MLE CALCULATION
            #Pic = np.exp(Lic - logsumexp(Lic))
        #print(np.argmax(Lic),'xx')
        Pic = np.exp(Lic - logsumexp(Lic))
        Predicted_lic[index] = np.argmax(Pic)
        #print(Predicted_lic)

        yi = np.argmax(Pic)
        #print(yi)
    #print(get_accuracy_percentage(data,Predicted_lic))
    #test = getAccuracy(data,Predicted_lic)
    #print(test)
    return tpic, thetak

def get_accuracy_percentage(data,predicted_classes):
    accuracy = [1 if d[0] == predicted_classes[index] else 0 for index, d in enumerate(data)]
    return (sum(accuracy) / len(data))*100

def get_prediction(data,Piec,Thetac,type_of_class):
    Predicted_lic = np.zeros(len(data))


    for index,row in enumerate(data):
        Lic = np.zeros(type_of_class)
        for c in range(type_of_class):
            Lic[c] = math.log10(Piec[c])
            for j in range(1, len(row)):
                #theTajc = Njc[c][j-1][row[j]]
                Lic[c] += np.log10(Thetac[c][j-1][row[j]]) # FOR MAP CALCULATION
                #if theTajc != 0: # FOR MLE CALCULATION
                    #Lic[c] += math.log10(theTajc)# FOR MLE CALCULATION
            Pic = np.exp(Lic - logsumexp(Lic))
        #print(np.argmax(Lic),'xx')
        Predicted_lic[index] = np.argmax(Pic)
        #print(Predicted_lic)

        yi = np.argmax(Pic)
        #print(yi)
    return Predicted_lic


def cross_validation(data, type_of_class, Piec, Thetac):
    accuracy = np.zeros(len(data))

    for index, row in enumerate(data.values):
        #print(row)
        training_dataset = data.drop(row)
        test_dataset = [row]
        #if t == 'map':
         #   Piec, Thetac = get_pie_theta_map(training_dataset.values, type_of_class, number_of_features, feature_values)
        #else:
         #   Piec, Thetac = get_pie_theta_mle(training_dataset.values, type_of_class, number_of_features, feature_values)
        Predicted_class = get_prediction(test_dataset, Piec, Thetac, type_of_class)
        #print("hjh")
        #print(Predicted_class)

        if Predicted_class[0] == row[0]:
            accuracy[index] = 1
    return sum(accuracy)*100 / len(data)

def main():
    Piec, Thetajc = get_pie_theta_for_prediction(trainData, 4, 7, [3,2,3,4,2,3,5])

    result = cross_validation(Set1, 4,Piec, Thetajc)
    print(result)
    z = 100-result
    print(z,"error rate")

main()
