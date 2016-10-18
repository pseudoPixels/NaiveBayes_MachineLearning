#@Author: Golam Mostaeen
#@NSID: gom766
#@Email: golammostaeen@gmail.com
#@Software Research Lab, University of Saskatchewan, Canada



#===========================================USEFUL IMPORTS================================
import numpy as np
import math
from scipy.misc import logsumexp










#==========================================LOADING THE DATA FILES========================
dataFile = open('DataFiles\Source_3_100_exemplars.txt', 'r')
dataSet = np.loadtxt(dataFile)




#=========================================get_total_number_of_samples()=================
#DESCRIPTION:
#   RETURNS THE TOTAL NUMBER OF SAMPLES/ROWS AVAILABLE IN THE DATA SOURCE
#PARAMETER:
#   dataSet(NDARRAY): THE DATASET FOR TRAINING THE CLASSIFIER
#RETURNS:
#   (INT) - THE TOTAL NUMBER OF SAMPLES IN THE DATASET
#=======================================================================================
def get_total_number_of_samples(dataSet):
    total_number_of_samples = len(dataSet)
    return total_number_of_samples







#========================================get_class_labels()=============================
#DESCRIPTION:
#   FINDS OUT THE POSSIBLE CLASS LABELS AVAILABLE IN THE DATA SOURCE. THE FIRST
#   COLUMN OF THE GIVEN DATASET CONTAINS THE CLASS INFORMATION. THIS FUNCTION FINDS
#   OUT THE POSSIBLE CLASS LABELS LIKE 0,1,2,3 ETC.
#PARAMETERS:
#   dataSet(NDARRAY): THE DATASET FOR TRAINING THE CLASSIFIER
#RETURNS:
#   (ARRAY) - THE POSSIBLE CLASS LABELS IN THE DATA SETS
#=======================================================================================
def get_class_labels(dataSet):
    classLabels = np.unique(dataSet[:, 0].astype(int))
    return classLabels






#=====================================get_number_of_classes()============================
#DESCRIPTION:
#   RETURNS THE NUMBER OF CLASSES AVAILABLE IN THE DATA SET. FOR EXAMPLE 4, 5 ETC
#PARAMETERS:
#   dataSet(NDARRAY): THE DATASET FOR TRAINING THE CLASSIFIER
#RETURNS:
#   (INT) - THE NUMBER OF CLASSES IN THE DATA SET
#=======================================================================================
def get_number_of_classes(dataSet):
    number_of_classes = get_class_labels(dataSet).shape[0]
    return number_of_classes







#===================================get_class_bins()====================================
#DESCRIPTION:
#   RETURNS THE COUNTS OF DIFFERENT CLASSES. FOR EXAMPLE THE NUMBER OF SAMPLES OF
#   CLASS 0,1 AND 2 ARE 80,65 AND 72 RESPECTIVELY
#PARAMETERS:
#   dataSet(NDARRAY): THE DATASET FOR TRAINING THE CLASSIFIER
#RETURNS:
#   (ARRAY) - THE CORRESPONDING CLASS BINS LIKE [80, 65, 72]
#======================================================================================
def get_class_bins(dataSet):
    classBins = np.bincount(dataSet[:, 0].astype(int))
    return classBins











#=================================get_pi_for_mle()=====================================
#DESCRIPTION:
#   RETURNS THE CLASS PRIORS FOR MLE. NUMBER OF SAMPLES FOR A GIVEN CLASSS DIVIDED
#   THE TOTAL NUMBER OF SAMPLES. N_C/N
#PARAMETERS:
#   dataSet(NDARRAY): THE DATASET FOR TRAINING THE CLASSIFIER
#RETURNS:
#   (ARRAY) - CLASS PRIORS FOR ALL THE CLASSES LIKE [.6,.1,.3]
#=====================================================================================
def get_pi_for_mle(dataSet):
    total_number_of_samples = get_total_number_of_samples(dataSet)
    classBins = get_class_bins(dataSet)
    pi_c = np.divide(classBins, total_number_of_samples)
    return pi_c







#=======================================get_theta_for_mle()========================================
#DESCRIPTION:
#   RETURNS THE LIKELIHOOD, ATTRIBUTE WISE FOR ALL CLASSES AND ITS CORRESPONDING ATTRIBUTE
#   CATEGORIES. FOR EXAMPLE, FOR FIRST ATTRIBUTE THE P(1|4) IS STORED IN allTheta[0][4][1].
#   FOR THE SECOND ATTRIBUTE P(2|3) IS STORED IN allTheta[1][3][2]
#PARAMETERS:
#   dataSet(NDARRAY): THE DATASET FOR TRAINING THE CLASSIFIER
#RETURNS:
#   (DICTIONARY) - KEYED WITH ATTRIBUTE INDEX LIKE 0,1,2
#=================================================================================================
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










#================================get_pi_for_map()=====================================
#DESCRIPTION:
#   RETURNS THE CLASS PRIORS FOR MAP. (N_K + ALPHA_K - 1)/(N+ALPHA_NOT - K)
#PARAMETERS:
#   dataSet(NDARRAY): THE DATASET FOR TRAINING THE CLASSIFIER
#   alpha (NDARRAY): THE IMPOSED ARRAY FOR PRIORS: ALPHA_K
#RETURNS:
#   (ARRAY) - CLASS PRIORS FOR ALL THE CLASSES LIKE [.6,.1,.3]
#======================================================================================
def get_pi_for_map(dataSet, alpha):
    classBins = get_class_bins(dataSet)
    total_number_of_samples = get_total_number_of_samples(dataSet)

    classBins_with_aplpha_addition = classBins + alpha - np.ones(classBins.shape[0]) #N_k + ALPHA_k - 1
    total_number_of_samples_with_all_aplpha_addition = total_number_of_samples + np.sum(alpha) - classBins.shape[0] #N + ALPHA_sum - K
    pi_c = np.divide(classBins_with_aplpha_addition, total_number_of_samples_with_all_aplpha_addition)

    return  pi_c







#=======================================get_theta_for_map()========================================
#DESCRIPTION:
#   RETURNS THE LIKELIHOOD FOR MAP, ATTRIBUTE WISE FOR ALL CLASSES AND ITS CORRESPONDING ATTRIBUTE
#   CATEGORIES. FOR EXAMPLE, FOR FIRST ATTRIBUTE THE P(1|4) IS STORED IN allTheta[0][4][1].
#   FOR THE SECOND ATTRIBUTE P(2|3) IS STORED IN allTheta[1][3][2]
#PARAMETERS:
#   dataSet(NDARRAY): THE DATASET FOR TRAINING THE CLASSIFIER
#RETURNS:
#   (DICTIONARY) - KEYED WITH ATTRIBUTE INDEX LIKE 0,1,2
#=================================================================================================
def get_theta_for_map(dataSet):
    allTheta ={}
    number_of_classes = get_number_of_classes(dataSet)
    categories_per_attributes = get_categories_per_attributes(dataSet)
    for i in range(dataSet.shape[1]-1):
        theta = np.zeros([int(number_of_classes),int(categories_per_attributes[i])])
        for row in range(len(dataSet)):
            theta[dataSet[row][0].astype(int)][dataSet[row][i+1].astype(int)] +=1
        theta = theta / get_class_bins(dataSet).reshape(get_class_bins(dataSet).shape[0],1)
        allTheta[i] = theta
    return allTheta







#======================================get_categories_per_attributes()===============================
#DESCRIPTION:
#   RETURNS THE DIFFERENT CATEGORIES AVAILABLE FOR ATTRIBUTES. FOR EXAMPLE THE FIRST ATTRIBUTE
#   CONTAINS 3 POSSIBLE VALUES(0,1,2), SECOND ATTRIBUTE CONTAINS 2 POSSIBLE VALUES(0,1) AND SO ON.
#   COUNTS THIS NUMBER FOR ALL ATTRIBUTES AND RETURNS IN ARRAY LIKE [3,2] FOR ABOVE EXAMPLE.
#PARAMETERS:
#   dataSet(NDARRAY): THE DATASET FOR TRAINING THE CLASSIFIER
#RETURNS:
#   (ARRAY) - POSSIBLE CATERGORIES FOR EACH ATTRIBUTES
#====================================================================================================
def get_categories_per_attributes(dataSet):
    categories_per_attributes = np.zeros(dataSet.shape[1]-1)
    for attribute_type in range(dataSet.shape[1]-1):
        categories_per_attributes[attribute_type] = np.unique(dataSet[:, attribute_type+1].astype(int)).shape[0]
    return categories_per_attributes








#======================================nbcPredict_with_logsumexp_trick()=================
#DESCRIPTION:
#   RETURNS THE CLASS PREDICTION WITH NBC LOG-SUM-EXP TRICK.
#PARAMETERS:
#   allTheta(DICTIONARY): ALL LIKELIHOODS OR THETA VALS KEYED AS ATTRIBUTE INDEX.
#   allPi(ARRAY): THE CLASS PRIORS
#RETURNS:
#   (INT) - PREDICITED CLASS LABEL
#========================================================================================
def nbcPredict_with_logsumexp_trick(allTheta, allPi, testSet):
    Li = np.zeros(allPi.shape[0]);
    for aClass in range(allPi.shape[0]):
        Li[aClass] = math.log10(allPi[aClass])
        for anAttribute in range(len(allTheta.keys())):
            if(allTheta[anAttribute][aClass][int(testSet[anAttribute])] != 0):
                Li[aClass] = Li[aClass] + math.log10(allTheta[anAttribute][aClass][int(testSet[anAttribute])])
    pi = np.exp(Li - logsumexp(Li))
    return np.argmax(pi)








#======================================nbcPredict()======================================
#DESCRIPTION:
#   RETURNS THE CLASS PREDICTION WITH NBC
#PARAMETERS:
#   allTheta(DICTIONARY): ALL LIKELIHOODS OR THETA VALS KEYED AS ATTRIBUTE INDEX.
#   allPi(ARRAY): THE CLASS PRIORS
#RETURNS:
#   (INT) - PREDICITED CLASS LABEL
#========================================================================================
def nbcPredict(allTheta, allPi, testSet):
    classProbabilities = np.ones(allPi.shape[0])

    for aClass in range(allPi.shape[0]):
        thetaProduct = 1
        for i in range(testSet.shape[0]):
            thetaProduct *= allTheta[i][aClass][int(testSet[i])]

        classProbabilities[aClass] = allPi[aClass]*thetaProduct
    return np.argmax(classProbabilities)








#=====================================get_error_rate_via_cross_validation()==============
#DESCRIPTION:
#   RETURNS THE ERROR RATE USING THE PREDICTION FUNCTION ON SUPPLIED DATA SET.
#   CROSS VALIDATION IS USED FOR THIS CALCULATION.
#PARAMETERS:
#   dataSet(NDARRAY): THE DATASET FOR TRAINING THE CLASSIFIER
#RETURNS:
#   (FLOAT) - ERROR RATE
#=======================================================================================
def get_error_rate_via_cross_validation(dataSet):
    correct_prediction_counter = 0
    for row in range(len(dataSet)-1):
        new_dataset_with_a_row_exclusion = np.delete(dataSet, row, 0)
        #==============================================================================#
        #PLEASE CALL DIFFERENT FUNCTION FROM HERE FOR CHEKING THE OUTPUTS.             #
        #LIKE nbcPredict_with_logsumexp_trick CAN BE USED INSTEAD nbcPredict OR SO ON  #
        #==============================================================================#
        if(int(dataSet[row][0]) == int(nbcPredict(get_theta_for_mle(new_dataset_with_a_row_exclusion), get_pi_for_mle(new_dataset_with_a_row_exclusion), dataSet[row][1:]))):
            correct_prediction_counter += 1
    return (100 - correct_prediction_counter*100/len(dataSet))




#======================================PRINT ERROR RATE===============================
print('Error Rate : ', get_error_rate_via_cross_validation(dataSet), ' %')








#===================================================================================
#====================                                           ====================
#==================== FOLLOWING ARE SOME FUNCTION FOR ANALYZING ====================
#====================                                           ====================
#===================================================================================



#===================================get_total_possible_theta_values()==============
#DESCRIPTION:
#   RETURNS ALL POSSIBLE THETA OR LIKELIHOOD VALUES FOR ALL ATTRIBUTES AND CLASSES
#PARAMETERS:
#   dataSet(NDARRAY): THE DATASET FOR TRAINING THE CLASSIFIER
#RETURNS:
#   (INT) - NUMBERS OF ALL POSSIBLE THETA
#===================================================================================
def get_total_possible_theta_values(dataSet):
    allPossibleThetas = get_theta_for_mle(dataSet)
    allPossibleThetaCounts = 0
    for anAttribute in range(len(allPossibleThetas)):
        allPossibleThetaCounts += allPossibleThetas[anAttribute].shape[0] * allPossibleThetas[anAttribute].shape[1]

    return allPossibleThetaCounts




#=================================get_possible_black_swan()========================
#DESCRIPTION:
#   RETURNS THE ZERO COUNT LIKELIHOOD IN THE GIVEN DATASET
#PARAMETERS:
#   dataSet(NDARRAY): THE DATASET FOR TRAINING THE CLASSIFIER
#RETURNS:
#   (INT) - TOTAL ZERO COUNTS
#=================================================================================
def get_possible_black_swan(dataSet):
    allPossibleThetas = get_theta_for_mle(dataSet)
    possible_black_swan_count = 0;
    for anAttribute in range(len(allPossibleThetas)):
        for row in range(allPossibleThetas[anAttribute].shape[0]):
            for column in range(allPossibleThetas[anAttribute].shape[1]):
                if allPossibleThetas[anAttribute][row][column] == 0:
                    possible_black_swan_count += 1

    return possible_black_swan_count



#=======================GET POSSIBLE BLACK SWAN OR ZERO COUNT PERCENTAGE=========
print('Black Swan Percentage : ', get_possible_black_swan(dataSet)*100/get_total_possible_theta_values(dataSet), '%')






#============================correlation_matrix()================================
#DESCRIPTION:
#   CALCULATES AND PLOT THE CORRELATION MATRIX
#PARAMETERS:
#   dataSet(NDARRAY): THE DATASET FOR TRAINING THE CLASSIFIER
#RETURNS:
#   PLOTS THE CORRELATION MATRIX
#===============================================================================
def correlation_matrix(dataSet):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(np.corrcoef(dataSet), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Pearson correlation coefficients (Source_1_50)')
    cbar = fig.colorbar(cax)
    plt.show()













