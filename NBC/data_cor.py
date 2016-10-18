import numpy as np
import math
from scipy.misc import logsumexp


#loading the data source file
dataFile = open('DataFiles\Source_1_50_exemplars.txt', 'r')
dataSet = np.loadtxt(dataFile)
#dataSet = dataSet[:,1:5]
print(dataSet)
print(np.corrcoef(np.transpose(dataSet)))

def correlation_matrix(df):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(np.corrcoef(dataSet), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Pearson correlation coefficients (Source_1_50)')
    #labels=['Sex','Length','Diam','Height','Whole','Shucked','Viscera','Shell','Rings',]
    #ax1.set_xticklabels(labels,fontsize=6)
    #ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax)
    plt.show()

correlation_matrix(np.transpose(dataSet))

#print(np.corrcoef(a,b))
