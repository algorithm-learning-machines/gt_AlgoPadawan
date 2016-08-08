import matplotlib.pyplot as plt
import numpy as np
import sys

_tensorTraining = None
_tensorAccuracy = None
trainingSuffix = "_train"

# ret data as vector
# may want to extend over more epochs
# could add exp_count?

def read_file(modelName, itRange):
    global _tensor
    nums = None

    for i in xrange(itRange[0], itRange[1] + 1):
        with open(modelName + trainingSuffix + str(i)) as f:
            nums += [float(x.strip()) for x in f.readlines()]
    
    _tensorTraining = np.array(nums)
    
         
if __name__=="__main__":

    if len(sys.argv) < 2:
        print("need error file for plotting")
        sys.exit(-1)
    

    modelName = 

    read_file(sys.argv[1])

    plt.errorbar([x for x in xrange(1,len(_tensor) + 1)], _tensor, 'ro', xerr=xerr, yerr=yerr)
    
    plt.show()

