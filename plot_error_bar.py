import matplotlib.pyplot as plt
import numpy as np
import sys

_tensor = None

# ret data as vector
def read_file(filename):
    global _tensor
    nums = None

    with open(filename) as f:
        nums = [float(x.strip()) for x in f.readlines()]

    _tensor = np.array(nums)
    
         
if __name__=="__main__":

    if len(sys.argv) < 2:
        print("need error file for plotting")
        sys.exit(-1)

    read_file(sys.argv[1])
    print(_tensor)

    # yerr = 0.1 + 0.2*np.sqrt(_tensor)
    # xerr = 0.1 + yerr

    print(yerr)

    plt.errorbar([x for x in xrange(1,len(_tensor) + 1)], _tensor, 'ro', xerr=xerr, yerr=yerr)
    
    plt.show()

