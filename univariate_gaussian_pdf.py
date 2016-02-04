'''
This script will generate and plot densities for a one dimensional 
Gaussian curve.
'''


import numpy as np
import matplotlib.pyplot as plt

def uni_gauss_pdf(x,mu,sigma):
    '''
    given some scalar, x, return it's probability density given a gaussian
    with a mean/mode of mu and variance of sigma
    '''
    y = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-((x-mu)**2)/(2*(sigma**2)))
    return y

if __name__ == "__main__":
    # transform the function which takes in scalar into a function which 
    # takes a vector but performs element-wise operations
    uni_gauss_pdf = np.vectorize(uni_gauss_pdf)
    mu=50
    sigma=10
    x = np.arange(100)
    y = uni_gauss_pdf(x,mu,sigma)
    plt.plot(x,y)
    plt.show()
