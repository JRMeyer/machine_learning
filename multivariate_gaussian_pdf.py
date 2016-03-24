'''
This script will give densities for D-dimensional points from a Gaussian
distribution. The code should work where D>2, but I've got it set up for 
D==2 because I wanted to plot the curve in 3D space
'''



import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def multi_gauss_pdf(x,mu,Sigma):
    '''
    x is a vector of length D
    mu is a vector of length D
    Sigma is a matrix where both dimensions are length D
    '''
    D = len(x)
    y = (1/np.sqrt((2*np.pi)**D * np.linalg.det(Sigma)) * 
         np.exp(-1/2 * np.dot((x-mu).T, np.dot( np.linalg.inv(Sigma), (x-mu)))))
    return y


if __name__ == "__main__":
    # transform the function which takes in scalar into a function which 
    # takes a vector but performs element-wise operations
    mu = np.ndarray(shape=(2,), buffer=np.array([.5,.5]),dtype=float)
    Sigma = np.ndarray(shape=(2,2), buffer=np.array([[.05,0.],
                                                     [0.,.05]]),dtype=float)
    numPoints = 100
    z = []
    x = np.random.rand(numPoints)
    y = np.random.rand(numPoints)

    numPoints = len(x)
    for i in range(numPoints):
        point = np.asarray([x[i],y[i]])
        z_i = multi_gauss_pdf(point,mu,Sigma)
        z.append(z_i)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    plt.show()
