'''
This script will give densities for D-dimensional points from a Gaussian
distribution. The code should work where D>2, but I've got it set up for 
D==2 because I wanted to plot the curve in 3D space
'''



import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def eval_data_on_pdf(x,mu,Sigma):
    '''
    x is a vector of length D
    mu is a vector of length D
    Sigma is a matrix where both dimensions are length D

    Given data and sufficient statistics for a Gaussian pdf, calculate the
    probability of that Gaussian generating that data
    '''
    D = len(x)
    y = (1/np.sqrt((2*np.pi)**D * np.linalg.det(Sigma)) * 
         np.exp(-1/2 * np.dot((x-mu).T, np.dot( np.linalg.inv(Sigma), (x-mu)))))
    return y


def generate_data(num_samples):
    '''
    assuming 2 dimensional features and just two generating functions

    ie make two blobs on an x,y plane

    '''
    from sklearn.datasets.samples_generator import make_blobs
    
    cluster_std = ([.25, 1.5])
    features, labels = make_blobs(n_samples=num_samples,
                                  centers=2,
                                  n_features=2,
                                  cluster_std=cluster_std,
                                  center_box=(-2, 2),
                                  random_state=None)
    x,y = features[:,0], features[:,1]
    
    return(x,y,labels)


def mle_est_mu(x,y,labels):
    '''
    x = D dimensional vector, number of data points
    y = D dimensional vector
    labels = D dimensional vector

    Estimate mu, aka find the empirical mean of the labeled data

    assuming 2 dimensional features right now (hence x,y)
    assuming label == 1 when the feature belongs to the Gaussian, 0 elsewhere

    $  \hat{\mu} = \frac{\sum\limits_{n=1}^{N} \gamma_{n} \cdot x_{n}}
                   {\sum\limits_{n=1}^{N} \gamma_{n}}$
    in the above equation, gamma is the label (ie. 0 or 1)

    MLE calculation
    '''
    
    mu_x = np.dot(labels,x)/np.sum(labels)
    mu_y = np.dot(labels,y)/np.sum(labels)

    mu = np.ndarray(shape=(2,), buffer=np.array([mu_x,mu_y]),dtype=float)
    
    return(mu)


def mle_est_Sigma(x,y,mu,labels):
    '''
    x = D dimensional vector, number of data points
    y = D dimensional vector
    labels = D dimensional vector
    mu = N dimensional vector, number of dimensions of features (2d here)

    returns Sigma, which is a diagonal covariance matrix

    $\sigma_{t}(i) = \sqrt{\frac{\sum\limits_{n=1}^{N}\gamma_{n} \Big(x_{n} - 
                     \mu_{n}\Big)^{2}}{\sum\limits_{n=1}^{N}\gamma_{n}}} $

    MLE calculation

    '''
    
    mu_x = mu[0]
    mu_y = mu[1]
    
    sigma_x = np.sqrt(np.sum(np.multiply(labels,np.square(x - mu_x)))/np.sum(labels))
    sigma_y = np.sqrt(np.sum(np.multiply(labels,np.square(y - mu_y)))/np.sum(labels))

    sigma = np.ndarray(shape=(2,), buffer=np.array([sigma_x,sigma_y]),dtype=float)
    Sigma = np.diag(sigma)
    
    return(Sigma)




def map_est_mu(x_new,y_new, mu_old, labels_new, tau):
    '''
    x_new = D dimensional vector, number of data points
    y_new = D dimensional vector
    labels_new = D dimensional vector
    mu_old = vector of mu_x,mu_y for original data
    tau = scaling factor for mu_new:mu_old

    Estimate mu_new, MAP estimate

    assuming 2 dimensional features right now (hence x,y)
    assuming label == 1 when the feature belongs to the Gaussian, 0 elsewhere

    $  \hat{\mu} = \frac{\tau \cdot \mu_{o} + \sum\limits_{n=1}^{N} \gamma_{n} 
    \cdot x_{n}}{\tau + \sum\limits_{n=1}^{N} \gamma_{n}}$

    in the above equation, gamma is the label (ie. 0 or 1)

    MAP calculation
    '''

    mu_x_old = mu_old[0]
    mu_y_old = mu_old[1]
    
    mu_x_new = tau*mu_x_old + np.dot(labels_new,x_new)/(tau + np.sum(labels_new))
    mu_y_new = tau*mu_y_old + np.dot(labels_new,y_new)/(tau + np.sum(labels_new))

    mu_new = np.ndarray(shape=(2,), buffer=np.array([mu_x_new,mu_y_new]),dtype=float)
    
    return(mu_new)


def map_est_Sigma(x,y,mu,labels):
    '''
    x = D dimensional vector, number of data points
    y = D dimensional vector
    labels = D dimensional vector
    mu = N dimensional vector, number of dimensions of features (2d here)

    returns Sigma, which is a diagonal covariance matrix

    $\Sigma_{t}(i) = \sqrt{\frac{\sum\limits_{n=1}^{N}\gamma_{n} \Big(x_{n} - 
                     \mu_{n}\Big)^{2}}{\sum\limits_{n=1}^{N}\gamma_{n}}} $

    MLE calculation

    '''
    
    mu_x = mu[0]
    mu_y = mu[1]
    
    sigma_x = np.sqrt(np.sum(np.multiply(labels,np.square(x - mu_x)))/np.sum(labels))
    sigma_y = np.sqrt(np.sum(np.multiply(labels,np.square(y - mu_y)))/np.sum(labels))

    sigma = np.ndarray(shape=(2,), buffer=np.array([sigma_x,sigma_y]),dtype=float)
    Sigma = np.diag(sigma)
    
    return(Sigma)




def plot_pdf(mu,Sigma,numPoints):
    '''

    Given sufficient stats for a Gaussian pdf, generate a plot of points 
    and their likelihoods given that pdf... if you plot enough points, you
    get a good picture of the shape of the pdf

    '''
    # sample dummy mu and Sigma
    # mu = np.ndarray(shape=(2,), buffer=np.array([.5,.5]),dtype=float)
    # Sigma = np.ndarray(shape=(2,2), buffer=np.array([[.05,0.],
    #                                                  [0.,.05]]),dtype=float)

    # generate points for evaluation
    z = []
    x = np.random.uniform(-3,3,numPoints)
    y = np.random.uniform(-3,3,numPoints)

    numPoints = len(x)
    for i in range(numPoints):
        point = np.asarray([x[i],y[i]])
        z_i = eval_data_on_pdf(point,mu,Sigma)
        z.append(z_i)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    plt.show()

    
if __name__ == "__main__":

    # make data and labels
    x,y,labels_1 = generate_data(num_samples=100)
    labels_2 = (labels_1 - 1)*(-1)

    # plot training data
    plt.scatter(x,y,c=labels_1)
    plt.show()
    
    # MLE for means
    mu_1 = mle_est_mu(x,y,labels_1)
#    mu_2 = mle_est_mu(x,y,labels_2)

    # MLE for sigmas
    Sigma_1 = mle_est_Sigma(x,y,mu_1,labels_1)
#    Sigma_2 = mle_est_Sigma(x,y,mu_2,labels_2)

    # plot points on pdf
    plot_pdf(mu_1,Sigma_1,1000)
#    plot_pdf(mu_2,Sigma_2,1000)


    # ADAPTATION
    # make data and labels
    x_new,y_new,labels_1_new = generate_data(num_samples=10)
    labels_2_new = (labels_1_new - 1)*(-1)

    # plot training data
    plt.scatter(x_new,y_new,c=labels_1_new)
    plt.show()
    
    # MAP for means with data from OTHER blob
    mu_1_map = map_est_mu(x_new,y_new, mu_1, labels_2_new, tau=1.5)


    plot_pdf(mu_1_map,Sigma_1,1000)
