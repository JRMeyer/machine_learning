from scipy.stats import multivariate_normal
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_3D_points(points):
    '''
    where 'points' is a list of (x,y,z) tuples,
    plot them in 3D space
    '''
    x,y,z = zip(*points)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x,y,z)
    ax.legend()
    plt.show()
    
def propose_new_point(oldPoint):
    '''
    Using a Gaussian distribution centered on oldPoint,
    generate a new, random point from the distribution.
    This new point will be the next proposed random step.
    '''
    mean = oldPoint
    cov = [[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]
    proposal = np.random.multivariate_normal(mean, cov)
    return proposal

def accept_point(oldPoint,proposal):
    '''
    p(r|p_i,p_f,t,Σ) * p(p_i,p_f|μ_l,Σ_l)
    '''
    cov = [[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]

    OldLogLikelihood = np.log(multivariate_normal.pdf(oldPoint,
                                                      mean=(0,0,0),
                                                      cov=cov))
    
    ProposedLogLikelihood = np.log(multivariate_normal.pdf(proposal,
                                                           mean=(0,0,0),
                                                           cov=cov))
    
    # if the new point is more probable, automatically accept it
    if OldLogLikelihood-ProposedLogLikelihood < 0:
        accepted = proposal
    else:
        # if proposal is less probable than oldPoint, accept the proposal
        # relative to it's probability
        if np.log(np.random.random()) < ProposedLogLikelihood:
            accepted = proposal
        else:
            accepted = oldPoint
    return accepted


points=[]
oldPoint = (20,20,20)
for i in range(1000):
    proposal = propose_new_point(oldPoint)
    oldPoint = accept_point(oldPoint,proposal)
    points.append(oldPoint)
    

plot_3D_points(points)
