import argparse
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_1D_points(points):
    '''
    where 'points' is a list of [y_1...y_n] points,
    plot them in 2D space, using their index in the list as
    the x value, and the point, y_n, as the y value
    '''
    fig = plt.figure()
    x,y = zip(*[(x,y) for x,y in enumerate(points)])
    ax = fig.gca()
    ax.plot(x,y)
    ax.legend()
    plt.show()
  
def plot_2D_points(points):
    '''
    where 'points' is a list of (x,y) tuples,
    plot them in 2D space
    '''
    x,y = zip(*points)
    fig = plt.figure()
    # gca = Get the current axes, creating one if necessary
    ax = fig.gca()
    ax.plot(x,y)
    ax.legend()
    plt.show()
    
def plot_3D_points(points):
    '''
    where 'points' is a list of (x,y,z) tuples,
    plot them in 3D space
    '''
    x,y,z = zip(*points)
    fig = plt.figure()
    # gca = Get the current axes, creating one if necessary
    ax = fig.gca(projection='3d')
    ax.plot(x,y,z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
def propose_new_point(oldPoint):
    '''
    Using a Gaussian distribution centered on oldPoint,
    generate a new, random point from the distribution.
    This new point will be the next proposed random step.
    '''
    mean = oldPoint
    cov = np.identity(6)
    proposal = np.random.multivariate_normal(mean, cov)
    return proposal


def convert_3D_to_2D(point3D,M):
    '''
    where M is our camera matrix, and point3D is a (x,y,z) tuple
    '''
    # 3,1 column vector for a single point in 3D space
    p = np.ndarray(shape=(3,1),buffer=np.array(point3D),dtype=int)
    # need to append '1' to p for matrix multiplication
    _p = np.concatenate((p,[[1]]))

    # 3,1 column vector of tilde values to make 2D point with
    uvw = np.dot(M,_p)
    # extract the scalar w and the column vector uv from uvw
    uv = uvw[:2,]
    w = uvw[2]
    # get the 2D points!
    q = (1/w)*uv
    return q


def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1','--infile1', type=str, help='1 input text file')
    parser.add_argument('-i2','--infile2', type=str, help='2 input text file')
    args = parser.parse_args()
    return args

    
def get_prob_point(point,Sigma,mu_L,Sigma_L,rsTs,M):
    '''
    given a 6D point, covariance matrices, mean, input points
    and camera matrix, find out how probable the two 3D points of 
    6D point are given the input data
    '''
    # split up 6D point into two 3D points, initial and final
    p_i = point[:3]
    p_f = point[3:]
    # convert each point to 2D
    q_i = convert_3D_to_2D(p_i,M)
    q_f = convert_3D_to_2D(p_f,M)
    # get prior prob for each point, then multiply them for joint prob
    prior_i = (multivariate_normal.pdf(p_i,mean=mu_L,cov=Sigma_L))
    prior_f = (multivariate_normal.pdf(p_f,mean=mu_L,cov=Sigma_L))
    joint_prior = prior_i*prior_f
    # evaluate p_i and p_f on all given, 2D rendered points
    probs=[]
    for dataPoint in rsTs:
        r = dataPoint[:2]
        t = dataPoint[2]
        q_s = q_i + (q_f-q_i)*t
        q_s = q_s.flatten()
        prob_r = (multivariate_normal.pdf(r,mean=q_s,cov=Sigma))
        probs.append(prob_r*joint_prior)
    sumProbs = np.sum(probs)
    return sumProbs

    
def metropolis_hastings(seedPoint,Sigma,mu_L,Sigma_L,rsTs,numIterations,M):
    # since we are searching in 6D space, for convenience return two lists
    # each containing 3D points for the initial and final point of 3D line
    iPoints=[]
    fPoints=[]
    oldPoint = seedPoint
    for i in range(numIterations):
        proposal = propose_new_point(oldPoint)
        probOldPoint = get_prob_point(oldPoint,Sigma,mu_L,Sigma_L,rsTs,M)
        probProposal = get_prob_point(proposal,Sigma,mu_L,Sigma_L,rsTs,M)
        acceptProb = np.log(probProposal)- np.log(probOldPoint)
        # when proposal is more likely than oldPoint, the ratio of proposal to
        # olPoint will be greater than 1 (we log values, but it doesn't matter)
        # Becuase np.random.random() generates numbers from 0 to 1, if
        # proposal is more likely than oldPoint, this is always true
        if np.log(np.random.random()) < acceptProb:
            accepted = proposal
        else:
            accepted = oldPoint
        # use the accepted point as center of Gaussian for next step
        oldPoint = accepted
        iPoints.append(oldPoint[:3])
        fPoints.append(oldPoint[3:])
        # print is nice to see the convergence of values to a distribution
        print(oldPoint)
    return iPoints,fPoints

def MAP_estimation(seedPoint,Sigma,mu_L,Sigma_L,rsTs,numIterations,M):
    iPoints=[]
    fPoints=[]
    oldPoint = seedPoint
    for i in range(numIterations):
        proposal = propose_new_point(oldPoint)
        probOldPoint = get_prob_point(oldPoint,Sigma,mu_L,Sigma_L,rsTs,M)
        probProposal = get_prob_point(proposal,Sigma,mu_L,Sigma_L,rsTs,M)
        acceptProb = np.log(probProposal)- np.log(probOldPoint)
        if acceptProb > 0:
            accepted = proposal
        else:
            accepted = oldPoint
        oldPoint = accepted
        iPoints.append(oldPoint[:3])
        fPoints.append(oldPoint[3:])
    return iPoints,fPoints

def demo(coordsFile,inputsFile):
    '''
    a demo of metropolis hastings with some visualizations
    '''
    Sigma = ((0.05)**2)*np.identity(2)
    Sigma_L = 10*np.identity(3)
    # mean prior distribution on line
    mu_L = (0,0,4)
    # 3,4 camera matrix
    M = np.ndarray(shape=(3,4),
                   buffer=np.array([1,0,0,0,
                                    0,1,0,0,
                                    0,0,1,0]),
                   dtype=int)
    
    rs = np.loadtxt(coordsFile,delimiter=',').reshape(20,2)
    ts = np.loadtxt(inputsFile).reshape(20,1)
    rsTs = np.concatenate((rs,ts),axis=1)

    seed = np.random.uniform(-20,20,6)
    numIterations=10000
    
    iPoints, fPoints = metropolis_hastings(seed,Sigma,mu_L,Sigma_L,rsTs,
                                           numIterations,M)

    # plot the searches for each 3D point
    plot_3D_points(iPoints)
    plot_3D_points(fPoints)

    # plot the search for each weight (dimension)
    x1,y1,z1 = zip(*iPoints)
    x2,y2,z2 = zip(*fPoints)
    weights = [x1,y1,z1,x2,y2,z2]
    for weight in weights:
        plot_1D_points(weight)

    
if __name__ == "__main__":
    args = parse_user_args()
    coordsFile = args.infile1
    inputsFile = args.infile2

    demo(coordsFile,inputsFile)



    

