import argparse
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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
    ax.legend()
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

def accept_point(oldPoint,proposal,mean,cov):
    '''
    given a new, proposed point and an old, previously accepted point,
    judge new point relative to its probability, and make a choice:
    return old or new point
    p(r|p_i, p_f, t, Σ) * p(p_i, p_f |μ_l, Σ_l)
    '''

    OldLogLikelihood = np.log(multivariate_normal.pdf(oldPoint,
                                                      mean=mean,
                                                      cov=cov))
    
    ProposedLogLikelihood = np.log(multivariate_normal.pdf(proposal,
                                                           mean=mean,
                                                           cov=cov))
    
    # if proposed point is more probable than old point, accept proposal,
    # else accept the proposed point randomly, relative to its probability
    acceptProb = ProposedLogLikelihood-OldLogLikelihood
    if np.log(np.random.random()) < acceptProb:
        # when proposal is more likely than old point, this is always true
        accepted = proposal
    else:
        accepted = oldPoint
    return accepted

def convert_3D_to_2D(point3D):
    # 3,1 column vector for a single point in 3D space
    p = np.ndarray(shape=(3,1),buffer=np.array(point3D),dtype=int)
    # need to append '1' to p for matrix multiplication
    _p = np.concatenate((p,[[1]]))
    # 3,4 camera matrix
    M = np.ndarray(shape=(3,4),
                   buffer=np.array([1,0,0,0,
                                    0,1,0,0,
                                    0,0,1,0]),
                   dtype=int)
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

    
def get_prob_point(point,Sigma,mu_L,Sigma_L,rsTs):
    # split up 6D point into two 3D points, initial and final
    p_i = point[:3]
    p_f = point[3:]
    # convert each point to 2D
    q_i = convert_3D_to_2D(p_i)
    q_f = convert_3D_to_2D(p_f)
    
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

if __name__ == "__main__":
    args = parse_user_args()
    coordsFile = args.infile1
    inputsFile = args.infile2

    Sigma = ((0.05)**2)*np.identity(2)
    Sigma_L = 10*np.identity(3)
    mu_L = (0,0,4)

    rs = np.loadtxt(coordsFile,delimiter=',').reshape(20,2)
    ts = np.loadtxt(inputsFile).reshape(20,1)
    rsTs = np.concatenate((rs,ts),axis=1)

    seed = np.random.uniform(-50,50,6)
    oldPoint = propose_new_point(seed)
    iPoints=[]
    fPoints=[]
    for i in range(10000):
        proposal = propose_new_point(oldPoint)
        probOldPoint = get_prob_point(oldPoint,Sigma,mu_L,Sigma_L,rsTs)
        probProposal = get_prob_point(proposal,Sigma,mu_L,Sigma_L,rsTs)

        acceptProb = probProposal/probOldPoint
        print(probProposal-probOldPoint)
        if (np.random.random()) < acceptProb:
            # when proposal is more likely than old point, this is always true
            accepted = proposal
        else:
            accepted = oldPoint
        oldPoint=accepted
        print(oldPoint)

        iPoints.append(oldPoint[:3])
        fPoints.append(oldPoint[3:])

    plot_3D_points(iPoints)
    plot_3D_points(fPoints)
