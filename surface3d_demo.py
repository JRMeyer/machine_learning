from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np



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


fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)

mu = np.ndarray(shape=(2,), buffer=np.array([-3.,2.]),dtype=float)
Sigma = np.ndarray(shape=(2,2), buffer=np.array([[.5,0.],
                                                 [1.,1.5]]),dtype=float)

zs = np.array([multi_gauss_pdf([x_i,y_i],mu,Sigma) 
               for x_i,y_i 
               in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                       linewidth=0, antialiased=False)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_zlim(0, .2)

ax.view_init(elev=30, azim=45)

fig.set_size_inches(12,10)
fig.savefig('out.png', dpi=100, transparent=True)

plt.show()
