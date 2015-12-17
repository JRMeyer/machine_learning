import numpy as np

# 3,1 column vector for a single point in 3D space
p = np.ndarray(shape=(3,1),buffer=np.array([1,2,3]),dtype=int)
# need to append '1' to p for matrix multiplication
_p = np.concatenate((p,[[1]]))

# 3,4 camera matrix
M = np.ndarray(shape=(3,4),buffer=np.array([1,0,0,0,
                                            0,1,0,0,
                                            0,0,1,0]),dtype=int)

# 3,1 column vector of tilde values to make 2D point with
uvw = np.dot(M,_p)

# extract the scalar w and the column vector uv from uvw
uv = uvw[:2,]
w = uvw[2]

# get the 2D points!
q = (1/w)*uv
print(q) 
