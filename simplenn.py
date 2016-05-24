import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

###
### FOUR EQUATIONS OF BACKPROP
###

def compute_error_final(a_final, z_final, gold_label):
    '''
    computing this as if we were using a quadratic cost function like MSE
    '''
    error_final = (a_final - gold_label) * sigmoid_prime(z_final)
    return error_final
    
def compute_error_l(weights_lplus1, error_lplus1, z_l):
    '''
    move error backward from previous layer through weights and
    then through the activation function
    '''
    error_l = np.dot(weights_lplus1, error_lplus1) * sigmoid_prime(z_l)
    return error_l

def compute_dC_db_l(error_l):
    '''
    The gradient of the cost function WRT the biases of a layer (l)
    '''
    return error_l

def compute_dC_dW_l(error_l, activation_lminus1):
    '''
    The gradient of the cost function WRT the weights of a layer (l)
    '''
    return np.dot(error_l, activation_lminus1.T)


###
### GRADIENT DESCENT
###

def update_weights_l(weights_l, activation_lminus1, error_l, learning_rate):
    dC_dW = compute_dC_dW_l(error_l, activation_lminus1)
    new_weights= weights_l - (learning_rate*dC_dW).T
    return new_weights
 
def update_biases_l(biases_l, error_l, learning_rate):
    dC_db = compute_dC_db_l(error_l)
    new_biases = biases_l - learning_rate*dC_db
    return new_biases


def initialize_weights_biases(numFeatures, numLabels):
    # Values are randomly sampled from a Gaussian with a standard deviation of:
    #     sqrt(6 / (numInputNodes + numOutputNodes + 1))
    W_1 = np.random.normal(size=(numFeatures,numLabels),
                           loc=0,
                           scale=(np.sqrt(6/numFeatures+numLabels+1)))
    b_1 = np.random.normal(size=(numLabels,1),
                           loc=0,
                           scale=(np.sqrt(6/numFeatures+numLabels+1)))
    return W_1, b_1


def initialize_weights_biases_hidden(numFeatures, numHidden, numLabels):
    # Values are randomly sampled from a Gaussian with a standard deviation of:
    #     sqrt(6 / (numInputNodes + numOutputNodes + 1))
    W_1 = np.random.normal(size=(numFeatures,numHidden),
                           loc=0,
                           scale=(np.sqrt(6/numFeatures+numHidden+1)))
    b_1 = np.random.normal(size=(numHidden,1),
                           loc=0,
                           scale=(np.sqrt(6/numFeatures+numHidden+1)))
    W_2 = np.random.normal(size=(numHidden,numLabels),
                           loc=0,
                           scale=(np.sqrt(6/numHidden+numLabels+1)))
    b_2 = np.random.normal(size=(numLabels,1),
                           loc=0,
                           scale=(np.sqrt(6/numHidden+numLabels+1)))
    return W_1, b_1, W_2, b_2


def feedforward(X,W_1,b_1):
    ### LAYER 1
    z_1 = np.add( np.dot( W_1.T,X ), b_1 )
    a_1 = sigmoid(z_1)
    return z_1,a_1


def feedforward_hidden(X, W_1,b_1, W_2,b_2):
    ### LAYER 1
    z_1 = np.add( np.dot( W_1.T,X ), b_1 )
    a_1 = sigmoid(z_1)
    ### LAYER 2
    z_2 = np.add( np.dot( W_2.T,a_1 ), b_2 )
    a_2 = sigmoid(z_2)
    return z_1,a_1,z_2,a_2

def demo():
    learning_rate=.1
    
    X = np.ndarray(buffer=np.array([-100,500,.1]),
                   shape=(3,1),
                   dtype=float)
    
    Y = np.ndarray(buffer=np.array([1,0]),
                   shape=(2,1),
                   dtype=float)

    W_1,b_1 = initialize_weights_biases(numFeatures=3,
                                        numLabels=2)
    
    z_1,a_1 = feedforward(X, W_1, b_1)
    
    error_final = compute_error_final(a_final=a_1, z_final=z_1, gold_label=Y)
    
    W_1 = update_weights_l(weights_l = W_1, 
                           activation_lminus1 = X, 
                           error_l = error_final, 
                           learning_rate = learning_rate)
    
    b_1 = update_biases_l(biases_l = b_1,
                          error_l = error_final,
                          learning_rate = learning_rate)

    print('done!')


if __name__ == '__main__':
    demo()
