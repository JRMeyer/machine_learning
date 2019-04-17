import numpy as np

def initialize_weights_biases(numFeatures, numLayers, numHidden, numLabels):
    '''
    Values are randomly sampled from a Gaussian with a standard deviation of:
         sqrt(6 / (numInputNodes + numOutputNodes + 1))
    '''
    
    layers=[]
    stdev=np.sqrt(6/numFeatures+numHidden+1)
    
    for i in range(numLayers):
        if i==0:
            # First layer
            W = np.random.normal(size=(numFeatures,numHidden),
                                 loc=0, scale=(stdev))
            b = np.random.normal(size=(numHidden,1),
                                 loc=0, scale=(stdev))
            layers.append((W,b))
            
        elif i==(numLayers-1):
            # Last layer
            W = np.random.normal(size=(numHidden,numLabels),
                                 loc=0, scale=(stdev))
            b = np.random.normal(size=(numLabels,1),
                                 loc=0, scale=(stdev))
            layers.append((W,b))
        else:
            # Hidden layer
            W = np.random.normal(size=(numHidden,numHidden),
                                 loc=0, scale=(stdev))
            b = np.random.normal(size=(numHidden,1),
                                 loc=0, scale=(stdev))
            layers.append((W,b))   
            
    return layers


def feedforward(X, layers):

    # treat the input features as 'a' (activations)
    # for the first layer
    a = X
    # keep all activations for all layers in list
    activations = []
    for layer in layers:
        # First layer gets input (X)
        z = np.add( np.dot( layer[0].T, a ), layer[1] )
        a = sigmoid(z)
        activations.append((z,a))
        
    return activations



### ACTIVATION FUNCTION

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

### FOUR EQUATIONS OF BACKPROP

def compute_error_final(a_final, z_final, gold_label):
    '''
    computing this as if we were using a quadratic cost function like MSE
    '''
    error_final = (a_final - gold_label) * sigmoid_prime(z_final)
    return error_final
    
def compute_error_l(weights_lplus1, error_lplus1, z_l):
    '''
    compute the total error at layer 'l'
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
    '''
    update weights at layer 'l'
    '''
    dC_dW = compute_dC_dW_l(error_l, activation_lminus1)
    new_weights= weights_l - (learning_rate*dC_dW).T
    return new_weights
 
def update_biases_l(biases_l, error_l, learning_rate):
    '''
    update biases at layer 'l'
    '''
    dC_db = compute_dC_db_l(error_l)
    new_biases = biases_l - learning_rate*dC_db
    return new_biases



def generate_data():
    Xs=[]
    Ys=[]
    for i in range(10):
        for j in range(10):
            for k in range(10):
                num=str(i)+str(j)+str(k)
                if int(num) < 333:
                    label=np.array([1.,0.,0.])
                elif int(num) < 666:
                    label=np.array([0.,1.,0.])
                else:
                    label=np.array([0.,0.,1.])
                    
                X = np.ndarray(buffer=np.array([i,j,k]),
                               shape=(3,1),
                               dtype=float)
                    
                Y = np.ndarray(buffer=label,
                               shape=(3,1),
                               dtype=float)
                    
                Xs.append(X)
                Ys.append(Y)
    return Xs,Ys

if __name__ == "__main__":

    Xs, Ys = generate_data()

    numLayers=4
    layers= initialize_weights_biases(numFeatures=3,
                                      numLayers=numLayers,
                                      numHidden=3,
                                      numLabels=3)
        
    epoch=0
    num_epochs=100
    learning_rate=.001

    
    while epoch<num_epochs:
        
        for i in range(1000):

            X=Xs[i]
            Y=Ys[i]
            
            activations = list(reversed(feedforward(X,layers)))

            updated_layers=[]
            for i, layer in enumerate(list(reversed(layers))):

                if i==0:
                    # Final layer
                    error = compute_error_final(a_final = activations[i][1],
                                                z_final = activations[i][0],
                                                gold_label = Y)

                    W = update_weights_l(weights_l = layer[0], 
                                         activation_lminus1 = activations[i+1][1],
                                         error_l = error, 
                                         learning_rate = learning_rate)
            
                    b = update_biases_l(biases_l = layer[1],
                                        error_l = error,
                                        learning_rate = learning_rate)
                    updated_layers.append((W,b))
                    
                elif i==(numLayers-1):
                    # Input Layer
                    error = compute_error_l(weights_lplus1 = W, 
                                            error_lplus1 = error, 
                                            z_l = activations[i][0])
                
                    W = update_weights_l(weights_l = layer[0], 
                                         activation_lminus1 = X, 
                                         error_l = error, 
                                         learning_rate = learning_rate)
                    
                    b= update_biases_l(biases_l = layer[1],
                                       error_l = error,
                                       learning_rate = learning_rate)
                    updated_layers.append((W,b))
                else:
                    # Hidden Layer
                    error = compute_error_l(weights_lplus1 = W, 
                                            error_lplus1 = error, 
                                            z_l = activations[i][0])
                
                    W = update_weights_l(weights_l = layer[0], 
                                         activation_lminus1 = activations[i+1][1],
                                         error_l = error, 
                                         learning_rate = learning_rate)
                    
                    b= update_biases_l(biases_l = layer[1],
                                       error_l = error,
                                       learning_rate = learning_rate)
                    updated_layers.append((W,b))
            
            layers=list(reversed(updated_layers))
            
            epoch+=1
            print(layers)
