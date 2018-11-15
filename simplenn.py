import numpy as np
import random



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

def initialize_weights_biases(numFeatures, numHidden, numLabels):
    # Values are randomly sampled from a Gaussian with a standard deviation of:
    #     sqrt(6 / (numInputNodes + numOutputNodes + 1))
    W_1 = np.random.normal(size=(numFeatures,numHidden),
                           loc=0,
                           scale=(np.sqrt(6/numFeatures+numHidden+1)))
    b_1 = np.random.normal(size=(numHidden,1),
                           loc=0,
                           scale=(np.sqrt(6/numFeatures+numHidden+1)))
    
    W_2A = np.random.normal(size=(numHidden,numLabels),
                           loc=0,
                           scale=(np.sqrt(6/numHidden+numLabels+1)))
    b_2A = np.random.normal(size=(numLabels,1),
                           loc=0,
                           scale=(np.sqrt(6/numHidden+numLabels+1)))

    W_2B = np.random.normal(size=(numHidden,numLabels),
                           loc=0,
                           scale=(np.sqrt(6/numHidden+numLabels+1)))
    b_2B = np.random.normal(size=(numLabels,1),
                           loc=0,
                           scale=(np.sqrt(6/numHidden+numLabels+1)))
    return W_1, b_1, W_2A, b_2A, W_2B, b_2B


def feedforward_pass(X, # input
                     W_1, b_1, # first hidden layer
                     W_2A, b_2A, # output layer A
                     W_2B, b_2B): # output layer B
    
    ### LAYER 1
    z_1 = np.add( np.dot( W_1.T,X ), b_1 )
    a_1 = sigmoid(z_1)

    ### LAYER 2A
    z_2A = np.add( np.dot( W_2A.T,a_1 ), b_2A )
    a_2A = sigmoid(z_2A)

    ### LAYER 2B
    z_2B = np.add( np.dot( W_2B.T,a_1 ), b_2B )
    a_2B = sigmoid(z_2B)

    return z_1,a_1,z_2A,a_2A,z_2B,a_2B


def create_data(limit):
    Xs=[]
    Ys=[]
    for i in range(10):
        for j in range(10):
            for k in range(10):
                num=str(i)+str(j)+str(k)
                if int(num) < limit:
                    label=np.array([1.,0.,0.])
                elif int(num) < 2*limit:
                    label=np.array([0.,1.,0.])
                else:
                    label=np.array([0.,0.,1.])
                    
                X = np.ndarray(buffer=np.array([float(i),float(j),float(k)]),
                               shape=(3,1),
                               dtype=float)
                    
                Y = np.ndarray(buffer=label,
                               shape=(3,1),
                               dtype=float)
                    
                Xs.append(X)
                Ys.append(Y)
                
    return(Xs, Ys)


def create_data(XorY):
    
    Xs=[]
    Ys=[]

    for anchor1 in range(10):
        for anchor2 in range(1,10):
            if anchor1 == 0:
                label=np.array([1.,0.,0.])
            else:
                label=np.array([0.,1.,0.])

            if XorY == 'x':
                
                X = np.ndarray(buffer=np.array([float(anchor1),float(anchor2),0.0]),
                               shape=(3,1),
                               dtype=float)

            elif XorY == 'y':
                
                X = np.ndarray(buffer=np.array([float(anchor2),float(anchor1),0.0]),
                               shape=(3,1),
                               dtype=float)
                
                
            Y = np.ndarray(buffer=label,
                           shape=(3,1),
                           dtype=float)
                    
            Xs.append(X)
            Ys.append(Y)

    print('done creating data')
                
    return(Xs, Ys)

                
def demo():

    Xs, Ys_A = create_data('x')
    Xs, Ys_B = create_data('y')
    
    learning_rate=.001

    example=0
    num_examples=1000
    
    W_1,b_1,W_2A,b_2A,W_2B,b_2B = initialize_weights_biases(numFeatures=3,
                                                            numHidden=3,
                                                            numLabels=3)
    
    print("Beginning Training")

    while example<num_examples:
        
        i=random.randint(0,5)
        print(i)

        X=Xs[i]
        Y_A=Ys_A[i]
        Y_B=Ys_B[i]

                    
        z_1,a_1,z_2A,a_2A,z_2B,a_2B = feedforward_pass(X, W_1, b_1, W_2A, b_2A , W_2B, b_2B )

        
        #
        # CALCULATE ERROR AND UPDATE TASK A
        #

        
        error_final_A = compute_error_final(a_final=a_2A, z_final=z_2A, gold_label=Y_A)
        
        W_2A = update_weights_l(weights_l = W_2A, 
                               activation_lminus1 = a_1, 
                               error_l = error_final_A, 
                               learning_rate = learning_rate)
        
        b_2A = update_biases_l(biases_l = b_2A,
                              error_l = error_final_A,
                              learning_rate = learning_rate)



        #
        # CALCULATE ERROR AND UPDATE TASK B
        #

        
        error_final_B = compute_error_final(a_final=a_2B, z_final=z_2B, gold_label=Y_B)
        
        W_2B = update_weights_l(weights_l = W_2B, 
                               activation_lminus1 = a_1, 
                               error_l = error_final_B, 
                               learning_rate = learning_rate)
        
        b_2B = update_biases_l(biases_l = b_2B,
                              error_l = error_final_B,
                              learning_rate = learning_rate)
        


        #
        # CALCULATE ERROR AND UPDATE SHARED HIDDEN LAYER
        #

        
        error_l_A = compute_error_l(weights_lplus1 = W_2A, 
                                  error_lplus1 = error_final_A, 
                                  z_l = z_1)

        error_l_B = compute_error_l(weights_lplus1 = W_2B, 
                                  error_lplus1 = error_final_B, 
                                  z_l = z_1)

        error_l = error_l_A + error_l_B
        
        
        W_1 = update_weights_l(weights_l = W_1, 
                               activation_lminus1 = X, 
                               error_l = error_l, 
                               learning_rate = learning_rate)
        
        b_1= update_biases_l(biases_l = b_1,
                             error_l = error_l,
                             learning_rate = learning_rate)
        
        
        example+=1

        

    ### VISUALIZATIONS !!! ###
    print("Starting Viz")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # plotting
    
    data_plot={}
    for i in range(10):
        for j in range(10):
            for k in range(10):
                num=str(i)+str(j)+str(k)
                try:
                    X=Xs[int(num)]
                except:
                    break
                
                if int(num) < 333:
                    
                    z_1,a_1,z_2A,a_2A,z_2B,a_2B = feedforward_pass(X, W_1, b_1, W_2A, b_2A, W_2B, b_2B)
                    data_plot[num]=[X,a_1,a_2A,a_2B, 'r']
                    
                elif int(num) < 666:

                    z_1,a_1,z_2A,a_2A,z_2B,a_2B = feedforward_pass(X, W_1, b_1, W_2A, b_2A, W_2B, b_2B)
                    data_plot[num]=[X,a_1,a_2A,a_2B, 'b']
       
                else:

                    z_1,a_1,z_2A,a_2A,z_2B,a_2B = feedforward_pass(X, W_1, b_1, W_2A, b_2A, W_2B, b_2B)
                    data_plot[num]=[X,a_1,a_2A,a_2B, 'g']

    x_org=[]
    y_org=[]
    z_org=[]
    x_a1=[]
    y_a1=[]
    z_a1=[]
    x_a2A=[]
    y_a2A=[]
    z_a2A=[]
    x_a2B=[]
    y_a2B=[]
    z_a2B=[]
    color=[]
    
    for i in range(1000):
        i="{0:03}".format(i)

        X,a_1,a_2A,a_2B, color_i = data_plot[str(i)]

        # org data plot
        x_org.append(X[0])
        y_org.append(X[1])
        z_org.append(X[2])

        # first activation plot
        x_a1.append(a_1[0])
        y_a1.append(a_1[1])
        z_a1.append(a_1[2])

        # final activation plot A
        x_a2A.append(a_2A[0])
        y_a2A.append(a_2A[1])
        z_a2A.append(a_2A[2])

        # final activation plot B
        x_a2B.append(a_2B[0])
        y_a2B.append(a_2B[1])
        z_a2B.append(a_2B[2])
        
        color.append(color_i)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_org,y_org,z_org,color=color)
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_a1,y_a1,z_a1,color=color)
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_a2A,y_a2A,z_a2A,color=color)
    plt.show()

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_a2B,y_a2B,z_a2B,color=color)
    plt.show()

    ### END VISUALIZATIONS ###
if __name__ == '__main__':
    demo()
