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
    #print("np.dot(weights_lplus1, error_lplus1.T) * sigmoid_prime(z_l) == ",  weights_lplus1.shape, error_lplus1.shape, z_l.shape)
    error_l = np.dot(error_lplus1, weights_lplus1.T) * sigmoid_prime(z_l)
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
    #print("compute_dC_dW_l: ", error_l.shape, activation_lminus1.shape)
    return np.dot(error_l.T, activation_lminus1)


###
### GRADIENT DESCENT
###

def update_weights_l(weights_l, activation_lminus1, error_l, learning_rate):
    #print("weights_l.shape, activation_lminus1.shape, error_l.shape = ", weights_l.shape, activation_lminus1.shape, error_l.shape)
    
    dC_dW = compute_dC_dW_l(error_l, activation_lminus1)
    new_weights= weights_l - (learning_rate*dC_dW).T
    return new_weights
 
def update_biases_l(biases_l, error_l, learning_rate):
    #print("biases_l.shape, error_l.shape, learning_rate = ", biases_l.shape, error_l.shape, learning_rate)
    dC_db = compute_dC_db_l(error_l)
    new_biases = biases_l - (learning_rate*dC_db)
    return new_biases

def initialize_weights_biases(numFeatures, numHidden, numLabels):
    # Values are randomly sampled from a Gaussian with a standard deviation of:
    #     sqrt(6 / (numInputNodes + numOutputNodes + 1))
    W_1 = np.random.normal(size=(numFeatures,numHidden),
                           loc=0,
                           scale=(np.sqrt(6/numFeatures+numHidden+1)))
    b_1 = np.random.normal(size=(1,numHidden),
                           loc=0,
                           scale=(np.sqrt(6/numFeatures+numHidden+1)))
    
    W_2A = np.random.normal(size=(numHidden,numLabels),
                           loc=0,
                           scale=(np.sqrt(6/numHidden+numLabels+1)))
    b_2A = np.random.normal(size=(1,numLabels),
                           loc=0,
                           scale=(np.sqrt(6/numHidden+numLabels+1)))

    W_2B = np.random.normal(size=(numHidden,numLabels),
                           loc=0,
                           scale=(np.sqrt(6/numHidden+numLabels+1)))
    b_2B = np.random.normal(size=(1,numLabels),
                           loc=0,
                           scale=(np.sqrt(6/numHidden+numLabels+1)))
    return W_1, b_1, W_2A, b_2A, W_2B, b_2B


def feedforward_pass(X, W_1, b_1, W_2A, b_2A, W_2B, b_2B):
    
    ### LAYER 1
    z_1 = np.add( np.dot( X,W_1), b_1 )
    a_1 = sigmoid(z_1)

    ### LAYER 2A
    z_2A = np.add( np.dot(a_1, W_2A), b_2A )
    a_2A = sigmoid(z_2A)

    ### LAYER 2B
    z_2B = np.add( np.dot( a_1, W_2B), b_2B )
    a_2B = sigmoid(z_2B)

    return z_1,a_1,z_2A,a_2A,z_2B,a_2B


def create_task_data(num_feats, num_classes, num_examples):
    '''
    creates a classification task from multi-dimensional Gaussians
    '''
    num_train=int(num_examples*.9)
    num_test=int(num_examples*.1)
    
    trainX=np.empty((0,num_feats), float)
    trainY=np.empty((0,num_classes), float)
    testX=np.empty((0,num_feats), float)
    testY=np.empty((0,num_classes), float)

    for class_num in range(num_classes):
        # make label for this class
        label = np.zeros(num_classes)
        label[class_num] = 1.0
        trainLabels = np.tile(label, (num_train,1))
        testLabels = np.tile(label, (num_test,1))

        # generate features from gaussian
        mean = np.random.uniform(0,1,[1,num_feats])
        cov = np.identity(num_feats)
        trainFeatures = np.random.multivariate_normal(mean[0], cov, num_train)
        testFeatures = np.random.multivariate_normal(mean[0], cov, num_test)
        trainX=np.append(trainX,trainFeatures, axis=0)
        trainY=np.append(trainY,trainLabels, axis=0)

        testX=np.append(testX,testFeatures, axis=0)
        testY=np.append(testY,testLabels, axis=0)

    print('done creating data for one task')
                
    return trainX, trainY, testX, testY

                
def demo():
    num_feats=5
    num_classes=20
    num_examples=1000

    num_hidden=1000
    learning_rate=.001
    num_epochs=1000
    total_iterations=num_examples*2 # per epoch// the 2 is because we have 2 tasks
    
    org_Xs_A, org_Ys_A, test_X_A, test_Y_A = create_task_data(num_feats, num_classes, num_examples)
    org_Xs_B, org_Ys_B, test_X_B, test_Y_B = create_task_data(num_feats, num_classes, num_examples)

    W_1,b_1,W_2A,b_2A,W_2B,b_2B = initialize_weights_biases(num_feats, num_hidden, num_classes)

    #print("## Beginning Training ##")

    for epoch in range(num_epochs):
        epoch_loss=0
        Xs_A=org_Xs_A
        Ys_A=org_Ys_A
        Xs_B=org_Xs_B
        Ys_B=org_Ys_B
        iteration=0

        ## TRAIN LOOP ##
        while iteration<total_iterations:
            iteration+=1

            if iteration%2:
                task='A'
                rand_example=np.random.randint(0,Xs_A.shape[0])
                X=Xs_A[rand_example].reshape(1,num_feats)
                Y_A=Ys_A[rand_example]
                Xs_A=np.delete(Xs_A,rand_example,0)
                Ys_A=np.delete(Ys_A,rand_example,0)
            else:
                task='B'
                rand_example=np.random.randint(0,Xs_B.shape[0])
                X=Xs_B[rand_example].reshape(1,num_feats)
                Y_B=Ys_B[rand_example]
                Xs_B=np.delete(Xs_B,rand_example,0)
                Ys_B=np.delete(Ys_B,rand_example,0)
            
            ## FEEDFORWARD ##
            
            z_1,a_1,z_2A,a_2A,z_2B,a_2B = feedforward_pass(X, W_1, b_1, W_2A, b_2A, W_2B, b_2B )

            ## MULTI-TASK BACKPROP ##
            if task=='A':
                # CALCULATE ERROR AND UPDATE TASK A-specific parameters
                error_final_A = compute_error_final(a_2A, z_2A, Y_A)
                W_2A = update_weights_l(W_2A, a_1, error_final_A, learning_rate)
                b_2A = update_biases_l(b_2A, error_final_A, learning_rate)
                # CALCULATE ERROR AND UPDATE SHARED HIDDEN LAYER
                error_l = compute_error_l(W_2A, error_final_A, z_1)
                W_1 = update_weights_l(W_1, X, error_l, learning_rate)
                b_1= update_biases_l(b_1, error_l, learning_rate)
                loss=np.sum(error_final_A**2)
                # #print("TASK A: ", epoch, iteration, ": ", loss)

            elif task=='B':
                # CALCULATE ERROR AND UPDATE TASK B params
                error_final_B = compute_error_final(a_2B, z_2B, Y_B)
                W_2B = update_weights_l(W_2B, a_1, error_final_B, learning_rate)
                b_2B = update_biases_l(b_2B, error_final_B, learning_rate)
                
                # CALCULATE ERROR AND UPDATE SHARED HIDDEN LAYER
                error_l = compute_error_l(W_2B, error_final_B, z_1)
                W_1 = update_weights_l(W_1, X, error_l, learning_rate)
                b_1= update_biases_l(b_1, error_l, learning_rate)

                loss=np.sum(error_final_B**2)
                # print("TASK B: ", epoch, iteration, ": ", loss)

            epoch_loss+=loss
        # print("LOSS: ", epoch_loss)

        print("###")

        ## TEST LOOP ##
        error_A=0
        for i in range(test_X_A.shape[0]):
            X=Xs_A[i].reshape(1,num_feats)
            Y_A=Ys_A[i]
            z_1,a_1,z_2A,a_2A,z_2B,a_2B = feedforward_pass(X, W_1, b_1, W_2A, b_2A, W_2B, b_2B )
            error_final_A = compute_error_final(a_2A, z_2A, Y_A)
            error_A+=np.sum(error_final_A**2)
        print("TEST ERROR A: ", error_A)
        
        error_B=0
        for i in range(test_X_B.shape[0]):
            X=Xs_B[i].reshape(1,num_feats)
            Y_B=Ys_B[i]
            z_1,a_1,z_2A,a_2A,z_2B,a_2B = feedforward_pass(X, W_1, b_1, W_2A, b_2A, W_2B, b_2B )
            error_final_B = compute_error_final(a_2B, z_2B, Y_B)
            error_B+=np.sum(error_final_B**2)
        print("TEST ERROR B: ", error_B)



                
    # ##print(W_1,b_1,W_2A,b_2A,W_2B,b_2B)
    # ### VISUALIZATIONS !!! ###
    # ##print("Starting Viz")
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # # plotting
    
    # data_plot={}
    # # for i in range(10):
    # #     for j in range(10):
    # #         for k in range(10):
    
    # for num in range(1000):
    #     num=str("{0:03}".format(num))
        
    #     try:
    #         X=Xs[int(num)]
    #     except:
    #         break
        
    #     if int(num) < 333:
            
    #         z_1,a_1,z_2A,a_2A,z_2B,a_2B = feedforward_pass(X, W_1, b_1, W_2A, b_2A, W_2B, b_2B)
    #         data_plot[num]=[X,a_1,a_2A,a_2B, 'r']
            
    #     elif int(num) < 666:
            
    #         z_1,a_1,z_2A,a_2A,z_2B,a_2B = feedforward_pass(X, W_1, b_1, W_2A, b_2A, W_2B, b_2B)
    #         data_plot[num]=[X,a_1,a_2A,a_2B, 'b']
            
    #     else:
            
    #         z_1,a_1,z_2A,a_2A,z_2B,a_2B = feedforward_pass(X, W_1, b_1, W_2A, b_2A, W_2B, b_2B)
    #         data_plot[num]=[X,a_1,a_2A,a_2B, 'g']
            
    # x_org=[]
    # y_org=[]
    # z_org=[]
    # x_a1=[]
    # y_a1=[]
    # z_a1=[]
    # x_a2A=[]
    # y_a2A=[]
    # z_a2A=[]
    # x_a2B=[]
    # y_a2B=[]
    # z_a2B=[]
    # color=[]
    
    # for i in range(1000):
    #     i="{0:03}".format(i)
    #     #print(i)
    #     X,a_1,a_2A,a_2B, color_i = data_plot[str(i)]

    #     # org data plot
    #     x_org.append(X[0])
    #     y_org.append(X[1])
    #     z_org.append(X[2])

    #     # first activation plot
    #     x_a1.append(a_1[0])
    #     y_a1.append(a_1[1])
    #     z_a1.append(a_1[2])

    #     # final activation plot A
    #     x_a2A.append(a_2A[0])
    #     y_a2A.append(a_2A[1])
    #     z_a2A.append(a_2A[2])

    #     # final activation plot B
    #     x_a2B.append(a_2B[0])
    #     y_a2B.append(a_2B[1])
    #     z_a2B.append(a_2B[2])
        
    #     color.append(color_i)


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_org,y_org,z_org,color=color)
    # plt.show()
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_a1,y_a1,z_a1,color=color)
    # plt.show()
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_a2A,y_a2A,z_a2A,color=color)
    # plt.show()

    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_a2B,y_a2B,z_a2B,color=color)
    # plt.show()

    ### END VISUALIZATIONS ###
if __name__ == '__main__':
    demo()
