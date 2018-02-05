'''
Implement the incomplete functions in the file.

Arguments:
    None
Returns:
    None
'''
import numpy as np
from read_dataset import mnist
import matplotlib.pyplot as plt
import pdb

def sigmoid(scores):
    '''
    calculates the sigmoid of scores
    Inputs: 
        scores array
    Returns:
        sigmoid of scores
    '''
    sigmoid = 1 / (1 + np.exp(-scores))
    return sigmoid

def step(X, Y, w, b):
    '''
    Implements cost and gradients for the logistic regression with one batch of data
    Inputs:
        X = (n,m) matrix
        Y = (1,m) matrix of labels
        w = (n,1) matrix
        b = scalar
    Returns:
        cost = cost of the batch
        gradients = dictionary of gradients dw and db
    '''

    scores = np.dot(w.T, X) + b
    A = sigmoid(scores)

    # compute the gradients and cost 
    m = X.shape[1]  # number of samples in the batch
    cost = (-1 / m) * (np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))) #None
    dw = (1 / m) * (np.matmul(X, (A - Y).T))    #None
    db = (1 / m) * (np.sum(A - Y))  #None
    gradients = {"dw": dw,
                 "db": db}
    return cost, gradients

def optimizer(X, Y, w, b, learning_rate, num_iterations):
    '''
    Implements gradient descent and updates w and b
    Inputs: 
        X = (n,m) matrix
        Y = (1,m) matrix of labels
        w = (n,1) matrix
        b = scalar
        learning_rate = rate at which the gradient is updated
        num_iterations = total number of batches for gradient descent
    Returns:
        parameters = dictionary containing w and b
        gradients = dictionary contains gradeints dw and db
        costs = array of costs over the training 
    '''
    costs = []
    # update weights by gradient descent
    for ii in range(num_iterations):
        cost, gradients = step(X, Y, w, b)
        dw = gradients["dw"]
        db = gradients["db"]
        w = w - learning_rate * dw #None
        b = b - learning_rate * db #None

        if ii % 10 == 0:
            costs.append(cost)
        if ii % 100 == 0:
            print("Cost at iteration %i is: %f" %(ii, cost))
    parameters = {"w": w, "b": b}
    return parameters, gradients, costs

def classify(X, w, b):
    '''
    Outputs the predictions for X

    Inputs: 
        X = (n,m) matrix
        w = (n,1) matrix
        b = scalar

    Returns:
        YPred = (1,m) matrix of predictions
    '''
    scores = np.dot(w.T, X) + b
    A = sigmoid(scores)
    YPred = np.zeros((1,X.shape[1])) #None
    for index, yhat in enumerate(A[0,:]):
        if yhat < 0.5:
            YPred[0, index] = 0
        else:
            YPred[0, index] = 1

    return YPred

    
def main():
    # getting the sbuset dataset from MNIST
    train_data, train_label, test_data, test_label = mnist()

    # initialize learning rate and num_iterations
    learning_rate = 0.1
    num_iterations = 2000

    # initialize w as array (d,1) and b as a scalar
    w = np.zeros(shape=(train_data.shape[0], 1)) #None
    b = 0.0 #None

    # learning the weights by using optimize function
    parameters, gradients, costs = optimizer(train_data, \
                    train_label, w, b, learning_rate, num_iterations)
    w = parameters["w"]
    b = parameters["b"]
    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data,w,b)
    test_Pred = classify(test_data,w,b)
    trAcc = (1 - np.mean(np.abs(train_label-train_Pred))) *100 #None
    teAcc = (1 - np.mean(np.abs(test_label-test_Pred)))*100 #None
    print("Accuracy for training set is {} %".format(trAcc))
    print("Accuracy for testing set is {} %".format(teAcc))

    #Plotting train Error vs Iteration
    iteration = [iter for iter in range(0,2000,10)]
    plt.plot(iteration, costs, color='blue')
    points = plt.scatter(iteration, costs, c='red', s=5)
    plt.title("Logistic Regression")
    plt.xlabel("Iteration")
    plt.ylabel("Train Error")
    plt.text(750, 0.6, "Training Accuracy = " + str(trAcc))
    plt.text(750, 0.5, "Testing  Accuracy = " + str(teAcc))
    plt.savefig("Logistic Regression")
    plt.show()

if __name__ == "__main__":
    main()
