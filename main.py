import matplotlib.pyplot as plt
from random import choice
from numpy import array, dot, zeros, random

heavside = lambda x:0 if x < 0 else 1

def fit(iterations, training_data_set,w):
    weights = []
    errors = []

    for i in range(iterations):
        training_data = choice(training_data_set)

        x = training_data[0]
        y = training_data[1]
        #print('Input is: ' + str(x))
        y_hat=heavside(dot(w,x))

        #print('y_hat is: ' + str(y))
        #print('y is: ' + str(y))
        error = y - y_hat
        errors.append(error)
        weights.append(w)
        #print('Weights before error: ' + str(w))
        #print('error: ' + str(error))
        w += error*x
        #print('Weights after error: ' + str(w))

    return errors, weights

def main():
    trainings_data_set = [
        (array([1, 0, 0]), 0),
        (array([1, 0, 1]), 1),
        (array([1, 1, 0]), 1),
        (array([1, 1, 1]), 1),
    ]
    random.seed(12)

    # w = zeros(3)
    w=[0,0,0]
    iterations = 30

    errors,weights = fit(iterations,trainings_data_set, w)
    w = weights[iterations-1]
    print("Gewichtsvektor am Ende des Trainings:")
    print(w)

    print("Auswertung am Ende des Trainings")
    for x,y in trainings_data_set:
        y_hat = heavside(dot(x,w))
        print("{}: {} -> {}".format(x,y,y_hat))


    fignr = 1
    plt.figure(fignr,figsize=(10,10))
    plt.plot(errors)
    plt.style.use('seaborn-whitegrid')
    plt.xlabel('Iteration')
    plt.ylabel('error')
    plt.show()

main()





