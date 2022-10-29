import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_random_state
from sklearn.utils.multiclass import unique_labels


class PerceptronEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, n_iterations=20, random_state=None):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.errors = []

    def heaviside(self, x):
        if x < 0:
            result = 0
        else:
            result = 1

        return result

    def fit(self, X=None, y=None):
        # Erzeugt Zufallsgenerator
        random_state = check_random_state(self.random_state)
        # Erzeugt random geweichtsvektor mit richtiger Länge
        self.w = random_state.random_sample(np.size(X, 1))
        # Prüft, ob X und y die korrekte shape haben
        X, y = check_X_y(X, y)
        # Zieht distinct Zielwerte, also 0 oder 1
        self.classes_ = unique_labels(y)
        #Abspeicherung der Lerndaten für predict
        self.X_ = X
        self.y_ = y

        # Lernen bzw. iterative Gewichtsanapssung
        for i in range(self.n_iterations):
            # Nimmt eine Testzeile bzw. deren index aus der Testmatrix
            rand_index = random_state.randint(0, np.size(X, 0))
            # Nimmt den Testinput
            x_ = X[rand_index]
            # Nimmt den passenden Output
            y_ = y[rand_index]
            # Berechnet vorhergesagten Output
            y_hat = self.heaviside(np.dot(self.w, x_))
            # Berechnet error
            error = y_ - y_hat
            # Speichert error
            self.errors.append(error)
            # passt Gewichte an
            self.w += x_ * error

        return self

    def predict(self, x):
        check_is_fitted(self, ['X_', 'y_'])
        y_hat = self.heaviside(np.dot(self.w, x))
        return y_hat

    def plot(self):
        fignr = 1
        plt.figure(fignr, figsize=(10, 10))
        plt.plot(self.errors)
        plt.style.use('seaborn-whitegrid')
        plt.xlabel('Iterationen')
        plt.ylabel('Error')
        plt.show()


def main():
    X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    y = np.array([0, 1, 1, 1])
    # Erzeugt Perceptron Objekt mit #Verbesserungsiterations und dem random seed
    Perceptron = PerceptronEstimator(4, 10)
    # Ruf fit methode auf
    Perceptron.fit(X, y)
    #x = np.array([1, 0, 0])
    for index, x in enumerate(X):
        p = Perceptron.predict(x)
        print("{}: {} -> {}".format(x, y[index], p))

    Perceptron.plot()


main()
