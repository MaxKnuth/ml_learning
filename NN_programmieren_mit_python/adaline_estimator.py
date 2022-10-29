import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.utils.multiclass import unique_labels

class AdalineEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, eta=.001, n_iterations=500, random_state=None):
        # eta: Lernrate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.errors = []
        self.eta = eta
        self.w = []
        self.wAll = []

    # Der gewichtete Input
    def net_i(self,x):
        return np.dot(x, self.w)

    # activation function
    def activation(self,x):
        return self.net_i(x)

    # output function
    def output(self,x):
        if self.activation(x) >= 0.0:
            return 1
        else:
            return -1

    def fit(self, X=None, y=None):
        # Erzeugung des Zufallsgenerators
        random_state = check_random_state(self.random_state)
        # Gewichtinitialisierung
        self.w = random_state.random_sample(np.size(X,1))
        # Prüfe ob X und y die korrekte Shape haben
        X,y = check_X_y(X,y)
        # Lerndaten für spätere Vernwedung ablegen
        self.X_ = X
        self.y_ = y

        #Lernen mit Gradientenabstieg
        for i in range(self.n_iterations):
            # zufälliges Auswählen eines Testdatensatzes für bach_size = 1
            rand_index = random_state.randint(0, np.size(X,0))
            # Testdatensatz
            x_ = X[rand_index]
            y_ = y[rand_index]
            # net input s berechnen
            s = self.activation(x_)
            # error berechnen als Quadrat der Differenz
            error = (y_ - s)**2
            self.errors.append(error)
            # Online Adaline lernen+
            self.w += self.eta*x_*(y_ - s)
            self.wAll.append(self.w.copy())

    # Auswertung
    def predict(self, x):
        # Auswerten eines Test Input Vektors
        check_is_fitted(self, ['X_', 'y_'])
        y_hat = self.output(x)
        return y_hat

    def plot(self):
        x1 = []
        x2 = []
        colors = []

        for i in range(self.X_.shape[0]):
            x1.append(self.X_[i][1])
            x2.append(self.X_[i][2])
            y = self.y_[i]
            if y==1:
                colors.append('r')
            else:
                colors.append('b')

        #Raster
        plt.style.use('seaborn-whitegrid')
        # Errors
        plt.plot(self.errors)
        #Learning curve
        plt.figure(1)
        plt.show()
        #Scatter
        plt.figure(2)
        plt.scatter(x1,x2, c = colors)
        # Result Line
        x1Line = np.linspace(0.0,1.0,2)
        x2Line = lambda x1,w0,w1,w2: (-x1*w1 - w0)/w2;
        alpha = 0.0
        for idx, weight in enumerate(self.wAll):
            if idx % 100 == 0:
                alpha = 1.0
                plt.plot(x1Line,x2Line(x1Line,weight[0], weight[1],weight[2]), alpha = alpha, linestyle = 'solid', label=str(idx), linewidth=1.5)
                plt.legend(loc='best', shadow=True)
        plt.plot(x1Line,x2Line(x1Line,weight[0],weight[1],weight[2]), alpha=alpha, linestyle='solid', label =str(idx), linewidth=2.0)
        plt.legend(loc='best', shadow = True)
        plt.show()

def main():
    # Erzeugung des Zufallsgenerators (RNG)
    random_state = check_random_state(1)
    # Initialisierung Datensätze
    I = []
    o = []

    for x in random_state.random_sample(20):
        y = random_state.random_sample()
        I.append([1,x,y+0.5])
        o.append(1)

    for x in random_state.random_sample(20):
        y = random_state.random_sample()
        I.append([1,x,y-0.5])
        o.append(-1)

    # Trainingsdaten
    X = np.array(I)
    y = np.array(o)
    # Estimater
    Adaline = AdalineEstimator(eta=0.1, n_iterations=900, random_state=10)
    # Lernen
    Adaline.fit(X,y)
    Adaline.plot()

main()










