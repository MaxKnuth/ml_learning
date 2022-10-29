import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import Perceptron

#_samples = Anzahl an Datenpunkten pro Kategorie
# features = Anzahl der Kategorien
# centers = Anzahl der Punkthaufen
# random_state = Seed für Zufallsgenerator

X,y = datasets.make_blobs(n_samples=100,n_features=2,centers=2,random_state=3)

#Klassifikation
# Aufbau eines Rasters, um auszuwerten und zu zeichen
s = 0.02 # Schrittweite im Raster

# Ermittlung der 1D Arrays, die die Koordinate im Ratser repräsentieren
x_min, x_max = X[:, 0].min() - 1, X[:,0].max() + 1 # x Achse
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1 # y Achse

# np arange liefert array mit werten beginnend bei _min in s schritten bis x_max
# np meshfrid liefert die Koordinatenmatrix
xx,yy = np.meshgrid(np.arange(x_min, x_max, s), np.arange(y_min, y_max, s))


# Instanziierung des Perceptrons
# max_iter = Anzahl Iterationen
# tol 0 Stoppkriterium
Perceptron = Perceptron(random_state=42, max_iter=1000)
# Nun soll gelernt werden
Perceptron.fit(X,y)

# Auswertung für alle Rasterpunkte, dazu wird ein Array aus ratserpunkten erzeugt
# ravel() erzeigt ein 1-D array
# np.c_ erzeugt ein Punktpärchen Arrays für jeden rasterpunkt die als Input für das Pereptron dienen
Prediction = Perceptron.predict(np.c_[xx.ravel(), yy.ravel()])


# Daten anzeigen
# Plotten der Punkthaufen
plt.plot(X[:,0][y==0],X[:,1][y==0], 'b^') # blaue Dreiecke
plt.plot(X[:,0][y==1],X[:,1][y==1], 'ys') # gelbe Quadrate
# Umwandlung des 1-D arrays der preicitions in die Rasterdimension ( wie die Dimension von xx)
Prediction = Prediction.reshape(xx.shape)
# Plotten der Vorhersagen
plt.contourf(xx,yy,Prediction, cmap=plt.cm.Paired)
plt.show()


