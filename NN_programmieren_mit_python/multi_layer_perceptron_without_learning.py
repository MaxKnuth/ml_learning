import numpy as np


def func_id(x):
    return x


# Sigmoid function
def func_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Rectifier function
def func_relu(x):
    return np.maximum(x, 0)


# Netzwerklasse definieren
class MLP(object):
    def __init__(self, n_input_neurons=2, n_hidden_neurons=2, n_output_neurons=1, weights=None, *args, **kwargs):
        # Aktivierungs- und Outputfunktion
        self.f_akt = func_sigmoid
        self.f_out = func_id
        # Anzahl der Neuronen pro Layer
        self.n_input_neurons = n_input_neurons
        self.n_hidden_neurons = n_hidden_neurons
        self.n_output_neurons = n_output_neurons
        # Gewichtsinitialisierung
        self.weights = weights
        W_IH = []
        W_HO = []

        # Hier werden alle Daten zur Netzberechnung abgelegt
        self.network = []

        # Input-Layer + Bias-Neuron: Spalten = o_i
        self.inputLayer = np.zeros((self.n_input_neurons + 1, 1))
        # Bias Neuron Output ist immer +1
        self.inputLayer[0] = 1.0
        # Den Input Layer zum netzwerk hinzufügen
        self.network.append(self.inputLayer)

        # Weights vom input Layer zum Hidden Layer W_IH
        # Nur intiliaisieren falls auch Gewichte vorhanden sind
        if weights:
            W_IH = self.weights[0]
        else:
            W_IH = np.zeros((self.n_hidden_neurons + 1, self.n_input_neurons + 1))
        self.network.append(W_IH)

        # Hidden Layer + Bias Neuron: Spalten net_i, a_i, o_1
        self.hiddenLayer = np.zeros((self.n_hidden_neurons+1, 3))
        # Bias Neuron Output immer 1
        self.hiddenLayer[0] = 1.0
        # Hidden Layer zum Neztwerk hinzufügen
        self.network.append(self.hiddenLayer)

        # Weights vom hidden Layer zum output Layer
        if weights:
            W_HO = self.weights[1]
        else:
            W_HO = np.zeros((self.n_output_neurons + 1, self.n_hidden_neurons + 1))
        self.network.append(W_HO)

        # Output Layer + Bias Neuron, Spalten = net_i, a_i, o_i
        self.outputLayer = np.zeros((self.n_output_neurons + 1, 3))
        self.outputLayer[0] = 0.0  # nicht relevant das dieses gewicht nicht verwendet wird zur Berechnung
        self.network.append(self.outputLayer)

    def print(self):
        print('Multi-Layer Perceptron - Netzwerkarchitektur')
        # Insgesamt 7 Stellen, mit drei Nachkommastellen ausgeben
        np.set_printoptions(
            formatter={'float': lambda x: "{0:7.3f}".format(x)}
        )
        for nn_part in self.network:
            print(nn_part)
            print('--------------=-----------------')

    def predict(self,x):
        # Für die Eingabe x wird die Ausgabe y_hat berechnet
        # Für den vektor x wird eine Vorhersage berechnet und die Matrizenwerte (nicht Gewichte) werden angepasst

        # die Input Werte setzen
        self.network[0][:,0] = x

        # Berechnung des Hidden Layers, erste Wert ist Bias somit mit index 1 starten
        self.network[2][1:,0] = np.dot(self.network[1][1:,:],self.network[0][:,0])
        #a_i
        self.network[2][1:,1] = self.f_akt(self.network[2][1:,0])
        # o_i
        self.network[2][1:, 2] = self.f_out(self.network[2][1:,1])

        # Berechnung des Output Layers, net_1
        self.network[4][1:,0] = np.dot(self.network[3][1:,:],self.network[2][:,2])
        # a_i
        self.network[4][1:, 1] = self.f_akt(self.network[4][1:,0])
        # o_i
        self.network[4][1:, 2] = self.f_out(self.network[4][1:,1])

        return self.network[4][1:, 2]


def main():
    # Initialisierung der Gewichte
    W_IH = np.matrix([[0.0,0.0,0.0],[-10,20.0,20.0],[30,-20.0,-20.0]])
    W_HO = np.matrix([[0.0,0.0,0.0],[-30, 20.0, 20.0]])
    weights = []
    weights.append(W_IH)
    weights.append(W_HO)
    nn = MLP(weights=weights)
    nn.print()

    #Test
    X = np.array([[1.0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 0], [1.0, 0, 0]])
    y = np.array([0, 1.0, 1.0, 0])
    print('Predict:: ')
    for idx,x in enumerate(X):
        print('{} {} -> {}'.format(x, y[idx], nn.predict(x)))

main()

