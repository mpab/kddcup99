# (c) Michael Alderson-Bythell
# Multi-layer Perceptron classifier
from sklearn.neural_network import MLPClassifier as c
import kddcup
import classifier

classifier.analyse(c(), kddcup.load_data(), "../analysis/MLP")
