# (c) Michael Alderson-Bythell
# Multi-layer Perceptron classifier
from sklearn.neural_network import MLPClassifier as Model
import kddcup
import pipeline

#clf = MLP(n_hidden=10, n_deep=3, l1_norm=0, drop=0.1, verbose=0)
#ACTIVATION_TYPES = ["identity", "logistic", "tanh", "relu"]#from sklearn.preprocessing import StandardScaler, MinMaxScaler

import distutils.dir_util
log_path = "../analysis/MLP_001/"

distutils.dir_util.mkpath(log_path)

model = Model()
data_10_percent = kddcup.load_data_10_percent()
p = pipeline.Pipeline(data_10_percent, log_path)

p.find_low_variance_features(0);
p.remove_low_variance_features()

p.find_low_variance_features(0.1)

p.split()


p.train(model)
p.test(model)
p.assess(model)
p.report()
p.graph()

#grapher.graph(p)

#data_100_percent = kddcup.load_data_100_percent()
#p.test_external_data(model, data_100_percent.data, data_100_percent.target)
#p.assess(model)
#p.report()


