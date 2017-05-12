# (c) Michael Alderson-Bythell
# Multi-layer Perceptron classifier
from sklearn.neural_network import MLPClassifier
import kddcup
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import distutils.dir_util

import MLP_005_Analyser

log_path = "../analysis/MLP_005/"

distutils.dir_util.mkpath(log_path)

dataset = kddcup.load_data_10_percent()

classifier = MLPClassifier(hidden_layer_sizes=(41, 41, 41), max_iter=500)
pca5 = PCA(n_components=5)
pipeline = make_pipeline(pca5, classifier)

analyser = MLP_005_Analyser.Analyser(log_path, dataset)

log = analyser.get_logger()
log.info("MLPClassifier(hidden_layer_sizes=(41, 41, 41), max_iter=500)")
log.info("no scaling, pca5")

analyser.split()
analyser.train(pipeline)
analyser.test(pipeline)
analyser.assess(pipeline)
analyser.report()
analyser.graph()
