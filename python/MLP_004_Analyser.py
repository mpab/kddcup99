import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.grid_search import GridSearchCV
import logging

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import datetime
from timeit import default_timer as timer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

import kddcup

#-----------------------------------------------------------

class Analyser(object):

    def __init__(self, log_path, dataset):
        self.dataset = dataset

        self.creation_timestamp = '{:%Y-%m-%d_%H.%M.%S}'.format(datetime.datetime.now())
        self.instance_name = self.creation_timestamp + '_' + type(self).__name__

        self.log = None
        self.log_file_path = log_path + '/' + self.instance_name
        self.get_logger()

        self.log.info("------------------------ NEW Analyser -----------------------------------")

    def get_logger(self):
        if self.log == None:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(self.instance_name)
            fh = logging.FileHandler(self.log_file_path + '.log')
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)
            log_init = True
            self.log = logger
        return self.log


    def eval(self, classifier, splitter):
        start = timer()
        scores = cross_val_score(classifier, self.dataset.data, self.dataset.target, cv=splitter)
        elapsed = timer() - start
        self.log.info("eval: took %.2fs", elapsed)

        self.log.info(scores)
        self.log.info("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        self.log.info("")

#-----------------------------------------------------------

dataset = kddcup.load_data_10_percent()

analyser = Analyser("../Analysis/MLP_004", dataset)
log = analyser.get_logger()

pca5 = PCA(n_components=5)
pca10 = PCA(n_components=10)

kbest5 = SelectKBest(k=5)
kbest10 = SelectKBest(k=10)

log.info("feature optimisation using combinations of PCA and KBest")
log.info("====================================================================")
log.info("MLPClassifier(hidden_layer_sizes=(41, 41, 41), max_iter=500)")
classifier = MLPClassifier(hidden_layer_sizes=(41, 41, 41), max_iter=500)

log.info("no scaling, pca5")
pipeline = make_pipeline(pca5, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("no scaling, pca10")
pipeline = make_pipeline(pca10, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("no scaling, kbest5")
pipeline = make_pipeline(kbest5, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("no scaling, pca10")
pipeline = make_pipeline(kbest10, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

#------------------------------------------------------------

log.info("StandardScaler, pca5")
make_pipeline(preprocessing.StandardScaler(), pca5, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("StandardScaler, pca10")
make_pipeline(preprocessing.StandardScaler(), pca10, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("StandardScaler, kbest5")
make_pipeline(preprocessing.StandardScaler(), kbest5, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("StandardScaler, kbest10")
make_pipeline(preprocessing.StandardScaler(), kbest10, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

#------------------------------------------------------------

log.info("MinMaxScaler, pca5")
make_pipeline(preprocessing.MinMaxScaler(), pca5, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("MinMaxScaler, pca10")
make_pipeline(preprocessing.MinMaxScaler(), pca10, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("MinMaxScaler, kbest5")
make_pipeline(preprocessing.MinMaxScaler(), kbest5, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("MinMaxScaler, kbest10")
make_pipeline(preprocessing.MinMaxScaler(), kbest10, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

#------------------------------------------------------------

log.info("MaxAbsScaler, pca5")
make_pipeline(preprocessing.MaxAbsScaler(), pca5, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("MaxAbsScaler, pca10")
make_pipeline(preprocessing.MaxAbsScaler(), pca10, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("MaxAbsScaler, kbest5")
make_pipeline(preprocessing.MaxAbsScaler(), kbest5, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("MaxAbsScaler, kbest10")
make_pipeline(preprocessing.MaxAbsScaler(), kbest10, classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))
