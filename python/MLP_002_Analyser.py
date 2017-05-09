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

analyser = Analyser("../Analysis/MLP_002", dataset)
log = analyser.get_logger()

log.info("test scaling and some sample MLP parameters")
log.info("====================================================================")
log.info("MLPClassifier() - default")
log.info("no scaling")
classifier = MLPClassifier()
analyser.eval(classifier, StratifiedKFold(dataset.target, 3))

log.info("StandardScaler")
pipeline = make_pipeline(preprocessing.StandardScaler(), classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("MinMaxScaler")
pipeline = make_pipeline(preprocessing.MinMaxScaler(), classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("MaxAbsScaler")
pipeline = make_pipeline(preprocessing.MaxAbsScaler(), classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("====================================================================")
log.info("MLPClassifier(hidden_layer_sizes=(41, 41, 41), max_iter=500)")
log.info("no scaling")
classifier = MLPClassifier(hidden_layer_sizes=(41, 41, 41), max_iter=500)
analyser.eval(classifier, StratifiedKFold(dataset.target, 3))

log.info("StandardScaler")
pipeline = make_pipeline(preprocessing.StandardScaler(), classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("MinMaxScaler")
pipeline = make_pipeline(preprocessing.MinMaxScaler(), classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("MaxAbsScaler")
c = make_pipeline(preprocessing.MaxAbsScaler(), classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("====================================================================")
log.info("MLPClassifier(hidden_layer_sizes=(41, 943, 23), max_iter=1000)")
log.info("no scaling")
classifier = MLPClassifier(hidden_layer_sizes=(41, 943, 23), max_iter=1000)
analyser.eval(classifier, StratifiedKFold(dataset.target, 3))

log.info("StandardScaler")
pipeline = make_pipeline(preprocessing.StandardScaler(), classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("MinMaxScaler")
pipeline = make_pipeline(preprocessing.MinMaxScaler(), classifier)
analyser.eval(pipeline, StratifiedKFold(dataset.target, 3))

log.info("MaxAbsScaler")
pipeline = make_pipeline(preprocessing.MaxAbsScaler(), classifier)
analyser.eval(pipeline, StratifiedKFoldd(dataset.target, 3))
