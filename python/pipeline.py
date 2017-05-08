# (c) Michael Alderson-Bythell
# pipeline.py
# generic pipeline for processing and classification of data

from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from timeit import default_timer as timer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# local utilities
import nltk_cm

#from sklearn.utils.testing import (assert_raises, assert_greater, assert_equal,assert_false, ignore_warnings)

import logging

class Pipeline(object):

    def __init__(self, dataset, log_path, log_name="pipeline"):
        self.dataset = dataset
        self.removed_features = []
        self.low_variance_features = []
        self.log = None
        self.log_file_path = log_path + "/" + log_name
        self.get_logger()
        self.log.info("------------------------ NEW PIPELINE -----------------------------------")

    def get_logger(self):
        if self.log == None:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(Pipeline.__name__)
            fh = logging.FileHandler(self.log_file_path + '.log')
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)
            log_init = True
            self.log = logger
        return self.log

    def scale(scaler=StandardScaler()):
        # Fit only to the training data
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    def remove_low_variance_features(self):   
        if (len(self.low_variance_features)):
            self.log.info("Removing following low-variance features %s", self.low_variance_features)
            self.dataset.data.drop(self.low_variance_features, axis=1, inplace=True)
            self.removed_features = self.low_variance_features
            self.low_variance_features = []

    def find_low_variance_features(self, threshold=0.0, skip_columns=[]):
        """
        Wrapper for sklearn VarianceThreshold for use on pandas dataframes.
        """
        df = self.dataset.data
        #print("Finding low-variance features.")
        #try:
        # get list of all the original df columns
        all_columns = df.columns

        # remove `skip_columns`
        remaining_columns = all_columns.drop(skip_columns)

        # get length of new index
        max_index = len(remaining_columns) - 1

        # get indices for `skip_columns`
        skipped_idx = [all_columns.get_loc(column)
                       for column
                       in skip_columns]

        # adjust insert location by the number of columns removed
        # (for non-zero insertion locations) to keep relative
        # locations intact
        for idx, item in enumerate(skipped_idx):
            if item > max_index:
                diff = item - max_index
                skipped_idx[idx] -= diff
            if item == max_index:
                diff = item - len(skip_columns)
                skipped_idx[idx] -= diff
            if idx == 0:
                skipped_idx[idx] = item

        # get values of `skip_columns`
        skipped_values = df.iloc[:, skipped_idx].values

        # get dataframe values
        X = df.loc[:, remaining_columns].values

        # instantiate VarianceThreshold object
        vt = VarianceThreshold(threshold=threshold)

        # fit vt to data
        vt.fit(X)

        # get the indices of the features that are being kept
        feature_indices = vt.get_support(indices=True)

        # remove low-variance columns from index
        feature_names = [remaining_columns[idx]
                         for idx, _
                         in enumerate(remaining_columns)
                         if idx
                         in feature_indices]

        # get the columns to be removed
        low_variance_features = list(np.setdiff1d(remaining_columns, feature_names))
        self.low_variance_features += low_variance_features
        
        if (len(low_variance_features)):
            self.log.info("find_low_variance_features: {0} features below {1}."
                .format(len(low_variance_features), threshold))
        else:
            self.log.info("find_low_variance_features: none found below threshold %s:", threshold)

        return low_variance_features

    def split(self, splitter=train_test_split, test_size=0.33):
        self.log.info("splitting data into train/test %0.2f/%0.2f using splitter: '%s'",
            1.0 - test_size,
            test_size,
            splitter.__name__)

        self.X_train, self.X_test, self.y_train, self.y_test = splitter(self.dataset.data, self.dataset.target)

    def train(self, model):
        start = timer()
        model.fit(self.X_train, self.y_train)
        elapsed = timer() - start
        self.log.info("train: took %.2fs", elapsed)

    def test(self, model):
        start = timer()
        self.y_predict = model.predict(self.X_test)
        elapsed = timer() - start

        #assert_greater(mlp.score(X_train, y_train), 0.95)
        #assert_equal((y_predict.shape[0], y_predict.dtype.kind), expected_shape_dtype)

        self.log.info("test: took %.2fs", elapsed)

    def test_external_data(self, model, X_test, y_test):
        self.log.info("using external data set for test")
        self.X_test = X_test
        self.y_test = y_test
        self.test(model)

    def assess(self, model):
        self.classification_report = metrics.classification_report(self.y_test, self.y_predict)
        self.accuracy_score = metrics.accuracy_score(self.y_test, self.y_predict)
        # can't get plots from this... had to use the pandas one
        # TODO improve/find new library
        self.confusion_matrix = metrics.confusion_matrix(self.y_test.values, self.y_predict)

    def report(self):
        self.log.info("------------------------- START REPORT ----------------------------------")

        df = self.classification_report_dataframe()
        df.to_csv(self.log_file_path + '.classification_report.csv', index = False)

        self.log.info('classification report:\n%s', self.classification_report)
        self.log.info('accuracy score: %f', self.accuracy_score)

        nltk_cm.report(self.y_test.values.tolist(), self.y_predict.tolist(), self.dataset.target_names, self.log)

        self.log.info("-------------------------- END REPORT -----------------------------------")

    def classification_report_dataframe(self):
        report_data = []
        lines = self.classification_report.split('\n')
        for line in lines[2:-3]:
            row = {}
            row_data = line.split()
            row['class'] = row_data[0]
            row['precision'] = row_data[1]
            row['recall'] = row_data[2]
            row['f1-score'] = row_data[3]
            row['support'] = row_data[4]
            report_data.append(row)
        return pd.DataFrame.from_dict(report_data)

    def graph(self):
        return

        #np.set_printoptions(precision=2)
        # graph confusion matrix
        #self.confusion_matrix.plot()
        #plt.savefig(self.log_file_path + "_confusion_matrix.png", bbox_inches='tight')
        #plt.show()


#categorise targets

#normalise

#select features

#split data

#train

#test

#report

#analyse

#graph