# (c) Michael Alderson-Bythell
# classification & report generation library

from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from timeit import default_timer as timer
import pandas as pd

#--------------------------------------------------------------------------------

def analyse(classifier, dataset, filename):
    a = Analyser(classifier, dataset)
    a.optimise_features()
    a.split()
    a.fit()
    a.predict()
    report(a, filename)

#--------------------------------------------------------------------------------

def report(analyser, filename):
    print analyser.model
    # time info
    print ("optimise: %.2fs, fit: %.2fs, predict: %.2fs" %
        (analyser.optimise_features_elapsed, analyser.fit_elapsed, analyser.predict_elapsed))

    r = metrics.classification_report(analyser.expected, analyser.predicted) 

    df = classification_report_csv_df(r)
    df.to_csv(filename + '.csv', index = False)

    print metrics.confusion_matrix(analyser.expected, analyser.predicted, labels=analyser.dataset.target_names)

    # import matplotlib.pyplot as plt
    # conf = metrics.confusion_matrix(analyser.expected, analyser.predicted, labels=analyser.dataset.target_names)
    # plt.imshow(conf, cmap='binary', interpolation='None')
    # plt.show()

#--------------------------------------------------------------------------------

def classification_report_csv_df(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split()
        row['class'] = row_data[0]
        row['precision'] = row_data[1]
        row['recall'] = row_data[2]
        row['f1_score'] = row_data[3]
        row['support'] = row_data[4]
        report_data.append(row)
    return pd.DataFrame.from_dict(report_data)

#--------------------------------------------------------------------------------
 
class Analyser(object):

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

        self.model = model
        
    def optimise_features(self):
        start = timer()

        #prune bad features
        bad_features = ['duration','protocol_type','service','flag','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins', 'logged_in','num_compromised']
        bad_features += ['root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds']
        bad_features += ['is_host_login','is_guest_login','count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','diff_srv_rate']
        bad_features += ['srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_diff_srv_rate','dst_host_serror_rate']
        bad_features += ['dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']
        for f in bad_features:
            self.dataset.data.drop(f, axis=1, inplace=True)

        #normalize
        #self.dataset.data = preprocessing.normalize(self.dataset.data)
        
        self.optimise_features_elapsed = timer() - start
        return

    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset.data, self.dataset.target, test_size=0.33)

    def fit(self):
        start = timer()
        #print "fitting model"
        self.model.fit(self.X_train, self.y_train)
        self.fit_elapsed = timer() - start

    def predict(self):
        start = timer()
        # make predictions
        self.expected = self.y_test
        #print "making predictions"
        self.predicted = self.model.predict(self.X_test)
        self.predict_elapsed = timer() - start
