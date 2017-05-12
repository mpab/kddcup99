# (c) Michael Alderson-Bythell
# Gaussian Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB as c
import kddcup
import classifier

classifier.analyse(c(), kddcup.load_data(), "../analysis/NBAYES")