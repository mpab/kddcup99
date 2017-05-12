# (c) Michael Alderson-Bythell
# Support Vector Machine classifier
from sklearn.svm import SVC as c
import kddcup
import classifier

classifier.analyse(c(), kddcup.load_data(), "../analysis/SVM")