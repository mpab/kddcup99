# (c) Michael Alderson-Bythell
# Logistic Regression classifier
from sklearn.linear_model import LogisticRegression as c
import kddcup
import classifier

classifier.analyse(c(), kddcup.load_data(), "../analysis/LOGRES")