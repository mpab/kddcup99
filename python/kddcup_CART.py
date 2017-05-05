# (c) Michael Alderson-Bythell
# Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier as c
import kddcup
import classifier

classifier.analyse(c(), kddcup.load_data(), "../analysis/CART")