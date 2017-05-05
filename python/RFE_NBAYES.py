from sklearn.neural_network import MLPClassifier as c
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
import kddcup
import classifier

rfe = RFE(GaussianNB(), 3)
rfe = rfe.fit(kddcup.load_data().data, kddcup.load_data().target)

print(rfe.support_)
print(rfe.ranking_)