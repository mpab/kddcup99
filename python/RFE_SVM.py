from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import kddcup
import classifier

rfe = RFE(SVC(), 3)
rfe = rfe.fit(kddcup.load_data().data, kddcup.load_data().target)

print(rfe.support_)
print(rfe.ranking_)