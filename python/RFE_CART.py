from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
import kddcup
import classifier

rfe = RFE(DecisionTreeClassifier(), 3)
rfe = rfe.fit(kddcup.load_data().data, kddcup.load_data().target)

print(rfe.support_)
print(rfe.ranking_)