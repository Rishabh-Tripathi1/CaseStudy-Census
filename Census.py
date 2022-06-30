import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('adult.csv')

logr = LogisticRegression(random_state=0)
rfc = RandomForestClassifier(random_state=1)
dtc = DecisionTreeClassifier(random_state=0)
svm = SVC()
nb = MultinomialNB()
mlp = MLPClassifier(solver='lbfgs',alpha=1e-5, hidden_layer_sizes=(5,2), random_state=0)
gbc = GradientBoostingClassifier(n_estimators=10)

data.drop('native.country', inplace = True, axis=1)

le = LabelEncoder()
data['workclass'] = le.fit_transform(data['workclass'])
data['education'] = le.fit_transform(data['education'])
data['race'] = le.fit_transform(data['race'])
data['occupation'] = le.fit_transform(data['occupation'])
data['marital.status'] = le.fit_transform(data['marital.status'])
data['relationship'] = le.fit_transform(data['relationship'])
data['sex'] = le.fit_transform(data['sex'])
data['income'] = le.fit_transform(data['income'])

data = data.replace('?', np.nan)

x = data.drop(['race','relationship','marital.status','income'], axis=1)
y = data['income']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

logr.fit(x_train, y_train)
rfc.fit(x_train, y_train)
dtc.fit(x_train, y_train)
svm.fit(x_train, y_train)
mlp.fit(x_train, y_train)
gbc.fit(x_train, y_train)
nb.fit(x_train, y_train)

ylogr_predict = logr.predict(x_test)
rfcy_predict = rfc.predict(x_test)
dtcy_predict = dtc.predict(x_test)
svmy_predict = svm.predict(x_test)
mlpy_predict = mlp.predict(x_test)
gbcy_predict = gbc.predict(x_test)
nby_predict = nb.predict(x_test)

print('Logistic:', accuracy_score(y_test, ylogr_predict))
print('Random Forest:', accuracy_score(y_test, rfcy_predict))
print('Decision Tree:', accuracy_score(y_test, dtcy_predict))
print('Support Vector:', accuracy_score(y_test, svmy_predict))
print('MLP:', accuracy_score(y_test,  mlpy_predict))
print('Gradient Boosting:', accuracy_score(y_test,  gbcy_predict))
print('Naive Bayes:', accuracy_score(y_test,  nby_predict))

'''
Accuracy Score:
Logistic: 0.8010133578995855
Random Forest: 0.8374021188392445
Decision Tree: 0.7901120835252572
Support Vector: 0.800706279748196
MLP: 0.2375249500998004
Gradient Boosting: 0.8172884999232305
Naive Bayes: 0.7879625364655305
'''
