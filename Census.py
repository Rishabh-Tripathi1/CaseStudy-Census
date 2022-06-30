import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('adult.csv')

rfc = RandomForestClassifier(random_state=1)

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

x = data.drop(['race','relationship','marital.status','income','education.num'], axis=1)
y = data['income']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

rfc.fit(x_train, y_train)

rfcy_predict = rfc.predict(x_test)

print('Random Forest:', accuracy_score(y_test, rfcy_predict))