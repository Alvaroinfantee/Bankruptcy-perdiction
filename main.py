import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/company-bankruptcy-prediction/data.csv')
df.head()
df.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df['Bankrupt?'].value_counts()
sns.countplot(x=df['Bankrupt?'])
plt.title('Target feature - Bankrupt?')


 #Feature Selection

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, f_classif, SelectKBest

feature_selection=SelectKBest(f_classif,k=30).fit(X,y)
#feat=feature_selection.fit(X_scale,y)
selected_features=X.columns[feature_selection.get_support()]

# Standardize the Independent Variable
scaler=StandardScaler()
X_scale=scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scale, y,test_size=0.3)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
log_reg.score(X_test,y_test)
y_pred=log_reg.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Metrics
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True)

# Support Vector Machine - Classification
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
model.score(X_test,y_test)
svc_predict=model.predict(X_test)

# Metrics
accuracy_score(y_test,svc_predict)
print(classification_report(y_test, svc_predict))
sns.heatmap(confusion_matrix(y_test,svc_predict,), annot=True)
